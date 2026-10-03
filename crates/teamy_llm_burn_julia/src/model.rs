//! ModernBERT plus the Julia marker head, with dropout disabled for inference.
//!
//! Port reference: Julia a85b127 and Transformers 5.0.0 ModernBERT. Weights are
//! checked by name/rank/shape; no TorchScript, LibTorch or Python executes here.

use std::{collections::BTreeMap, fs::File, ops::Range, path::Path, sync::Arc};

use burn::tensor::{Bool, Int, Tensor, TensorData, activation, backend::Backend};
use eyre::{WrapErr, ensure};
use facet::Facet;
use memmap2::{Mmap, MmapOptions};
use safetensors::{Dtype, SafeTensors};
use teamy_cancellation::CancellationToken;
use teamy_llm_julia_common::EncodedJuliaRequest;

use crate::matmul::JuliaBackend;

#[derive(Facet)]
struct RopeConfig {
    rope_theta: f32,
    rope_type: String,
}

#[derive(Facet)]
struct EncoderConfig {
    model_type: String,
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    global_attn_every_n_layers: usize,
    local_attention: usize,
    norm_eps: f64,
    norm_bias: bool,
    attention_bias: bool,
    mlp_bias: bool,
    hidden_activation: String,
    rope_parameters: BTreeMap<String, RopeConfig>,
}

#[derive(Facet)]
struct JuliaConfig {
    format_version: u32,
    architecture: String,
    weight_dtype: String,
    head_layers: usize,
}

struct Linear<B: Backend> {
    weight: Tensor<B, 2>,
    bias: Option<Tensor<B, 1>>,
}

#[derive(Clone, Copy)]
pub(crate) enum MatmulMode {
    Burn,
    /// Whole-matrix Register-tile matmul, retaining FP32 CUDA operands.
    Fp32Tiled,
    /// Elementwise multiply/reduction avoids accelerated TF32 matrix multiplication.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "Retain the FP32 reduction oracle for qualification tests."
        )
    )]
    Fp32Reference,
}

impl<B: JuliaBackend> Linear<B> {
    fn forward(&self, input: Tensor<B, 3>, mode: MatmulMode) -> Tensor<B, 3> {
        let output = matmul3(
            input,
            self.weight.clone().transpose().unsqueeze::<3>(),
            mode,
        );
        match &self.bias {
            Some(bias) => output + bias.clone().unsqueeze::<3>(),
            None => output,
        }
    }
}

struct Norm<B: Backend> {
    weight: Tensor<B, 1>,
    bias: Option<Tensor<B, 1>>,
    epsilon: f64,
}

impl<B: Backend> Norm<B> {
    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let centered = input.clone() - input.mean_dim(2);
        let variance = (centered.clone() * centered.clone()).mean_dim(2);
        let output =
            centered / (variance + self.epsilon).sqrt() * self.weight.clone().unsqueeze::<3>();
        match &self.bias {
            Some(bias) => output + bias.clone().unsqueeze::<3>(),
            None => output,
        }
    }
}

struct EncoderLayer<B: Backend> {
    attention_norm: Option<Norm<B>>,
    qkv: Linear<B>,
    attention_output: Linear<B>,
    mlp_norm: Norm<B>,
    mlp_input: Linear<B>,
    mlp_output: Linear<B>,
    local: bool,
    rope_theta: f32,
}

struct HeadLayer<B: Backend> {
    norm1: Norm<B>,
    qkv: Linear<B>,
    attention_output: Linear<B>,
    norm2: Norm<B>,
    linear1: Linear<B>,
    linear2: Linear<B>,
}

/// Keep the full validated vocabulary on the host; each request uploads only
/// its ordered FP32 rows. The map owns its storage, so no borrowed safetensors
/// views escape loading and no unaligned typed pointer is constructed.
struct MappedEmbeddings {
    mapped: Arc<Mmap>,
    bytes: Range<usize>,
    rows: usize,
    width: usize,
}

impl MappedEmbeddings {
    fn from_checkpoint(
        mapped: Arc<Mmap>,
        expected: [usize; 2],
        cancellation: &CancellationToken,
    ) -> eyre::Result<Self> {
        cancellation.bail_if_cancelled()?;
        let (header_bytes, metadata) =
            SafeTensors::read_metadata(&mapped).wrap_err("Read Julia embedding tensor index")?;
        let name = "encoder.embeddings.tok_embeddings.weight";
        let info = metadata
            .info(name)
            .ok_or_else(|| eyre::eyre!("Missing Julia tensor {name}"))?;
        ensure!(
            info.shape == expected,
            "Tensor {name} has shape {:?}; expected {expected:?}",
            info.shape
        );
        ensure!(
            info.dtype == Dtype::F32,
            "Tensor {name} must contain float32 values"
        );
        // Safetensors stores an eight-byte header length, then its JSON header,
        // then tensor payloads. Offsets in TensorInfo are relative to the payload.
        let payload = header_bytes
            .checked_add(8)
            .ok_or_else(|| eyre::eyre!("Julia embedding header offset overflow"))?;
        let start = payload
            .checked_add(info.data_offsets.0)
            .ok_or_else(|| eyre::eyre!("Julia embedding start offset overflow"))?;
        let end = payload
            .checked_add(info.data_offsets.1)
            .ok_or_else(|| eyre::eyre!("Julia embedding end offset overflow"))?;
        Self::from_range(mapped, start..end, expected, cancellation)
    }

    fn from_range(
        mapped: Arc<Mmap>,
        bytes: Range<usize>,
        [rows, width]: [usize; 2],
        cancellation: &CancellationToken,
    ) -> eyre::Result<Self> {
        cancellation.bail_if_cancelled()?;
        ensure!(
            rows > 0 && width > 0,
            "Julia embedding shape must be positive"
        );
        let expected_bytes = rows
            .checked_mul(width)
            .and_then(|elements| elements.checked_mul(4))
            .ok_or_else(|| eyre::eyre!("Julia embedding shape overflows its byte bound"))?;
        let data = mapped
            .get(bytes.clone())
            .ok_or_else(|| eyre::eyre!("Julia embedding range is outside the checkpoint"))?;
        ensure!(
            data.len() == expected_bytes && data.len().is_multiple_of(4),
            "Julia embedding payload must contain exactly its declared FP32 values"
        );
        let _span = tracing::debug_span!("julia_embedding_validate", bytes = data.len()).entered();
        // Check every row, including rows not used by the first request. Chunk
        // boundaries retain prompt-independent validation and cancellation.
        for chunk in data.chunks(1024 * 1024) {
            cancellation.bail_if_cancelled()?;
            ensure!(
                crate::finite_scan::all_finite_f32_le(chunk),
                "Julia embedding tensor contains nonfinite values"
            );
        }
        cancellation.bail_if_cancelled()?;
        Ok(Self {
            mapped,
            bytes,
            rows,
            width,
        })
    }

    fn gather(
        &self,
        input_ids: &[i64],
        cancellation: &CancellationToken,
    ) -> eyre::Result<Vec<f32>> {
        cancellation.bail_if_cancelled()?;
        let elements = input_ids
            .len()
            .checked_mul(self.width)
            .ok_or_else(|| eyre::eyre!("Julia embedding request size overflow"))?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(elements)
            .wrap_err("Allocate Julia input embeddings")?;
        for &token in input_ids {
            cancellation.bail_if_cancelled()?;
            let row = usize::try_from(token).wrap_err("Julia token index must be nonnegative")?;
            ensure!(
                row < self.rows,
                "Julia token index is outside the embedding vocabulary"
            );
            let relative = row
                .checked_mul(self.width)
                .and_then(|offset| offset.checked_mul(4))
                .ok_or_else(|| eyre::eyre!("Julia embedding row offset overflow"))?;
            let start = self
                .bytes
                .start
                .checked_add(relative)
                .ok_or_else(|| eyre::eyre!("Julia embedding row start overflow"))?;
            let end = self
                .width
                .checked_mul(4)
                .and_then(|row_bytes| start.checked_add(row_bytes))
                .ok_or_else(|| eyre::eyre!("Julia embedding row end overflow"))?;
            ensure!(
                end <= self.bytes.end,
                "Julia embedding row exceeds its validated range"
            );
            let data = self
                .mapped
                .get(start..end)
                .ok_or_else(|| eyre::eyre!("Julia embedding row is outside the checkpoint"))?;
            values.extend(
                data.chunks_exact(4)
                    .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk"))),
            );
        }
        cancellation.bail_if_cancelled()?;
        Ok(values)
    }
}

pub(crate) struct JuliaModel<B: Backend> {
    device: B::Device,
    config: EncoderConfig,
    embeddings: MappedEmbeddings,
    embedding_norm: Norm<B>,
    layers: Vec<EncoderLayer<B>>,
    final_norm: Norm<B>,
    type_embeddings: Tensor<B, 2>,
    head: Vec<HeadLayer<B>>,
    scorer_norm: Norm<B>,
    scorer_hidden: Linear<B>,
    scorer_output: Linear<B>,
    matmul_mode: MatmulMode,
    // The retained read handle denies writes/deletion on Windows for the entire
    // mapped-table lifetime. Fields above (including the map) drop before this.
    _weights_file: File,
}

/// Backend-independent, validated host data. File ownership prevents mutation
/// throughout preparation and transfers with the mapping into the final model.
pub(crate) struct PreparedCheckpoint {
    config: EncoderConfig,
    julia: JuliaConfig,
    mapped: Arc<Mmap>,
    embeddings: MappedEmbeddings,
    file: File,
}

impl PreparedCheckpoint {
    #[tracing::instrument(level = "debug", name = "julia_checkpoint_prepare", skip_all)]
    pub(crate) fn load(model_dir: &Path, cancellation: &CancellationToken) -> eyre::Result<Self> {
        cancellation.bail_if_cancelled()?;
        let config: EncoderConfig = facet_json::from_str(&std::fs::read_to_string(
            model_dir.join("encoder/config.json"),
        )?)?;
        let julia: JuliaConfig = facet_json::from_str(&std::fs::read_to_string(
            model_dir.join("julia_config.json"),
        )?)?;
        ensure!(
            julia.format_version == 1
                && julia.architecture == "JuliaDecisionModel"
                && julia.weight_dtype == "float32",
            "Unsupported Julia checkpoint format/architecture/dtype"
        );
        ensure!(
            julia.head_layers == 2,
            "This Julia port requires the pinned two-layer decision head"
        );
        ensure!(
            config.model_type == "modernbert"
                && config.hidden_size == 384
                && config.intermediate_size == 1152
                && config.num_attention_heads == 6
                && config.num_hidden_layers == 22
                && config.vocab_size == 256000
                && config.max_position_embeddings == 8192
                && config.global_attn_every_n_layers == 3
                && config.local_attention == 128
                && !config.norm_bias
                && !config.attention_bias
                && !config.mlp_bias
                && config.hidden_activation == "gelu"
                && (config.norm_eps - 1e-5).abs() < 1e-12,
            "Encoder configuration does not match the pinned Julia architecture"
        );
        for kind in ["full_attention", "sliding_attention"] {
            let rope = config
                .rope_parameters
                .get(kind)
                .ok_or_else(|| eyre::eyre!("Missing {kind} rotary configuration"))?;
            ensure!(
                rope.rope_type == "default" && rope.rope_theta == 160000.0,
                "Unsupported rotary configuration"
            );
        }
        cancellation.bail_if_cancelled()?;
        let mut options = std::fs::OpenOptions::new();
        options.read(true);
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            // FILE_SHARE_READ permits other readers while excluding writers and
            // deletion/replacement of the checkpoint while its map is retained.
            options.share_mode(1);
        }
        let file = options
            .open(model_dir.join("model.safetensors"))
            .wrap_err("Open immutable Julia safetensors")?;
        ensure!(
            file.metadata()?.len() <= 1_000_000_000,
            "Julia checkpoint exceeds the supported file bound"
        );
        // SAFETY: the selected acquisition file is immutable while the model is
        // resident. Windows enforces this through the retained read-only handle;
        // other platforms retain the existing immutable-acquisition requirement.
        // Arc owns the mapping; only checked byte slices are retained and decoded.
        let mapped =
            Arc::new(unsafe { MmapOptions::new().map(&file) }.wrap_err("Map Julia safetensors")?);
        // Validate the complete index now. Rebuild its small borrowed view on
        // the numerical thread, avoiding a self-referential checkpoint object.
        let _ = SafeTensors::deserialize(&mapped).wrap_err("Read Julia tensor index")?;
        let width = config.hidden_size;
        let embeddings = MappedEmbeddings::from_checkpoint(
            Arc::clone(&mapped),
            [config.vocab_size, width],
            cancellation,
        )?;
        Ok(Self {
            config,
            julia,
            mapped,
            embeddings,
            file,
        })
    }
}

impl<B: JuliaBackend> JuliaModel<B> {
    pub(crate) fn load(
        model_dir: &Path,
        device: B::Device,
        cancellation: &CancellationToken,
        matmul_mode: MatmulMode,
    ) -> eyre::Result<Self> {
        Self::load_prepared(
            PreparedCheckpoint::load(model_dir, cancellation)?,
            device,
            cancellation,
            matmul_mode,
        )
    }

    pub(crate) fn load_prepared(
        checkpoint: PreparedCheckpoint,
        device: B::Device,
        cancellation: &CancellationToken,
        matmul_mode: MatmulMode,
    ) -> eyre::Result<Self> {
        cancellation.bail_if_cancelled()?;
        let PreparedCheckpoint {
            config,
            julia,
            mapped,
            embeddings,
            file,
        } = checkpoint;
        let tensors = SafeTensors::deserialize(&mapped).wrap_err("Read Julia tensor index")?;
        let loader = Loader {
            tensors: &tensors,
            device: &device,
            cancellation,
        };
        let width = config.hidden_size;
        let embedding_norm =
            loader.norm("encoder.embeddings.norm", width, false, config.norm_eps)?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            cancellation.bail_if_cancelled()?;
            let prefix = format!("encoder.layers.{index}");
            let local = index % config.global_attn_every_n_layers != 0;
            layers.push(EncoderLayer {
                attention_norm: if index == 0 {
                    None
                } else {
                    Some(loader.norm(
                        &format!("{prefix}.attn_norm"),
                        width,
                        false,
                        config.norm_eps,
                    )?)
                },
                qkv: loader.linear(&format!("{prefix}.attn.Wqkv"), width * 3, width, false)?,
                attention_output: loader.linear(
                    &format!("{prefix}.attn.Wo"),
                    width,
                    width,
                    false,
                )?,
                mlp_norm: loader.norm(
                    &format!("{prefix}.mlp_norm"),
                    width,
                    false,
                    config.norm_eps,
                )?,
                mlp_input: loader.linear(
                    &format!("{prefix}.mlp.Wi"),
                    config.intermediate_size * 2,
                    width,
                    false,
                )?,
                mlp_output: loader.linear(
                    &format!("{prefix}.mlp.Wo"),
                    width,
                    config.intermediate_size,
                    false,
                )?,
                local,
                rope_theta: config.rope_parameters[if local {
                    "sliding_attention"
                } else {
                    "full_attention"
                }]
                .rope_theta,
            });
        }
        let final_norm = loader.norm("encoder.final_norm", width, false, config.norm_eps)?;
        let type_embeddings = loader.tensor("type_emb.weight", [3, width])?;
        let mut head = Vec::with_capacity(julia.head_layers);
        for index in 0..julia.head_layers {
            let prefix = format!("head.layers.{index}");
            head.push(HeadLayer {
                norm1: loader.norm(&format!("{prefix}.norm1"), width, true, 1e-5)?,
                qkv: Linear {
                    weight: loader.tensor(
                        &format!("{prefix}.self_attn.in_proj_weight"),
                        [width * 3, width],
                    )?,
                    bias: Some(
                        loader.tensor(&format!("{prefix}.self_attn.in_proj_bias"), [width * 3])?,
                    ),
                },
                attention_output: loader.linear(
                    &format!("{prefix}.self_attn.out_proj"),
                    width,
                    width,
                    true,
                )?,
                norm2: loader.norm(&format!("{prefix}.norm2"), width, true, 1e-5)?,
                linear1: loader.linear(&format!("{prefix}.linear1"), width * 4, width, true)?,
                linear2: loader.linear(&format!("{prefix}.linear2"), width, width * 4, true)?,
            });
        }
        let scorer_norm = loader.norm("scorer.0", width, true, 1e-5)?;
        let scorer_hidden = loader.linear("scorer.1", width, width, true)?;
        let scorer_output = loader.linear("scorer.3", 1, width, true)?;
        Ok(Self {
            device,
            config,
            embeddings,
            embedding_norm,
            layers,
            final_norm,
            type_embeddings,
            head,
            scorer_norm,
            scorer_hidden,
            scorer_output,
            matmul_mode,
            _weights_file: file,
        })
    }

    pub(crate) fn infer(
        &self,
        encoded: &EncodedJuliaRequest,
        cancellation: &CancellationToken,
    ) -> eyre::Result<Vec<f32>> {
        encoded.validate()?;
        let length = encoded.token_count();
        ensure!(
            length <= self.config.max_position_embeddings,
            "Julia sequence exceeds model capacity"
        );
        let width = self.config.hidden_size;
        let heads = self.config.num_attention_heads;
        let head_width = width / heads;
        let embedding_span = tracing::debug_span!(
            "julia_embedding_gather",
            tokens = length,
            upload_bytes = length * width * 4
        )
        .entered();
        let input_embeddings = Tensor::<B, 3>::from_data(
            TensorData::new(
                self.embeddings.gather(&encoded.input_ids, cancellation)?,
                [1, length, width],
            ),
            &self.device,
        );
        drop(embedding_span);
        let mut hidden = self.embedding_norm.forward(input_embeddings);
        let global_mask = attention_mask::<B>(&encoded.attention_mask, None, &self.device);
        let local_mask = attention_mask::<B>(
            &encoded.attention_mask,
            Some(self.config.local_attention / 2),
            &self.device,
        );
        let (cos, sin) =
            rotary_tables::<B>(length, head_width, self.layers[0].rope_theta, &self.device);
        for (index, layer) in self.layers.iter().enumerate() {
            cancellation.bail_if_cancelled()?;
            let _span = tracing::debug_span!("julia_encoder_layer", index).entered();
            let normalized = match &layer.attention_norm {
                Some(norm) => norm.forward(hidden.clone()),
                None => hidden.clone(),
            };
            let attended = attention(
                layer.qkv.forward(normalized, self.matmul_mode),
                heads,
                if layer.local {
                    local_mask.clone()
                } else {
                    global_mask.clone()
                },
                Some((&cos, &sin)),
                self.matmul_mode,
            );
            hidden = hidden + layer.attention_output.forward(attended, self.matmul_mode);
            let projected = layer
                .mlp_input
                .forward(layer.mlp_norm.forward(hidden.clone()), self.matmul_mode);
            let intermediate = self.config.intermediate_size;
            let input = projected.clone().slice([0..1, 0..length, 0..intermediate]);
            let gate = projected.slice([0..1, 0..length, intermediate..intermediate * 2]);
            hidden = hidden
                + layer
                    .mlp_output
                    .forward(activation::gelu(input) * gate, self.matmul_mode);
        }
        hidden = self.final_norm.forward(hidden);
        let type_id =
            Tensor::<B, 1, Int>::from_data(TensorData::new(vec![encoded.qtype], [1]), &self.device);
        hidden = hidden
            + self
                .type_embeddings
                .clone()
                .select(0, type_id)
                .unsqueeze::<3>();
        for (index, layer) in self.head.iter().enumerate() {
            cancellation.bail_if_cancelled()?;
            let _span = tracing::debug_span!("julia_decision_layer", index).entered();
            let attended = attention(
                layer
                    .qkv
                    .forward(layer.norm1.forward(hidden.clone()), self.matmul_mode),
                heads,
                global_mask.clone(),
                None,
                self.matmul_mode,
            );
            hidden = hidden + layer.attention_output.forward(attended, self.matmul_mode);
            let intermediate = activation::relu(
                layer
                    .linear1
                    .forward(layer.norm2.forward(hidden.clone()), self.matmul_mode),
            );
            hidden = hidden + layer.linear2.forward(intermediate, self.matmul_mode);
        }
        cancellation.bail_if_cancelled()?;
        let positions = Tensor::<B, 1, Int>::from_data(
            TensorData::new(encoded.marker_positions.clone(), [encoded.option_count()]),
            &self.device,
        );
        let selected = hidden.select(1, positions);
        let selected = self.scorer_norm.forward(selected);
        let selected = activation::gelu(self.scorer_hidden.forward(selected, self.matmul_mode));
        let logits = self
            .scorer_output
            .forward(selected, self.matmul_mode)
            .reshape([encoded.option_count()]);
        logits
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|error| eyre::eyre!("Read Julia logits: {error:?}"))
    }
}

struct Loader<'a, B: Backend> {
    tensors: &'a SafeTensors<'a>,
    device: &'a B::Device,
    cancellation: &'a CancellationToken,
}

impl<B: Backend> Loader<'_, B> {
    fn tensor<const D: usize>(
        &self,
        name: &str,
        expected: [usize; D],
    ) -> eyre::Result<Tensor<B, D>> {
        self.cancellation.bail_if_cancelled()?;
        let view = self
            .tensors
            .tensor(name)
            .wrap_err_with(|| format!("Missing Julia tensor {name}"))?;
        ensure!(
            view.shape() == expected,
            "Tensor {name} has shape {:?}; expected {expected:?}",
            view.shape()
        );
        ensure!(
            view.dtype() == Dtype::F32,
            "Tensor {name} must contain float32 values"
        );
        ensure!(
            crate::finite_scan::all_finite_f32_le(view.data()),
            "Tensor {name} contains nonfinite values"
        );
        let values = view
            .data()
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
            .collect::<Vec<_>>();
        Ok(Tensor::from_data(
            TensorData::new(values, expected),
            self.device,
        ))
    }

    fn norm(&self, prefix: &str, width: usize, bias: bool, epsilon: f64) -> eyre::Result<Norm<B>> {
        Ok(Norm {
            weight: self.tensor(&format!("{prefix}.weight"), [width])?,
            bias: if bias {
                Some(self.tensor(&format!("{prefix}.bias"), [width])?)
            } else {
                None
            },
            epsilon,
        })
    }

    fn linear(
        &self,
        prefix: &str,
        output: usize,
        input: usize,
        bias: bool,
    ) -> eyre::Result<Linear<B>> {
        Ok(Linear {
            weight: self.tensor(&format!("{prefix}.weight"), [output, input])?,
            bias: if bias {
                Some(self.tensor(&format!("{prefix}.bias"), [output])?)
            } else {
                None
            },
        })
    }
}

fn attention<B: JuliaBackend>(
    qkv: Tensor<B, 3>,
    heads: usize,
    mask: Tensor<B, 4, Bool>,
    rotary: Option<(&Tensor<B, 4>, &Tensor<B, 4>)>,
    mode: MatmulMode,
) -> Tensor<B, 3> {
    let [batch, length, packed] = qkv.dims();
    let width = packed / 3;
    let head_width = width / heads;
    let split = |start| {
        qkv.clone()
            .slice([0..batch, 0..length, start..start + width])
            .reshape([batch, length, heads, head_width])
            .swap_dims(1, 2)
    };
    let mut query = split(0);
    let mut key = split(width);
    let value = split(width * 2);
    if let Some((cos, sin)) = rotary {
        query = apply_rotary(query, cos, sin);
        key = apply_rotary(key, cos, sin);
    }
    let scores = (matmul4(query, key.swap_dims(2, 3), mode) / (head_width as f64).sqrt())
        .mask_fill(mask, f32::NEG_INFINITY);
    matmul4(activation::softmax(scores, 3), value, mode)
        .swap_dims(1, 2)
        .reshape([batch, length, width])
}

fn matmul3<B: JuliaBackend>(
    lhs: Tensor<B, 3>,
    rhs: Tensor<B, 3>,
    mode: MatmulMode,
) -> Tensor<B, 3> {
    let [batch_l, rows, inner] = lhs.dims();
    let [batch_r, _, columns] = rhs.dims();
    matmul4(
        lhs.reshape([batch_l, 1, rows, inner]),
        rhs.reshape([batch_r, 1, inner, columns]),
        mode,
    )
    .reshape([batch_l.max(batch_r), rows, columns])
}

/// Correctness configuration using ordinary F32 multiplication and reduction kernels.
/// All matrix dimensions are chunked: the largest multiplication temporary contains
/// batch * heads * 32 * 128 * 32 floats. No accelerated Tensor::matmul runs in this mode.
fn matmul4<B: JuliaBackend>(
    lhs: Tensor<B, 4>,
    rhs: Tensor<B, 4>,
    mode: MatmulMode,
) -> Tensor<B, 4> {
    if matches!(mode, MatmulMode::Burn) {
        return lhs.matmul(rhs);
    }
    if matches!(mode, MatmulMode::Fp32Tiled) {
        return B::fp32_matmul(lhs, rhs);
    }
    let [batch_l, heads_l, rows, inner] = lhs.dims();
    let [batch_r, heads_r, inner_r, columns] = rhs.dims();
    assert_eq!(inner, inner_r, "Matrix inner dimensions must match");
    let mut row_blocks = Vec::new();
    for row in (0..rows).step_by(32) {
        let row_end = (row + 32).min(rows);
        let mut column_blocks = Vec::new();
        for column in (0..columns).step_by(32) {
            let column_end = (column + 32).min(columns);
            let mut accumulated = None;
            for inner_start in (0..inner).step_by(128) {
                let inner_end = (inner_start + 128).min(inner);
                let left = lhs
                    .clone()
                    .slice([0..batch_l, 0..heads_l, row..row_end, inner_start..inner_end])
                    .unsqueeze_dim::<5>(4);
                let right = rhs
                    .clone()
                    .slice([
                        0..batch_r,
                        0..heads_r,
                        inner_start..inner_end,
                        column..column_end,
                    ])
                    .unsqueeze_dim::<5>(2);
                let partial = (left * right).sum_dim(3).squeeze_dim::<4>(3);
                accumulated = Some(match accumulated {
                    Some(previous) => previous + partial,
                    None => partial,
                });
            }
            column_blocks
                .push(accumulated.expect("Model matrices have a positive inner dimension"));
        }
        row_blocks.push(Tensor::cat(column_blocks, 3));
    }
    Tensor::cat(row_blocks, 2)
}

fn apply_rotary<B: Backend>(
    input: Tensor<B, 4>,
    cos: &Tensor<B, 4>,
    sin: &Tensor<B, 4>,
) -> Tensor<B, 4> {
    let [batch, heads, length, width] = input.dims();
    let half = width / 2;
    let first = input
        .clone()
        .slice([0..batch, 0..heads, 0..length, 0..half]);
    let second = input
        .clone()
        .slice([0..batch, 0..heads, 0..length, half..width]);
    let rotated = Tensor::cat(vec![-second, first], 3);
    input * cos.clone() + rotated * sin.clone()
}

fn rotary_tables<B: Backend>(
    length: usize,
    width: usize,
    theta: f32,
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let frequencies = (0..width / 2)
        .map(|index| 1.0 / theta.powf((index * 2) as f32 / width as f32))
        .collect::<Vec<_>>();
    let mut cosine = Vec::with_capacity(length * width);
    let mut sine = Vec::with_capacity(length * width);
    for position in 0..length {
        for _ in 0..2 {
            for &frequency in &frequencies {
                let angle = position as f32 * frequency;
                cosine.push(angle.cos());
                sine.push(angle.sin());
            }
        }
    }
    (
        Tensor::from_data(TensorData::new(cosine, [1, 1, length, width]), device),
        Tensor::from_data(TensorData::new(sine, [1, 1, length, width]), device),
    )
}

fn attention_mask<B: Backend>(
    padding: &[i64],
    radius: Option<usize>,
    device: &B::Device,
) -> Tensor<B, 4, Bool> {
    let length = padding.len();
    let mut masked = Vec::with_capacity(length * length);
    for query in 0..length {
        for (key, &present) in padding.iter().enumerate() {
            masked.push(present == 0 || radius.is_some_and(|radius| query.abs_diff(key) > radius));
        }
    }
    Tensor::from_data(TensorData::new(masked, [1, 1, length, length]), device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    type Cpu = NdArray<f32>;

    fn read_only_fixture(bytes: &[u8]) -> Arc<Mmap> {
        let mut mapped = memmap2::MmapMut::map_anon(bytes.len()).unwrap();
        mapped.copy_from_slice(bytes);
        Arc::new(mapped.make_read_only().unwrap())
    }

    fn embedding_fixture(values: &[f32], shape: [usize; 2]) -> Arc<Mmap> {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let view =
            safetensors::tensor::TensorView::new(Dtype::F32, shape.to_vec(), &bytes).unwrap();
        let checkpoint = safetensors::tensor::serialize(
            [("encoder.embeddings.tok_embeddings.weight", view)],
            None,
        )
        .unwrap();
        read_only_fixture(&checkpoint)
    }

    #[test]
    fn mapped_embedding_gather_preserves_row_order_duplicates_padding_and_f32_bits() {
        let cancellation = CancellationToken::new();
        let embeddings = MappedEmbeddings::from_checkpoint(
            embedding_fixture(&[-0.0, 1.25, 2.0, -3.5, 4.0, 5.0], [3, 2]),
            [3, 2],
            &cancellation,
        )
        .unwrap();
        let actual = embeddings.gather(&[2, 0, 2, 1], &cancellation).unwrap();
        let expected = [4.0_f32, 5.0, -0.0, 1.25, 4.0, 5.0, 2.0, -3.5];
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        let uploaded =
            Tensor::<Cpu, 3>::from_data(TensorData::new(actual, [1, 4, 2]), &Default::default());
        assert_eq!(uploaded.dims(), [1, 4, 2]);
        assert_eq!(uploaded.into_data().to_vec::<f32>().unwrap(), expected);
    }

    #[test]
    fn mapped_embedding_gather_rejects_negative_out_of_vocabulary_and_huge_indices() {
        let cancellation = CancellationToken::new();
        let embeddings = MappedEmbeddings::from_checkpoint(
            embedding_fixture(&[1.0, 2.0, 3.0, 4.0], [2, 2]),
            [2, 2],
            &cancellation,
        )
        .unwrap();
        for index in [-1, 2, i64::MAX] {
            assert!(embeddings.gather(&[index], &cancellation).is_err());
        }
    }

    #[test]
    fn mapped_embedding_load_rejects_nonfinite_values_in_unused_rows() {
        let cancellation = CancellationToken::new();
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let result = MappedEmbeddings::from_checkpoint(
                embedding_fixture(&[1.0, 2.0, 3.0, value], [2, 2]),
                [2, 2],
                &cancellation,
            );
            assert!(result.is_err_and(|error| error.to_string().contains("nonfinite")));
        }
    }

    #[test]
    fn mapped_embedding_load_checks_shape_dtype_payload_and_byte_bounds() {
        let cancellation = CancellationToken::new();
        let fixture = embedding_fixture(&[1.0, 2.0, 3.0, 4.0], [2, 2]);
        assert!(
            MappedEmbeddings::from_checkpoint(Arc::clone(&fixture), [4, 1], &cancellation).is_err()
        );
        let bytes = [0_u8; 8];
        let integer_view =
            safetensors::tensor::TensorView::new(Dtype::I32, vec![2, 1], &bytes).unwrap();
        let integer_checkpoint = safetensors::tensor::serialize(
            [("encoder.embeddings.tok_embeddings.weight", integer_view)],
            None,
        )
        .unwrap();
        assert!(
            MappedEmbeddings::from_checkpoint(
                read_only_fixture(&integer_checkpoint),
                [2, 1],
                &cancellation,
            )
            .is_err()
        );
        for (range, shape) in [(0..3, [1, 1]), (0..9, [2, 1]), (0..8, [0, 2])] {
            assert!(
                MappedEmbeddings::from_range(
                    read_only_fixture(&bytes),
                    range,
                    shape,
                    &cancellation
                )
                .is_err()
            );
        }
        assert!(
            MappedEmbeddings::from_range(
                read_only_fixture(&bytes),
                0..8,
                [usize::MAX, 2],
                &cancellation,
            )
            .is_err()
        );
    }

    #[test]
    fn mapped_embedding_load_and_gather_respect_cancellation() {
        let cancellation = CancellationToken::new();
        let fixture = embedding_fixture(&[1.0, 2.0], [2, 1]);
        let embeddings =
            MappedEmbeddings::from_checkpoint(Arc::clone(&fixture), [2, 1], &cancellation).unwrap();
        cancellation.request_cancel("fixture stop");
        assert!(MappedEmbeddings::from_checkpoint(fixture, [2, 1], &cancellation).is_err());
        assert!(embeddings.gather(&[0, 1], &cancellation).is_err());
    }

    #[test]
    fn rotary_uses_two_halves_and_preserves_norm() {
        let device = Default::default();
        let values = Tensor::<Cpu, 4>::from_data(
            TensorData::new(vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0], [1, 1, 2, 4]),
            &device,
        );
        let (cos, sin) = rotary_tables::<Cpu>(2, 4, 160000.0, &device);
        let result = apply_rotary(values, &cos, &sin)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(&result[..4], &[1.0, 2.0, 3.0, 4.0]);
        let actual = result[4..].iter().map(|x| x * x).sum::<f32>();
        assert!((actual - 30.0).abs() < 1e-4);
    }

    #[test]
    fn local_mask_includes_boundary_and_masks_padding() {
        let device = Default::default();
        let mask = attention_mask::<Cpu>(&[1, 1, 1, 0], Some(1), &device)
            .into_data()
            .to_vec::<bool>()
            .unwrap();
        assert_eq!(&mask[..4], &[false, false, true, true]);
        assert_eq!(&mask[4..8], &[false, false, false, true]);
    }

    #[test]
    fn layer_norm_matches_independent_population_variance() {
        let device = Default::default();
        let norm = Norm::<Cpu> {
            weight: Tensor::from_data([2.0, 3.0], &device),
            bias: Some(Tensor::from_data([1.0, -1.0], &device)),
            epsilon: 1e-5,
        };
        let values =
            Tensor::<Cpu, 3>::from_data(TensorData::new(vec![1.0, 3.0], [1, 1, 2]), &device);
        let actual = norm.forward(values).into_data().to_vec::<f32>().unwrap();
        assert!((actual[0] - (-2.0 / 1.00001_f32.sqrt() + 1.0)).abs() < 1e-5);
        assert!((actual[1] - (3.0 / 1.00001_f32.sqrt() - 1.0)).abs() < 1e-5);
    }

    #[test]
    fn fp32_matmul_matches_scalar_reference_across_chunks_and_head_broadcast() {
        let device = Default::default();
        let (heads, rows, inner, columns) = (2, 33, 129, 35);
        let left = (0..heads * rows * inner)
            .map(|i| (i % 23) as f32 / 23.0 - 0.5)
            .collect::<Vec<_>>();
        let right = (0..inner * columns)
            .map(|i| (i % 19) as f32 / 19.0 - 0.5)
            .collect::<Vec<_>>();
        let actual = matmul4(
            Tensor::<Cpu, 4>::from_data(
                TensorData::new(left.clone(), [1, heads, rows, inner]),
                &device,
            ),
            Tensor::<Cpu, 4>::from_data(
                TensorData::new(right.clone(), [1, 1, inner, columns]),
                &device,
            ),
            MatmulMode::Fp32Reference,
        )
        .into_data()
        .to_vec::<f32>()
        .unwrap();
        for head in 0..heads {
            for row in 0..rows {
                for column in 0..columns {
                    let expected = (0..inner)
                        .map(|k| {
                            left[(head * rows + row) * inner + k] * right[k * columns + column]
                        })
                        .sum::<f32>();
                    let observed = actual[(head * rows + row) * columns + column];
                    assert!((observed - expected).abs() <= 1e-5 + expected.abs() * 1e-5);
                }
            }
        }
    }

    #[test]
    fn fp32_tiled_matmul_matches_transposed_weights_with_tails_and_head_broadcast() {
        let device = Default::default();
        let (heads, rows, inner, columns) = (2, 33, 129, 35);
        let left = (0..heads * rows * inner)
            .map(|i| (i % 23) as f32 / 23.0 - 0.5)
            .collect::<Vec<_>>();
        let weights = (0..columns * inner)
            .map(|i| (i % 19) as f32 / 19.0 - 0.5)
            .collect::<Vec<_>>();
        let actual = matmul4(
            Tensor::<Cpu, 4>::from_data(
                TensorData::new(left.clone(), [1, heads, rows, inner]),
                &device,
            ),
            Tensor::<Cpu, 4>::from_data(
                TensorData::new(weights.clone(), [1, 1, columns, inner]),
                &device,
            )
            .swap_dims(2, 3),
            MatmulMode::Fp32Tiled,
        )
        .into_data()
        .to_vec::<f32>()
        .unwrap();
        for head in 0..heads {
            for row in 0..rows {
                for column in 0..columns {
                    let expected = (0..inner)
                        .map(|k| {
                            left[(head * rows + row) * inner + k] * weights[column * inner + k]
                        })
                        .sum::<f32>();
                    let observed = actual[(head * rows + row) * columns + column];
                    assert!((observed - expected).abs() <= 1e-5 + expected.abs() * 1e-5);
                }
            }
        }
    }
}
