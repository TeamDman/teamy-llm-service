//! ModernBERT plus the Julia marker head, with dropout disabled for inference.
//!
//! Port reference: Julia a85b127 and Transformers 5.0.0 ModernBERT. Weights are
//! checked by name/rank/shape; no TorchScript, LibTorch or Python executes here.

use std::{collections::BTreeMap, fs::File, path::Path};

use burn::tensor::{Bool, Int, Tensor, TensorData, activation, backend::Backend, module};
use eyre::{WrapErr, ensure};
use facet::Facet;
use memmap2::MmapOptions;
use safetensors::{Dtype, SafeTensors};
use teamy_cancellation::CancellationToken;
use teamy_llm_julia_common::EncodedJuliaRequest;

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
    /// Elementwise multiply/reduction avoids accelerated TF32 matrix multiplication.
    Fp32Reference,
}

impl<B: Backend> Linear<B> {
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

pub(crate) struct JuliaModel<B: Backend> {
    device: B::Device,
    config: EncoderConfig,
    embeddings: Tensor<B, 2>,
    embedding_norm: Norm<B>,
    layers: Vec<EncoderLayer<B>>,
    final_norm: Norm<B>,
    type_embeddings: Tensor<B, 2>,
    head: Vec<HeadLayer<B>>,
    scorer_norm: Norm<B>,
    scorer_hidden: Linear<B>,
    scorer_output: Linear<B>,
    matmul_mode: MatmulMode,
}

impl<B: Backend> JuliaModel<B> {
    pub(crate) fn load(
        model_dir: &Path,
        device: B::Device,
        cancellation: &CancellationToken,
        matmul_mode: MatmulMode,
    ) -> eyre::Result<Self> {
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
        let file =
            File::open(model_dir.join("model.safetensors")).wrap_err("Open Julia safetensors")?;
        ensure!(
            file.metadata()?.len() <= 1_000_000_000,
            "Julia checkpoint exceeds the supported file bound"
        );
        // SAFETY: the selected immutable acquisition file is read-only. The map remains alive
        // until all views have been copied into backend-owned tensors, and never escapes loading.
        let mapped = unsafe { MmapOptions::new().map(&file) }.wrap_err("Map Julia safetensors")?;
        let tensors = SafeTensors::deserialize(&mapped).wrap_err("Read Julia tensor index")?;
        let loader = Loader {
            tensors: &tensors,
            device: &device,
            cancellation,
        };
        let width = config.hidden_size;
        let embeddings = loader.tensor(
            "encoder.embeddings.tok_embeddings.weight",
            [config.vocab_size, width],
        )?;
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
        let token_ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(encoded.input_ids.clone(), [1, length]),
            &self.device,
        );
        let mut hidden = self
            .embedding_norm
            .forward(module::embedding(self.embeddings.clone(), token_ids));
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
        let values = view
            .data()
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
            .collect::<Vec<_>>();
        ensure!(
            values.iter().all(|value| value.is_finite()),
            "Tensor {name} contains nonfinite values"
        );
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

fn attention<B: Backend>(
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

fn matmul3<B: Backend>(lhs: Tensor<B, 3>, rhs: Tensor<B, 3>, mode: MatmulMode) -> Tensor<B, 3> {
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
fn matmul4<B: Backend>(lhs: Tensor<B, 4>, rhs: Tensor<B, 4>, mode: MatmulMode) -> Tensor<B, 4> {
    if matches!(mode, MatmulMode::Burn) {
        return lhs.matmul(rhs);
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
}
