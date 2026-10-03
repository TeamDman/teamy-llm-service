//! Julia-1's numerical operations in Rust over the existing Burn backend.
//! Native CUDA is the explicit production target; NdArray is an explicit debug/reference device.

mod cuda_runtime;
mod finite_scan;
mod kernel_cache;
mod matmul;
mod model;

use std::{path::Path, time::Instant};

use burn::backend::{Cuda, NdArray, cuda::CudaDevice};
use facet::Facet;
use teamy_cancellation::CancellationToken;
use teamy_llm_core::finite_choice::{
    BackendCapabilities, FiniteChoiceError, FiniteChoiceErrorKind, FiniteChoiceProvider,
    FiniteChoiceResult, ValidatedFiniteChoiceRequest,
};
use teamy_llm_julia_common::{
    EncodedJuliaRequest, EncodingPolicy, JULIA_MODEL_REVISION, JuliaEncoder, julia_capabilities,
};

type CudaBackend = Cuda<f32, i32>;
type CpuBackend = NdArray<f32>;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Facet)]
pub enum NativeDevice {
    Cuda,
    Cpu,
}

#[derive(Clone, Debug, Facet)]
pub struct NativeJuliaMetadata {
    pub backend: String,
    pub device: NativeDevice,
    pub precision: String,
    /// Revision defining this port, not an assertion about an unchecked file's origin.
    pub source_reference_revision: String,
    pub weights_bytes: u64,
    pub load_millis: u64,
}

enum Runtime {
    Cuda(Box<model::JuliaModel<CudaBackend>>),
    Cpu(Box<model::JuliaModel<CpuBackend>>),
    #[cfg(test)]
    Fixture(InferenceFixture),
}

#[cfg(test)]
type InferenceFixture = Box<dyn Fn(&CancellationToken) -> eyre::Result<Vec<f32>>>;

/// Resident batch-one Julia provider. Selecting CUDA never falls back to CPU.
pub struct NativeJuliaProvider {
    encoder: JuliaEncoder,
    runtime: Runtime,
    cancellation: CancellationToken,
    metadata: NativeJuliaMetadata,
}

impl NativeJuliaProvider {
    /// Verify CUDA and prepare its executable/toolkit-scoped compilation cache.
    /// No model tensor is allocated or inference launched. Callers may overlap
    /// this bounded setup with parsing a validated request's tokenizer.
    fn prepare_cuda_runtime(cancellation: &CancellationToken) -> eyre::Result<()> {
        cancellation.bail_if_cancelled()?;
        let identity = cuda_runtime::verify_native_cuda_runtime()?;
        kernel_cache::initialize(&identity, cancellation)
    }

    pub fn load(
        model_dir: &Path,
        device: NativeDevice,
        policy: EncodingPolicy,
        cancellation: CancellationToken,
    ) -> eyre::Result<Self> {
        let started = Instant::now();
        cancellation.bail_if_cancelled()?;
        policy.validate()?;
        let (encoder, runtime, metadata) = std::thread::scope(|scope| -> eyre::Result<_> {
            let dispatch = tracing::dispatcher::get_default(Clone::clone);
            let span = tracing::Span::current();
            let tokenizer = std::thread::Builder::new()
                .name("julia-tokenizer".into())
                .spawn_scoped(scope, move || {
                    tracing::dispatcher::with_default(&dispatch, || {
                        span.in_scope(|| JuliaEncoder::load(model_dir, policy))
                    })
                })?;
            // CUDA tensors remain on this thread; only the CPU encoder crosses
            // the join. A failed tokenizer never publishes a partial provider.
            let numerical = Self::load_runtime(model_dir, device, &cancellation);
            let parsed = tokenizer
                .join()
                .map_err(|_| eyre::eyre!("Julia tokenizer worker panicked"));
            cancellation.bail_if_cancelled()?;
            let encoder = parsed??;
            let (runtime, metadata) = numerical?;
            Ok((encoder, runtime, metadata))
        })?;
        cancellation.bail_if_cancelled()?;
        let mut provider = Self {
            encoder,
            runtime,
            cancellation,
            metadata,
        };
        // Keep this convenience API's original timing boundary, including tokenizer loading.
        provider.metadata.load_millis =
            started.elapsed().as_millis().try_into().unwrap_or(u64::MAX);
        Ok(provider)
    }

    /// Load numerical weights while reusing a previously validated, parsed Julia encoder.
    /// Its shared tokenizer remains owned by this provider after the caller drops its clone.
    /// The load timing excludes tokenizer work completed before this call.
    pub fn load_with_encoder(
        model_dir: &Path,
        device: NativeDevice,
        encoder: JuliaEncoder,
        cancellation: CancellationToken,
    ) -> eyre::Result<Self> {
        cancellation.bail_if_cancelled()?;
        encoder.policy().validate()?;
        let (runtime, metadata) = Self::load_runtime(model_dir, device, &cancellation)?;
        Ok(Self {
            encoder,
            runtime,
            cancellation,
            metadata,
        })
    }

    fn load_runtime(
        model_dir: &Path,
        device: NativeDevice,
        cancellation: &CancellationToken,
    ) -> eyre::Result<(Runtime, NativeJuliaMetadata)> {
        let started = Instant::now();
        cancellation.bail_if_cancelled()?;
        let runtime = {
            let _span = tracing::debug_span!("julia_native_model_load").entered();
            match device {
                NativeDevice::Cuda => {
                    let checkpoint = std::thread::scope(|scope| -> eyre::Result<_> {
                        let dispatch = tracing::dispatcher::get_default(Clone::clone);
                        let span = tracing::Span::current();
                        let host = std::thread::Builder::new()
                            .name("julia-checkpoint".into())
                            .spawn_scoped(scope, move || {
                                tracing::dispatcher::with_default(&dispatch, || {
                                    span.in_scope(|| {
                                        model::PreparedCheckpoint::load(model_dir, cancellation)
                                    })
                                })
                            })?;
                        let setup = Self::prepare_cuda_runtime(cancellation);
                        let prepared = host
                            .join()
                            .map_err(|_| eyre::eyre!("Julia checkpoint worker panicked"));
                        cancellation.bail_if_cancelled()?;
                        setup?;
                        prepared?
                    })?;
                    Runtime::Cuda(Box::new(model::JuliaModel::load_prepared(
                        checkpoint,
                        CudaDevice::default(),
                        cancellation,
                        model::MatmulMode::Fp32Tiled,
                    )?))
                }
                NativeDevice::Cpu => Runtime::Cpu(Box::new(model::JuliaModel::load(
                    model_dir,
                    Default::default(),
                    cancellation,
                    model::MatmulMode::Burn,
                )?)),
            }
        };
        cancellation.bail_if_cancelled()?;
        let metadata = NativeJuliaMetadata {
            backend: "burn-julia".into(),
            device,
            precision: match device {
                NativeDevice::Cuda => "FP32 SimpleUnit tiled matmul; no accelerated TF32 matmul",
                NativeDevice::Cpu => "FP32 NdArray debug/reference matmul",
            }
            .into(),
            source_reference_revision: JULIA_MODEL_REVISION.into(),
            weights_bytes: std::fs::metadata(model_dir.join("model.safetensors"))?.len(),
            load_millis: started.elapsed().as_millis().try_into().unwrap_or(u64::MAX),
        };
        Ok((runtime, metadata))
    }

    pub fn metadata(&self) -> &NativeJuliaMetadata {
        &self.metadata
    }

    pub fn encoder(&self) -> &JuliaEncoder {
        &self.encoder
    }

    pub fn infer_encoded(&mut self, encoded: &EncodedJuliaRequest) -> eyre::Result<Vec<f64>> {
        self.infer_encoded_with_cancellation(encoded, &self.cancellation.clone())
    }

    /// Use the current job's token when the provider survives multiple requests.
    /// Cancellation is checked between layers; an executing kernel is not preempted.
    pub fn infer_encoded_with_cancellation(
        &mut self,
        encoded: &EncodedJuliaRequest,
        cancellation: &CancellationToken,
    ) -> eyre::Result<Vec<f64>> {
        encoded.validate()?;
        eyre::ensure!(
            encoded.token_count() <= self.encoder.policy().max_tokens,
            "Encoded Julia request exceeds this provider's configured context limit"
        );
        cancellation.bail_if_cancelled()?;
        let _span = tracing::info_span!(
            "julia_native_infer",
            tokens = encoded.token_count(),
            options = encoded.option_count()
        )
        .entered();
        let logits = match &mut self.runtime {
            Runtime::Cuda(model) => model.infer(encoded, cancellation)?,
            Runtime::Cpu(model) => model.infer(encoded, cancellation)?,
            #[cfg(test)]
            Runtime::Fixture(infer) => infer(cancellation)?,
        };
        cancellation.bail_if_cancelled()?;
        eyre::ensure!(
            logits.iter().all(|value| value.is_finite()),
            "Julia produced nonfinite logits"
        );
        Ok(logits.into_iter().map(f64::from).collect())
    }
}

impl FiniteChoiceProvider for NativeJuliaProvider {
    fn capabilities(&self) -> BackendCapabilities {
        julia_capabilities()
    }

    fn infer(
        &mut self,
        request: &ValidatedFiniteChoiceRequest<'_>,
    ) -> Result<FiniteChoiceResult, FiniteChoiceError> {
        // Revalidate for Julia even if the wrapper originated from a visual-capable provider.
        let encoded = self.encoder.encode(request)?;
        let logits = self.infer_encoded(&encoded).map_err(|error| {
            FiniteChoiceError::new(
                FiniteChoiceErrorKind::ProviderFailure,
                format!("Julia native inference: {error:#}"),
            )
        })?;
        FiniteChoiceResult::from_logits(request, logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_provider(
        load_token: CancellationToken,
        infer: InferenceFixture,
    ) -> NativeJuliaProvider {
        let json = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":4,"content":"<mask>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],"normalizer":null,"pre_tokenizer":{"type":"WhitespaceSplit"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"<pad>":0,"<eos>":1,"<bos>":2,"<unk>":3,"<mask>":4},"unk_token":"<unk>"}}"#;
        NativeJuliaProvider {
            encoder: JuliaEncoder::from_tokenizer_json(json, EncodingPolicy::default()).unwrap(),
            runtime: Runtime::Fixture(infer),
            cancellation: load_token,
            metadata: NativeJuliaMetadata {
                backend: "fixture".into(),
                device: NativeDevice::Cpu,
                precision: "fixture".into(),
                source_reference_revision: "fixture".into(),
                weights_bytes: 0,
                load_millis: 0,
            },
        }
    }

    fn fixture_encoding() -> EncodedJuliaRequest {
        EncodedJuliaRequest {
            version: 1,
            raw_input_ids: vec![2, 3, 4, 3, 4, 3, 1],
            input_ids: vec![2, 3, 4, 3, 4, 3, 1, 0],
            attention_mask: vec![1, 1, 1, 1, 1, 1, 1, 0],
            marker_positions: vec![2, 4],
            marker_mask: vec![true, true],
            qtype: 0,
            option_ids: vec!["a".into(), "b".into()],
        }
    }

    #[test]
    fn resident_inference_uses_current_job_instead_of_cancelled_load_token() {
        let load = CancellationToken::new();
        load.request_cancel("previous job cancelled");
        let mut provider = fixture_provider(
            load,
            Box::new(|token| {
                token.bail_if_cancelled()?;
                Ok(vec![1.0, 2.0])
            }),
        );
        let encoding = fixture_encoding();
        assert_eq!(
            provider
                .infer_encoded_with_cancellation(&encoding, &CancellationToken::new())
                .unwrap(),
            vec![1.0, 2.0]
        );
        assert!(provider.infer_encoded(&encoding).is_err());
    }

    #[test]
    fn current_job_cancellation_reaches_the_resident_inference_boundary() {
        let current = CancellationToken::new();
        let trigger = current.clone();
        let mut provider = fixture_provider(
            CancellationToken::new(),
            Box::new(move |received| {
                trigger.request_cancel("current job cancelled during inference");
                received.bail_if_cancelled()?;
                Ok(vec![1.0, 2.0])
            }),
        );
        let error = provider
            .infer_encoded_with_cancellation(&fixture_encoding(), &current)
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("current job cancelled during inference")
        );
    }

    #[test]
    fn cancelled_load_stops_before_reading_any_model_artifacts() {
        let cancellation = CancellationToken::new();
        cancellation.request_cancel("fixture stop");
        assert!(
            NativeJuliaProvider::load(
                Path::new("unused-fixture-model"),
                NativeDevice::Cpu,
                EncodingPolicy::default(),
                cancellation
            )
            .is_err()
        );
    }
}
