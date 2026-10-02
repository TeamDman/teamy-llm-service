//! Julia-1's numerical operations in Rust over the existing Burn backend.
//! Native CUDA is the explicit production target; NdArray is an explicit debug/reference device.

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
}

/// Resident batch-one Julia provider. Selecting CUDA never falls back to CPU.
pub struct NativeJuliaProvider {
    encoder: JuliaEncoder,
    runtime: Runtime,
    cancellation: CancellationToken,
    metadata: NativeJuliaMetadata,
}

impl NativeJuliaProvider {
    pub fn load(
        model_dir: &Path,
        device: NativeDevice,
        policy: EncodingPolicy,
        cancellation: CancellationToken,
    ) -> eyre::Result<Self> {
        let started = Instant::now();
        cancellation.bail_if_cancelled()?;
        let encoder = JuliaEncoder::load(model_dir, policy)?;
        let runtime = match device {
            NativeDevice::Cuda => {
                teamy_llm_burn_qwen35::burn_text::verify_required_cuda_runtime()?;
                Runtime::Cuda(Box::new(model::JuliaModel::load(
                    model_dir,
                    CudaDevice::default(),
                    &cancellation,
                    model::MatmulMode::Fp32Reference,
                )?))
            }
            NativeDevice::Cpu => Runtime::Cpu(Box::new(model::JuliaModel::load(
                model_dir,
                Default::default(),
                &cancellation,
                model::MatmulMode::Burn,
            )?)),
        };
        cancellation.bail_if_cancelled()?;
        let metadata = NativeJuliaMetadata {
            backend: "burn-julia".into(),
            device,
            precision: match device {
                NativeDevice::Cuda => {
                    "FP32 chunked multiply/sum correctness baseline; no accelerated TF32 matmul"
                }
                NativeDevice::Cpu => "FP32 NdArray debug/reference matmul",
            }
            .into(),
            source_reference_revision: JULIA_MODEL_REVISION.into(),
            weights_bytes: std::fs::metadata(model_dir.join("model.safetensors"))?.len(),
            load_millis: started.elapsed().as_millis().try_into().unwrap_or(u64::MAX),
        };
        Ok(Self {
            encoder,
            runtime,
            cancellation,
            metadata,
        })
    }

    pub fn metadata(&self) -> &NativeJuliaMetadata {
        &self.metadata
    }

    pub fn encoder(&self) -> &JuliaEncoder {
        &self.encoder
    }

    pub fn infer_encoded(&mut self, encoded: &EncodedJuliaRequest) -> eyre::Result<Vec<f64>> {
        encoded.validate()?;
        eyre::ensure!(
            encoded.token_count() <= self.encoder.policy().max_tokens,
            "Encoded Julia request exceeds this provider's configured context limit"
        );
        self.cancellation.bail_if_cancelled()?;
        let _span = tracing::info_span!(
            "julia_native_infer",
            tokens = encoded.token_count(),
            options = encoded.option_count()
        )
        .entered();
        let logits = match &mut self.runtime {
            Runtime::Cuda(model) => model.infer(encoded, &self.cancellation)?,
            Runtime::Cpu(model) => model.infer(encoded, &self.cancellation)?,
        };
        self.cancellation.bail_if_cancelled()?;
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
