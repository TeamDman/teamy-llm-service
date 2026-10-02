//! Resident batch-one Julia-1 ONNX inference.
//!
//! The encoder is shared with the native implementation. CUDA registration failure is an error;
//! graph placement on CPU is disabled unless the caller explicitly permits it. Cancellation is
//! cooperative around hashing, loading and inference, and cannot preempt an executing GPU kernel.

use std::ffi::CStr;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use eyre::{Result, WrapErr, ensure, eyre};
use facet::Facet;
use ort::ep;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::{Tensor, TensorElementType, ValueType};
use sha2::{Digest, Sha256};
use teamy_cancellation::CancellationToken;
use teamy_llm_core::finite_choice::{
    BackendCapabilities, FiniteChoiceCapabilities, FiniteChoiceError, FiniteChoiceErrorKind,
    FiniteChoiceProvider, FiniteChoiceResult, ValidatedFiniteChoiceRequest,
};
use teamy_llm_julia_common::{EncodedJuliaRequest, EncodingPolicy, JuliaEncoder};

pub const ORT_VERSION: &str = "1.24.2";
pub const ORT_BINDING_VERSION: &str = "2.0.0-rc.12";
pub const ONNX_MODEL_REVISION: &str = "82a2fadf8fccfccdc5fd4e1009ba8f1a265eb7a8";

#[derive(Clone, Debug, PartialEq, Eq, Facet)]
#[repr(u8)]
pub enum OnnxDevice {
    Cpu,
    Cuda { device_id: i32 },
}

#[derive(Clone, Debug)]
pub struct JuliaOnnxOptions {
    pub model_dir: PathBuf,
    /// Explicit inspected DLL or shared-library path; no PATH search or download occurs here.
    pub runtime_library: PathBuf,
    pub device: OnnxDevice,
    pub encoding: EncodingPolicy,
    /// Defaults should be false. Permitting fallback does not establish actual node placement.
    pub allow_cpu_fallback: bool,
    pub tf32: bool,
    pub intra_threads: usize,
    /// Optional ORT profiling filename prefix. ORT adds a timestamp; finish_profiling returns the
    /// actual local trace path. Caller chooses this explicit metadata output location.
    pub profile_prefix: Option<PathBuf>,
    /// Optional frozen artifact identities supplied by the acquisition layer.
    pub runtime_sha256: Option<String>,
    pub graph_sha256: Option<String>,
    pub weights_sha256: Option<String>,
}

impl JuliaOnnxOptions {
    /// Caller supplies inspected local artifacts and an explicit device. Defaults use exact
    /// TF32 disabled and reject CPU fallback for CUDA; this never acquires artifacts.
    #[must_use]
    pub fn new(model_dir: PathBuf, runtime_library: PathBuf, device: OnnxDevice) -> Self {
        Self {
            model_dir,
            runtime_library,
            device,
            encoding: EncodingPolicy::default(),
            allow_cpu_fallback: false,
            tf32: false,
            intra_threads: 2,
            profile_prefix: None,
            runtime_sha256: None,
            graph_sha256: None,
            weights_sha256: None,
        }
    }
}

#[derive(Clone, Debug, Facet)]
pub struct OnnxAudit {
    pub binding_version: String,
    pub runtime_version: String,
    pub runtime_library_sha256: String,
    pub runtime_build_info: String,
    pub graph_sha256: String,
    pub external_weights_sha256: String,
    pub requested_device: OnnxDevice,
    /// Provider successfully registered for this session, not a measured per-node trace.
    pub registered_provider: String,
    pub cpu_fallback_permitted: bool,
    pub tf32: bool,
    pub max_tokens: usize,
    pub head_tokens: usize,
}

struct LoadedRuntime {
    path: PathBuf,
    sha256: String,
    version: String,
}

static LOADED_RUNTIME: OnceLock<Mutex<Option<LoadedRuntime>>> = OnceLock::new();

pub struct JuliaOnnxProvider {
    session: Session,
    encoder: JuliaEncoder,
    cancellation: CancellationToken,
    audit: OnnxAudit,
}

impl JuliaOnnxProvider {
    pub fn load(options: JuliaOnnxOptions, cancellation: CancellationToken) -> Result<Self> {
        cancellation.bail_if_cancelled()?;
        ensure!(
            options.intra_threads > 0,
            "ONNX thread count must be positive"
        );
        if let OnnxDevice::Cuda { device_id } = options.device {
            ensure!(device_id >= 0, "CUDA device ID must be nonnegative");
        }
        let runtime = initialize_runtime(
            &options.runtime_library,
            options.runtime_sha256.as_deref(),
            &cancellation,
        )?;
        let graph = options.model_dir.join("model.onnx");
        let weights = options.model_dir.join("model.onnx.data");
        let graph_sha256 = hash_file(&graph, &cancellation)?;
        let weights_sha256 = hash_file(&weights, &cancellation)?;
        verify_hash("ONNX graph", &graph_sha256, options.graph_sha256.as_deref())?;
        verify_hash(
            "ONNX external weights",
            &weights_sha256,
            options.weights_sha256.as_deref(),
        )?;
        let max_tokens = options.encoding.max_tokens;
        let head_tokens = options.encoding.head_tokens;
        let encoder = JuliaEncoder::load(&options.model_dir, options.encoding)?;
        cancellation.bail_if_cancelled()?;

        let mut builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort_error)?
            .with_intra_threads(options.intra_threads)
            .map_err(ort_error)?;
        if let Some(prefix) = &options.profile_prefix {
            builder = builder.with_profiling(prefix).map_err(ort_error)?;
        }
        let (registered_provider, cpu_fallback_permitted) = match options.device {
            OnnxDevice::Cpu => {
                builder = builder
                    .with_execution_providers([ep::CPU::default().build().error_on_failure()])
                    .map_err(ort_error)?;
                ("CPUExecutionProvider", false)
            }
            OnnxDevice::Cuda { device_id } => {
                builder = builder
                    .with_execution_providers([ep::CUDA::default()
                        .with_device_id(device_id)
                        .with_tf32(options.tf32)
                        .build()
                        .error_on_failure()])
                    .map_err(ort_error)?;
                if !options.allow_cpu_fallback {
                    builder = builder.with_disable_cpu_fallback().map_err(ort_error)?;
                }
                ("CUDAExecutionProvider", options.allow_cpu_fallback)
            }
        };
        cancellation.bail_if_cancelled()?;
        // ORT session preparation is synchronous. A cancellation observed afterwards discards it.
        let session = builder
            .commit_from_file(&graph)
            .wrap_err("Preparing the pinned Julia ONNX graph failed")?;
        cancellation.bail_if_cancelled()?;
        verify_graph_contract(&session)?;
        let audit = OnnxAudit {
            binding_version: ORT_BINDING_VERSION.into(),
            runtime_version: runtime.version,
            runtime_library_sha256: runtime.sha256,
            runtime_build_info: ort::info().into(),
            graph_sha256,
            external_weights_sha256: weights_sha256,
            requested_device: options.device,
            registered_provider: registered_provider.into(),
            cpu_fallback_permitted,
            tf32: options.tf32,
            max_tokens,
            head_tokens,
        };
        tracing::info!(provider = %audit.registered_provider, cpu_fallback = audit.cpu_fallback_permitted,
            "Julia ONNX session prepared");
        Ok(Self {
            session,
            encoder,
            cancellation,
            audit,
        })
    }

    #[must_use]
    pub fn audit(&self) -> &OnnxAudit {
        &self.audit
    }

    /// Flush enabled ORT profiling and return the actual timestamped trace path.
    pub fn finish_profiling(&mut self) -> Result<PathBuf> {
        Ok(PathBuf::from(self.session.end_profiling()?))
    }

    /// Scores exact shared-encoder arrays, including explicit state/question/type reference cases.
    pub fn infer_encoded(&mut self, encoded: &EncodedJuliaRequest) -> Result<Vec<f64>> {
        self.cancellation.bail_if_cancelled()?;
        validate_encoded(encoded)?;
        ensure!(
            encoded.token_count() <= self.audit.max_tokens,
            "Encoded request exceeds this session's declared token limit"
        );
        let tokens = encoded.input_ids.len();
        let options = encoded.option_ids.len();
        let input_ids = Tensor::from_array(([1_usize, tokens], encoded.input_ids.clone()))?;
        let attention = Tensor::from_array(([1_usize, tokens], encoded.attention_mask.clone()))?;
        let markers = Tensor::from_array(([1_usize, options], encoded.marker_positions.clone()))?;
        let mask = Tensor::from_array(([1_usize, options], encoded.marker_mask.clone()))?;
        let qtype = Tensor::from_array(([1_usize], vec![encoded.qtype]))?;
        self.cancellation.bail_if_cancelled()?;
        let outputs = self.session.run(ort::inputs![
            "input_ids" => input_ids,
            "attention_mask" => attention,
            "marker_pos" => markers,
            "marker_mask" => mask,
            "qtype" => qtype,
        ])?;
        let (shape, values) = outputs["logits"].try_extract_tensor::<f32>()?;
        ensure!(
            shape.as_ref() == [1_i64, options as i64],
            "Unexpected Julia logit shape: {shape:?}"
        );
        let logits: Vec<f64> = values.iter().map(|value| f64::from(*value)).collect();
        ensure!(
            logits.iter().all(|value| value.is_finite()),
            "Julia emitted non-finite logits"
        );
        self.cancellation.bail_if_cancelled()?;
        Ok(logits)
    }

    pub fn encode(
        &self,
        request: &ValidatedFiniteChoiceRequest<'_>,
    ) -> Result<EncodedJuliaRequest, FiniteChoiceError> {
        self.encoder.encode(request)
    }
}

impl FiniteChoiceProvider for JuliaOnnxProvider {
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            text_generation: false,
            finite_choice: Some(FiniteChoiceCapabilities::julia_1_text_only()),
        }
    }

    fn infer(
        &mut self,
        request: &ValidatedFiniteChoiceRequest<'_>,
    ) -> Result<FiniteChoiceResult, FiniteChoiceError> {
        self.cancellation
            .bail_if_cancelled()
            .map_err(provider_error)?;
        let encoded = self.encoder.encode(request)?;
        let logits = self.infer_encoded(&encoded).map_err(provider_error)?;
        FiniteChoiceResult::from_logits(request, logits)
    }
}

fn provider_error(error: impl std::fmt::Display) -> FiniteChoiceError {
    FiniteChoiceError::new(FiniteChoiceErrorKind::ProviderFailure, error.to_string())
}

// rc.12 builder errors carry a recoverable builder, which is deliberately not Send/Sync.
// Preserve its diagnostic without passing that recovery state into a thread-safe eyre report.
fn ort_error<R>(error: ort::Error<R>) -> eyre::Report {
    eyre!("{error}")
}

/// ort loads its DLL and environment once per process. Reject later requests for another identity.
fn initialize_runtime(
    path: &Path,
    expected_sha256: Option<&str>,
    cancellation: &CancellationToken,
) -> Result<LoadedRuntime> {
    let path = path
        .canonicalize()
        .wrap_err("Resolving the explicit ONNX runtime library failed")?;
    let sha256 = hash_file(&path, cancellation)?;
    verify_hash("ONNX Runtime library", &sha256, expected_sha256)?;
    let mut state = LOADED_RUNTIME
        .get_or_init(|| Mutex::new(None))
        .lock()
        .map_err(|_| eyre!("ONNX Runtime initialization lock was poisoned"))?;
    if let Some(loaded) = state.as_ref() {
        ensure!(
            loaded.path == path && loaded.sha256 == sha256,
            "A different ONNX Runtime is already loaded in this process"
        );
        return Ok(LoadedRuntime {
            path,
            sha256,
            version: loaded.version.clone(),
        });
    }
    cancellation.bail_if_cancelled()?;
    // SAFETY: Explicit inspected runtime path. Keep the library alive while calling its documented
    // C entry point and copying the returned static version string. No model source is executed.
    let library = unsafe { libloading::Library::new(&path) }
        .wrap_err("Loading the explicit ONNX Runtime failed")?;
    let getter: libloading::Symbol<'_, unsafe extern "system" fn() -> *const ort::sys::OrtApiBase> =
        unsafe { library.get(b"OrtGetApiBase\0") }?;
    let base = unsafe { getter() };
    ensure!(!base.is_null(), "ONNX Runtime returned a null API base");
    let version_pointer = unsafe { ((*base).GetVersionString)() };
    ensure!(
        !version_pointer.is_null(),
        "ONNX Runtime returned a null version string"
    );
    let version = unsafe { CStr::from_ptr(version_pointer) }
        .to_str()?
        .to_owned();
    ensure!(
        version == ORT_VERSION,
        "Expected ONNX Runtime {ORT_VERSION}, found {version}"
    );
    ensure!(
        ort::init_from(&path)?.commit(),
        "An ONNX environment was already initialized outside this adapter"
    );
    *state = Some(LoadedRuntime {
        path: path.clone(),
        sha256: sha256.clone(),
        version: version.clone(),
    });
    cancellation.bail_if_cancelled()?;
    Ok(LoadedRuntime {
        path,
        sha256,
        version,
    })
}

fn hash_file(path: &Path, cancellation: &CancellationToken) -> Result<String> {
    let mut input =
        File::open(path).wrap_err_with(|| format!("Opening {} failed", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        cancellation.bail_if_cancelled()?;
        let read = input.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn verify_hash(label: &str, actual: &str, expected: Option<&str>) -> Result<()> {
    if let Some(expected) = expected {
        ensure!(
            expected.len() == 64 && expected.bytes().all(|value| value.is_ascii_hexdigit()),
            "Invalid expected {label} SHA-256"
        );
        ensure!(
            actual.eq_ignore_ascii_case(expected),
            "{label} SHA-256 does not match the frozen identity"
        );
    }
    Ok(())
}

fn verify_graph_contract(session: &Session) -> Result<()> {
    let expected = [
        ("input_ids", TensorElementType::Int64, 2),
        ("attention_mask", TensorElementType::Int64, 2),
        ("marker_pos", TensorElementType::Int64, 2),
        ("marker_mask", TensorElementType::Bool, 2),
        ("qtype", TensorElementType::Int64, 1),
    ];
    ensure!(
        session.inputs().len() == expected.len(),
        "Unexpected Julia ONNX input count"
    );
    for (name, ty, rank) in expected {
        let input = session
            .inputs()
            .iter()
            .find(|input| input.name() == name)
            .ok_or_else(|| eyre!("Missing Julia ONNX input {name}"))?;
        ensure!(
            matches!(input.dtype(), ValueType::Tensor { ty: found, shape, .. } if *found == ty && shape.len() == rank),
            "Unexpected Julia ONNX type or rank for {name}"
        );
    }
    ensure!(
        session.outputs().len() == 1 && session.outputs()[0].name() == "logits",
        "Unexpected Julia ONNX output names"
    );
    ensure!(
        matches!(session.outputs()[0].dtype(), ValueType::Tensor { ty: TensorElementType::Float32, shape, .. } if shape.len() == 2),
        "Julia ONNX logits must be a rank-two float32 tensor"
    );
    Ok(())
}

fn validate_encoded(encoded: &EncodedJuliaRequest) -> Result<()> {
    encoded.validate()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encoded() -> EncodedJuliaRequest {
        EncodedJuliaRequest {
            version: 1,
            raw_input_ids: vec![2, 4, 5, 4, 6, 1],
            input_ids: vec![2, 4, 5, 4, 6, 1, 0, 0],
            attention_mask: vec![1, 1, 1, 1, 1, 1, 0, 0],
            marker_positions: vec![1, 3],
            marker_mask: vec![true, true],
            qtype: 0,
            option_ids: vec!["first".into(), "second".into()],
        }
    }

    #[test]
    fn accepts_ordered_padded_encoder_arrays() {
        validate_encoded(&encoded()).unwrap();
    }

    #[test]
    fn rejects_marker_on_padding_and_duplicate_ids() {
        let mut request = encoded();
        request.marker_positions[1] = 6;
        assert!(validate_encoded(&request).is_err());
        let mut request = encoded();
        request.option_ids[1] = "first".into();
        assert!(validate_encoded(&request).is_err());
    }

    #[test]
    fn rejects_stale_schema_non_boolean_masks_and_wrong_shapes() {
        let mut request = encoded();
        request.version = 2;
        assert!(validate_encoded(&request).is_err());
        let mut request = encoded();
        request.attention_mask[0] = 2;
        assert!(validate_encoded(&request).is_err());
        let mut request = encoded();
        request.marker_mask.pop();
        assert!(validate_encoded(&request).is_err());
    }

    #[test]
    fn hash_verification_rejects_changed_or_malformed_identity() {
        assert!(verify_hash("fixture", &"a".repeat(64), Some(&"a".repeat(64))).is_ok());
        assert!(verify_hash("fixture", &"a".repeat(64), Some(&"b".repeat(64))).is_err());
        assert!(verify_hash("fixture", &"a".repeat(64), Some("not a hash")).is_err());
    }

    #[test]
    fn cancelled_hashing_stops_before_reading() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let cancellation = CancellationToken::new();
        cancellation.request_cancel("fixture stop");
        assert!(hash_file(file.path(), &cancellation).is_err());
    }
}
