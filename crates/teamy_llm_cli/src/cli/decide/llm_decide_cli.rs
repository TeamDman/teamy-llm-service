//! Versioned finite-choice CLI. Requests are data, not authorization to act on a choice.
use std::fs::File;
use std::io::{self, IsTerminal, Read, Write};
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use crate::cli::output::OutputFormat;
use eyre::{Result, WrapErr, ensure, eyre};
use facet::Facet;
use figue as args;
use teamy_cancellation::CancellationToken;
use teamy_llm_burn_julia::{NativeDevice, NativeJuliaMetadata, NativeJuliaProvider};
use teamy_llm_core::ModelSelector;
use teamy_llm_core::finite_choice::{
    ChoiceContent, FiniteChoiceOption, FiniteChoiceRequest, FiniteChoiceResult,
};
use teamy_llm_julia_common::{
    EncodedJuliaRequest, EncodingPolicy, JuliaEncoder, julia_capabilities,
};
use teamy_llm_onnx_julia::{JuliaOnnxOptions, JuliaOnnxProvider, OnnxAudit, OnnxDevice};

const MAX_REQUEST_BYTES: usize = 2 * 1024 * 1024;

#[derive(Debug, Facet)]
#[facet(rename_all = "kebab-case", deny_unknown_fields)]
struct DecideRequest {
    version: u32,
    #[facet(default)]
    model: Option<String>,
    #[facet(default)]
    question: Option<String>,
    input: ChoiceContent,
    options: Vec<FiniteChoiceOption>,
}

#[derive(Debug, Facet)]
#[facet(rename_all = "kebab-case")]
pub struct DecideOutput {
    pub version: u32,
    pub task: String,
    pub model: String,
    pub provider: String,
    pub device: String,
    pub question: String,
    pub encoding_policy: EncodingPolicy,
    pub token_count: usize,
    pub option_count: usize,
    pub load_ms: f64,
    pub inference_ms: Vec<f64>,
    pub best_option_id: String,
    pub result: FiniteChoiceResult,
    pub encoding: Option<EncodedJuliaRequest>,
    pub onnx_audit: Option<OnnxAudit>,
    pub native_metadata: Option<NativeJuliaMetadata>,
    pub profiling_file: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Facet)]
#[repr(u8)]
#[facet(rename_all = "kebab-case")]
pub enum DecisionProvider {
    Native,
    Onnx,
}
impl DecisionProvider {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::Onnx => "onnx",
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Facet)]
#[repr(u8)]
#[facet(rename_all = "kebab-case")]
pub enum DecisionDevice {
    Cuda,
    Cpu,
}
impl DecisionDevice {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Cpu => "cpu",
        }
    }
}

/// Choose between 2–20 text options using the independently qualified Julia-1 model.
///
/// Input is version 1 JSON: input {text, visuals: []}, options [{id, content:
/// {text, visuals: []}}], optional model and question. Images are rejected.
/// --request accepts quoted '@file', '-' for stdin, or inline JSON. Omitted
/// --request reads piped UTF-8 stdin, bounded to 2 MiB. Output is versioned JSON.
/// CUDA never silently falls back. Executing GPU kernels cannot be preempted.
#[derive(Debug, Clone, PartialEq, Eq, Facet)]
pub struct DecideCli {
    /// Request source: '@file', '-', or inline JSON; omitted means piped stdin.
    #[facet(args::named)]
    pub request: Option<String>,
    /// Explicit qualified backend: native or onnx.
    #[facet(args::named)]
    pub provider: DecisionProvider,
    /// julia-1, or a matching model field in the JSON request.
    #[facet(args::named)]
    pub model: Option<String>,
    /// Existing locally acquired Julia artifact directory.
    #[facet(args::named)]
    pub model_dir: PathBuf,
    /// Explicit compute device; no automatic fallback.
    #[facet(args::named, default = DecisionDevice::Cuda)]
    pub device: DecisionDevice,
    /// Explicit ONNX runtime library path; required for ONNX only.
    #[facet(args::named)]
    pub runtime_library: Option<PathBuf>,
    /// Override the request question. Default: Which option should be chosen?
    #[facet(args::named)]
    pub question: Option<String>,
    /// Repeated inference requests against the same resident provider, 1–1000.
    #[facet(args::named, default = 1)]
    pub repeat: usize,
    /// Include exact encoded arrays in the JSON result.
    #[facet(args::named, default)]
    pub include_encoding: bool,
    /// Julia encoding limit, including options and question.
    #[facet(args::named, default = 1024)]
    pub max_tokens: usize,
    /// Julia input head-token budget.
    #[facet(args::named, default = 256)]
    pub head_tokens: usize,
    /// Explicitly permit mixed CPU/CUDA placement; ONNX CUDA only.
    #[facet(args::named, default)]
    pub allow_cpu_fallback: bool,
    /// Write an ONNX node trace; output reports the timestamped filename.
    #[facet(args::named)]
    pub profile_prefix: Option<PathBuf>,
}
impl DecideCli {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (1..=1000).contains(&self.repeat),
            "--repeat must be in 1..=1000"
        );
        ensure!(
            self.provider != DecisionProvider::Onnx || self.runtime_library.is_some(),
            "ONNX requires an explicit --runtime-library path"
        );
        ensure!(
            self.provider != DecisionProvider::Native || self.runtime_library.is_none(),
            "--runtime-library applies only to ONNX"
        );
        ensure!(
            self.provider == DecisionProvider::Onnx
                || (!self.allow_cpu_fallback && self.profile_prefix.is_none()),
            "--allow-cpu-fallback and --profile-prefix apply only to ONNX"
        );
        ensure!(
            !self.allow_cpu_fallback || self.device == DecisionDevice::Cuda,
            "--allow-cpu-fallback requires --device cuda"
        );
        self.encoding_policy().validate()?;
        Ok(())
    }
    fn encoding_policy(&self) -> EncodingPolicy {
        let mut policy = EncodingPolicy {
            max_tokens: self.max_tokens,
            head_tokens: self.head_tokens,
            ..Default::default()
        };
        if let Some(question) = &self.question {
            policy.question.clone_from(question);
        }
        policy
    }
}

enum Provider {
    Native(Box<NativeJuliaProvider>),
    Onnx(Box<JuliaOnnxProvider>),
}
impl Provider {
    fn infer_encoded(&mut self, encoding: &EncodedJuliaRequest) -> Result<Vec<f64>> {
        match self {
            Self::Native(provider) => provider.infer_encoded(encoding),
            Self::Onnx(provider) => provider.infer_encoded(encoding),
        }
    }
}

impl DecideCli {
    pub fn for_interactive(args: &crate::cli::interactive::InteractiveCli) -> Result<Self> {
        let options = Self {
            request: None,
            provider: args.decision_provider.ok_or_else(|| {
                eyre!("interactive decisions require --decision-provider native|onnx")
            })?,
            model: Some("julia-1".into()),
            model_dir: args
                .decision_model_dir
                .clone()
                .ok_or_else(|| eyre!("interactive decisions require --decision-model-dir"))?,
            device: args.device,
            runtime_library: args.runtime_library.clone(),
            question: None,
            repeat: 1,
            include_encoding: false,
            max_tokens: args.max_tokens,
            head_tokens: args.head_tokens,
            allow_cpu_fallback: args.allow_cpu_fallback,
            profile_prefix: None,
        };
        options.validate()?;
        Ok(options)
    }
}

/// Persistent decision configuration. Construction validates options without artifact reads.
/// Providers retain the first evaluation's engine-lifetime cancellation token.
/// Drop and replace this engine after cancellation; the persistent worker can continue.
/// Executing GPU kernels cannot be preempted.
pub struct DecisionEngine {
    options: DecideCli,
    encoder: Option<(EncodingPolicy, JuliaEncoder)>,
    resident: Option<ResidentDecision>,
}
struct ResidentDecision {
    provider: Provider,
    audit: Option<OnnxAudit>,
    native_metadata: Option<NativeJuliaMetadata>,
}
impl DecisionEngine {
    pub fn new(options: DecideCli) -> Result<Self> {
        options.validate()?;
        Ok(Self {
            options,
            encoder: None,
            resident: None,
        })
    }
    /// Decode and validate the same bounded versioned request as `decide`, then infer.
    /// No output stream is touched. Encoder and numerical provider remain resident.
    #[tracing::instrument(level = "info", skip_all)]
    pub fn evaluate_json(
        &mut self,
        bytes: &[u8],
        cancellation: CancellationToken,
    ) -> Result<DecideOutput> {
        let wire = decode_request(bytes)?;
        let model = self
            .options
            .model
            .clone()
            .or(wire.model)
            .ok_or_else(|| eyre!("specify --model julia-1 or model in the request"))?;
        ensure!(
            ["julia-1", "SupersonicLabs/Julia-1"].contains(&model.as_str()),
            "decide supports Julia-1; requested model {model}"
        );
        let mut policy = self.options.encoding_policy();
        if let Some(question) = self.options.question.clone().or(wire.question) {
            policy.question = question;
        }
        policy.validate()?;
        let request = FiniteChoiceRequest {
            selector: ModelSelector {
                model: Some(model.clone()),
                model_dir: Some(self.options.model_dir.display().to_string()),
            },
            input: wire.input,
            options: wire.options,
        };
        // Capability checks still precede tokenizer and numerical artifact reads.
        let validated = request.validate_for(&julia_capabilities())?;
        cancellation.bail_if_cancelled()?;
        if self
            .encoder
            .as_ref()
            .is_none_or(|(cached, _)| cached != &policy)
        {
            self.encoder = Some((
                policy.clone(),
                JuliaEncoder::load(&self.options.model_dir, policy.clone())?,
            ));
        }
        let encoding = self
            .encoder
            .as_ref()
            .expect("encoder loaded")
            .1
            .encode(&validated)?;
        cancellation.bail_if_cancelled()?;
        let load_started = Instant::now();
        let load_ms = if self.resident.is_none() {
            self.resident = Some(self.load_provider(policy.clone(), cancellation.clone())?);
            load_started.elapsed().as_secs_f64() * 1000.0
        } else {
            0.0
        };
        let mut inference_ms = Vec::with_capacity(self.options.repeat);
        let mut last = None;
        for repetition in 0..self.options.repeat {
            cancellation.bail_if_cancelled()?;
            let _span = tracing::info_span!("julia_decide", repetition).entered();
            let started = Instant::now();
            // Retain the timing contract: encode + completed inference + result validation.
            let encoded = self
                .encoder
                .as_ref()
                .expect("encoder loaded")
                .1
                .encode(&validated)?;
            let logits = self
                .resident
                .as_mut()
                .expect("provider loaded")
                .provider
                .infer_encoded(&encoded);
            let logits = match logits {
                Ok(values) => values,
                Err(error) => {
                    self.resident = None;
                    return Err(error);
                }
            };
            let result = FiniteChoiceResult::from_logits(&validated, logits)?;
            // Preserve finite_choice's final contract guard as well as constructing
            // scores from exact input IDs and max-shifted softmax probabilities.
            result.validate(&validated)?;
            cancellation.bail_if_cancelled()?;
            inference_ms.push(started.elapsed().as_secs_f64() * 1000.0);
            last = Some(result);
        }
        let result = last.ok_or_else(|| eyre!("no inference result"))?;
        let resident = self.resident.as_mut().expect("provider loaded");
        let profiling_file = match &mut resident.provider {
            Provider::Onnx(provider) if self.options.profile_prefix.is_some() => {
                Some(provider.finish_profiling()?.display().to_string())
            }
            _ => None,
        };
        let output = DecideOutput {
            version: 1,
            task: "finite-choice".into(),
            model,
            provider: self.options.provider.as_str().into(),
            device: self.options.device.as_str().into(),
            question: policy.question.clone(),
            encoding_policy: policy,
            token_count: encoding.token_count(),
            option_count: encoding.option_count(),
            load_ms,
            inference_ms,
            best_option_id: result
                .best_option()
                .ok_or_else(|| eyre!("empty result"))?
                .option_id
                .clone(),
            result,
            encoding: self.options.include_encoding.then_some(encoding),
            onnx_audit: resident.audit.clone(),
            native_metadata: resident.native_metadata.clone(),
            profiling_file,
        };
        // An explicit profile finalizes a session. Subsequent evaluations start a new trace.
        if self.options.profile_prefix.is_some() {
            self.resident = None;
        }
        Ok(output)
    }
    fn load_provider(
        &self,
        policy: EncodingPolicy,
        cancellation: CancellationToken,
    ) -> Result<ResidentDecision> {
        let _span = tracing::info_span!(
            "julia_load",
            provider = self.options.provider.as_str(),
            device = self.options.device.as_str()
        )
        .entered();
        if self.options.provider == DecisionProvider::Native {
            let device = if self.options.device == DecisionDevice::Cuda {
                NativeDevice::Cuda
            } else {
                NativeDevice::Cpu
            };
            let provider =
                NativeJuliaProvider::load(&self.options.model_dir, device, policy, cancellation)?;
            let metadata = provider.metadata().clone();
            Ok(ResidentDecision {
                provider: Provider::Native(Box::new(provider)),
                audit: None,
                native_metadata: Some(metadata),
            })
        } else {
            let provider = JuliaOnnxProvider::load(
                JuliaOnnxOptions {
                    model_dir: self.options.model_dir.clone(),
                    runtime_library: self
                        .options
                        .runtime_library
                        .clone()
                        .expect("validated ONNX runtime path"),
                    device: if self.options.device == DecisionDevice::Cuda {
                        OnnxDevice::Cuda { device_id: 0 }
                    } else {
                        OnnxDevice::Cpu
                    },
                    encoding: policy,
                    allow_cpu_fallback: self.options.allow_cpu_fallback,
                    tf32: false,
                    intra_threads: 1,
                    profile_prefix: self.options.profile_prefix.clone(),
                    runtime_sha256: None,
                    graph_sha256: None,
                    weights_sha256: None,
                },
                cancellation,
            )?;
            let audit = provider.audit().clone();
            Ok(ResidentDecision {
                provider: Provider::Onnx(Box::new(provider)),
                audit: Some(audit),
                native_metadata: None,
            })
        }
    }
}

fn decode_request(bytes: &[u8]) -> Result<DecideRequest> {
    ensure!(
        bytes.len() <= MAX_REQUEST_BYTES,
        "request exceeds 2 MiB limit"
    );
    let text = std::str::from_utf8(bytes).wrap_err("request must be UTF-8 JSON")?;
    let request: DecideRequest = facet_json::from_str(text).wrap_err("invalid decide request")?;
    ensure!(
        request.version == 1,
        "unsupported decide request version {}; expected 1",
        request.version
    );
    Ok(request)
}

fn read_bounded(mut reader: impl Read, cancellation: &CancellationToken) -> Result<Vec<u8>> {
    let mut result = Vec::new();
    let mut buffer = [0_u8; 8192];
    loop {
        cancellation.bail_if_cancelled()?;
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        ensure!(
            result.len() + count <= MAX_REQUEST_BYTES,
            "request exceeds 2 MiB limit"
        );
        result.extend_from_slice(&buffer[..count]);
    }
    Ok(result)
}

fn read_request(source: Option<String>, cancellation: &CancellationToken) -> Result<Vec<u8>> {
    if let Some(value) = source.as_deref().filter(|value| *value != "-") {
        if let Some(path) = value.strip_prefix('@') {
            ensure!(!path.is_empty(), "@file request requires a file path");
            return read_bounded(
                File::open(path).wrap_err("cannot open request file")?,
                cancellation,
            );
        }
        ensure!(
            value.len() <= MAX_REQUEST_BYTES,
            "request exceeds 2 MiB limit"
        );
        return Ok(value.as_bytes().to_vec());
    }
    ensure!(
        !io::stdin().is_terminal(),
        "provide --request '@choice.json', inline JSON or piped stdin"
    );
    // The reader may block on a producer. The command can still observe Ctrl+C or a deadline
    // without waiting for that producer to close its pipe. Process exit releases the worker.
    let (sender, receiver) = mpsc::sync_channel(1);
    let token = cancellation.clone();
    std::thread::spawn(move || {
        let _ = sender.send(read_bounded(io::stdin().lock(), &token));
    });
    loop {
        cancellation.bail_if_cancelled()?;
        match receiver.recv_timeout(Duration::from_millis(25)) {
            Ok(result) => return result,
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(error) => return Err(eyre!(error)),
        }
    }
}

pub fn run(
    mut options: DecideCli,
    output_format: Option<OutputFormat>,
    cancellation: CancellationToken,
) -> Result<()> {
    ensure!(
        output_format.is_none_or(|format| format == OutputFormat::Json),
        "decide currently supports only JSON output"
    );
    options.validate()?;
    let bytes = read_request(options.request.take(), &cancellation)?;
    let mut engine = DecisionEngine::new(options)?;
    let output = engine.evaluate_json(&bytes, cancellation.clone())?;
    let encoded = facet_json::to_string(&output)?;
    cancellation.bail_if_cancelled()?;
    writeln!(io::stdout().lock(), "{encoded}")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    const VALID: &str = r#"{"version":1,"model":"julia-1","input":{"text":"Sort a note","visuals":[]},"options":[{"id":"writing","content":{"text":"Writing programs","visuals":[]}},{"id":"playing","content":{"text":"Playing games","visuals":[]}}]}"#;
    #[test]
    fn request_transport_preserves_ids_and_rejects_schema_drift() {
        let parsed = decode_request(VALID.as_bytes()).unwrap();
        assert_eq!(parsed.options[1].id, "playing");
        assert!(
            decode_request(VALID.replace("\"version\":1", "\"version\":2").as_bytes()).is_err()
        );
        assert!(
            decode_request(
                VALID
                    .replace("\"version\":1", "\"typo\":1,\"version\":1")
                    .as_bytes()
            )
            .is_err()
        );
        assert!(decode_request(&[0xff]).is_err());
        assert!(read_bounded(io::repeat(b' '), &CancellationToken::default()).is_err());
    }
    #[test]
    fn capability_validation_rejects_visuals_and_duplicate_ids() {
        let mut wire = decode_request(VALID.as_bytes()).unwrap();
        wire.options[1].id = wire.options[0].id.clone();
        let mut request = FiniteChoiceRequest {
            selector: ModelSelector::default(),
            input: wire.input,
            options: wire.options,
        };
        assert!(request.validate_for(&julia_capabilities()).is_err());
        request.options[1].id = "playing".into();
        request
            .input
            .visuals
            .push(teamy_llm_core::finite_choice::VisualInput {
                format: teamy_llm_core::finite_choice::VisualFormat::Png,
                bytes: vec![1],
            });
        assert!(request.validate_for(&julia_capabilities()).is_err());
    }
    #[test]
    fn resident_engine_rejects_bad_requests_before_artifact_reads() {
        let cli = crate::cli::parse([
            "decide",
            "--provider",
            "native",
            "--model-dir",
            "deliberately-missing-artifacts",
            "--model",
            "julia-1",
        ])
        .into_result()
        .unwrap();
        let crate::cli::Command::Decide(options) = cli.value.command else {
            panic!("decide")
        };
        let mut engine = DecisionEngine::new(options).unwrap();
        let error = engine
            .evaluate_json(
                VALID.replace("\"version\":1", "\"version\":2").as_bytes(),
                CancellationToken::default(),
            )
            .unwrap_err();
        assert!(error.to_string().contains("version"));
        assert!(engine.resident.is_none());
        assert!(engine.encoder.is_none());
        let invalid_ids = VALID.replace("\"playing\"", "\"writing\"");
        assert!(
            engine
                .evaluate_json(invalid_ids.as_bytes(), CancellationToken::default())
                .is_err()
        );
        assert!(engine.encoder.is_none());
    }
    #[test]
    fn provider_selection_and_no_positional_shortcut_are_enforced() {
        fn parsed(arguments: &[&str]) -> Result<DecideCli> {
            let cli = crate::cli::parse(arguments.iter().copied()).into_result()?;
            let crate::cli::Command::Decide(options) = cli.value.command else {
                panic!("decide command")
            };
            options.validate()?;
            Ok(options)
        }
        assert!(parsed(&["decide", "--provider", "native", "--model-dir", "model"]).is_ok());
        assert!(parsed(&["decide", "src.txt", "dest.txt"]).is_err());
        assert!(parsed(&["decide", "--provider", "onnx", "--model-dir", "model"]).is_err());
        assert!(
            parsed(&[
                "decide",
                "--provider",
                "native",
                "--model-dir",
                "model",
                "--repeat",
                "0"
            ])
            .is_err()
        );
    }
}
