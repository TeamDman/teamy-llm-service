//! Versioned finite-choice CLI. Requests are data, not authorization to act on a choice.
use std::fs::File;
use std::io::{self, IsTerminal, Read, Write};
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use eyre::{Result, WrapErr, ensure, eyre};
use facet::Facet;
use teamy_cancellation::CancellationToken;
use teamy_llm_burn_julia::{NativeDevice, NativeJuliaMetadata, NativeJuliaProvider};
use teamy_llm_core::ModelSelector;
use teamy_llm_core::finite_choice::{
    BackendCapabilities, ChoiceContent, FiniteChoiceError, FiniteChoiceOption,
    FiniteChoiceProvider, FiniteChoiceRequest, FiniteChoiceResult, ValidatedFiniteChoiceRequest,
    finite_choice,
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

#[derive(Facet)]
#[facet(rename_all = "kebab-case")]
struct DecideOutput {
    version: u32,
    task: String,
    model: String,
    provider: String,
    device: String,
    question: String,
    encoding_policy: EncodingPolicy,
    token_count: usize,
    option_count: usize,
    load_ms: f64,
    inference_ms: Vec<f64>,
    best_option_id: String,
    result: FiniteChoiceResult,
    encoding: Option<EncodedJuliaRequest>,
    onnx_audit: Option<OnnxAudit>,
    native_metadata: Option<NativeJuliaMetadata>,
    profiling_file: Option<String>,
}

#[derive(Debug)]
struct Options {
    request: Option<String>,
    provider: String,
    model: Option<String>,
    model_dir: PathBuf,
    device: String,
    runtime: Option<PathBuf>,
    question: Option<String>,
    repeat: usize,
    include_encoding: bool,
    policy: EncodingPolicy,
    allow_cpu_fallback: bool,
    profile_prefix: Option<PathBuf>,
}

enum Provider {
    Native(Box<NativeJuliaProvider>),
    Onnx(Box<JuliaOnnxProvider>),
}
impl FiniteChoiceProvider for Provider {
    fn capabilities(&self) -> BackendCapabilities {
        match self {
            Self::Native(p) => p.capabilities(),
            Self::Onnx(p) => p.capabilities(),
        }
    }
    fn infer(
        &mut self,
        request: &ValidatedFiniteChoiceRequest<'_>,
    ) -> std::result::Result<FiniteChoiceResult, FiniteChoiceError> {
        match self {
            Self::Native(p) => p.infer(request),
            Self::Onnx(p) => p.infer(request),
        }
    }
}

fn parse(mut args: Vec<String>) -> Result<Options> {
    let mut request = None;
    let mut provider = None;
    let mut model = None;
    let mut model_dir = None;
    let mut device = String::from("cuda");
    let mut runtime = None;
    let mut question = None;
    let mut repeat = 1;
    let mut include_encoding = false;
    let mut policy = EncodingPolicy::default();
    let mut allow_cpu_fallback = false;
    let mut profile_prefix = None;
    let mut seen = std::collections::HashSet::new();
    while !args.is_empty() {
        let flag = args.remove(0);
        ensure!(seen.insert(flag.clone()), "duplicate argument {flag}");
        if flag == "--include-encoding" {
            include_encoding = true;
            continue;
        }
        if flag == "--allow-cpu-fallback" {
            allow_cpu_fallback = true;
            continue;
        }
        ensure!(
            [
                "--request",
                "--provider",
                "--model",
                "--model-dir",
                "--device",
                "--runtime-library",
                "--question",
                "--repeat",
                "--max-tokens",
                "--head-tokens",
                "--output-format",
                "--profile-prefix"
            ]
            .contains(&flag.as_str()),
            "unknown decide argument {flag}; use `teamy-llm decide --help`"
        );
        let value = args
            .first()
            .filter(|v| !v.starts_with("--"))
            .ok_or_else(|| eyre!("{flag} requires a value"))?
            .clone();
        args.remove(0);
        match flag.as_str() {
            "--request" => request = Some(value),
            "--provider" => provider = Some(value),
            "--model" => model = Some(value),
            "--model-dir" => model_dir = Some(PathBuf::from(value)),
            "--device" => device = value,
            "--runtime-library" => runtime = Some(PathBuf::from(value)),
            "--question" => question = Some(value),
            "--profile-prefix" => profile_prefix = Some(PathBuf::from(value)),
            "--repeat" => repeat = value.parse().wrap_err("invalid --repeat")?,
            "--max-tokens" => policy.max_tokens = value.parse().wrap_err("invalid --max-tokens")?,
            "--head-tokens" => {
                policy.head_tokens = value.parse().wrap_err("invalid --head-tokens")?
            }
            "--output-format" => ensure!(
                value == "json",
                "decide currently supports only JSON output"
            ),
            _ => unreachable!(),
        }
    }
    let provider = provider.ok_or_else(|| eyre!("--provider native|onnx is required"))?;
    ensure!(
        ["native", "onnx"].contains(&provider.as_str()),
        "provider must be native or onnx"
    );
    ensure!(
        ["cuda", "cpu"].contains(&device.as_str()),
        "device must be cuda or cpu; no automatic fallback is used"
    );
    ensure!((1..=1000).contains(&repeat), "--repeat must be in 1..=1000");
    ensure!(
        provider != "onnx" || runtime.is_some(),
        "ONNX requires an explicit --runtime-library path"
    );
    ensure!(
        provider != "native" || runtime.is_none(),
        "--runtime-library applies only to ONNX"
    );
    ensure!(
        provider == "onnx" || (!allow_cpu_fallback && profile_prefix.is_none()),
        "--allow-cpu-fallback and --profile-prefix apply only to ONNX"
    );
    ensure!(
        !allow_cpu_fallback || device == "cuda",
        "--allow-cpu-fallback requires --device cuda"
    );
    let model_dir = model_dir
        .ok_or_else(|| eyre!("--model-dir requires a locally acquired Julia artifact directory"))?;
    Ok(Options {
        request,
        provider,
        model,
        model_dir,
        device,
        runtime,
        question,
        repeat,
        include_encoding,
        policy,
        allow_cpu_fallback,
        profile_prefix,
    })
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

#[tracing::instrument(level = "info", skip_all)]
pub fn run(args: Vec<String>, cancellation: CancellationToken) -> Result<()> {
    if args.len() == 1 && ["help", "--help", "-h", "-?", "?"].contains(&args[0].as_str()) {
        print_help();
        return Ok(());
    }
    let mut options = parse(args)?;
    let wire = decode_request(&read_request(options.request.take(), &cancellation)?)?;
    let model = options
        .model
        .or(wire.model)
        .ok_or_else(|| eyre!("specify --model julia-1 or model in the request"))?;
    ensure!(
        ["julia-1", "SupersonicLabs/Julia-1"].contains(&model.as_str()),
        "decide supports Julia-1; requested model {model}"
    );
    if let Some(question) = options.question.or(wire.question) {
        options.policy.question = question;
    }
    options.policy.validate()?;
    let request = FiniteChoiceRequest {
        selector: ModelSelector {
            model: Some(model.clone()),
            model_dir: Some(options.model_dir.display().to_string()),
        },
        input: wire.input,
        options: wire.options,
    };
    // Capability checks precede tokenizer and numerical artifact reads, including unsupported images.
    let validated = request.validate_for(&julia_capabilities())?;
    cancellation.bail_if_cancelled()?;
    let encoding =
        JuliaEncoder::load(&options.model_dir, options.policy.clone())?.encode(&validated)?;
    cancellation.bail_if_cancelled()?;
    let load_started = Instant::now();
    let (mut provider, audit, native_metadata): (
        Provider,
        Option<OnnxAudit>,
        Option<NativeJuliaMetadata>,
    ) = {
        let _span = tracing::info_span!("julia_load", provider = %options.provider, device = %options.device).entered();
        if options.provider == "native" {
            let device = if options.device == "cuda" {
                NativeDevice::Cuda
            } else {
                NativeDevice::Cpu
            };
            let provider = NativeJuliaProvider::load(
                &options.model_dir,
                device,
                options.policy.clone(),
                cancellation.clone(),
            )?;
            let metadata = provider.metadata().clone();
            (Provider::Native(Box::new(provider)), None, Some(metadata))
        } else {
            let provider = JuliaOnnxProvider::load(
                JuliaOnnxOptions {
                    model_dir: options.model_dir.clone(),
                    runtime_library: options.runtime.unwrap(),
                    device: if options.device == "cuda" {
                        OnnxDevice::Cuda { device_id: 0 }
                    } else {
                        OnnxDevice::Cpu
                    },
                    encoding: options.policy.clone(),
                    allow_cpu_fallback: options.allow_cpu_fallback,
                    tf32: false,
                    intra_threads: 1,
                    profile_prefix: options.profile_prefix.clone(),
                    runtime_sha256: None,
                    graph_sha256: None,
                    weights_sha256: None,
                },
                cancellation.clone(),
            )?;
            let audit = provider.audit().clone();
            (Provider::Onnx(Box::new(provider)), Some(audit), None)
        }
    };
    let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
    let mut inference_ms = Vec::with_capacity(options.repeat);
    let mut last = None;
    for index in 0..options.repeat {
        cancellation.bail_if_cancelled()?;
        let _span = tracing::info_span!("julia_decide", repetition = index).entered();
        let started = Instant::now();
        let result = finite_choice(&mut provider, &request)?;
        cancellation.bail_if_cancelled()?;
        inference_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        last = Some(result);
    }
    let result = last.ok_or_else(|| eyre!("no inference result"))?;
    let profiling_file = match &mut provider {
        Provider::Onnx(provider) if options.profile_prefix.is_some() => {
            Some(provider.finish_profiling()?.display().to_string())
        }
        _ => None,
    };
    let output = DecideOutput {
        version: 1,
        task: "finite-choice".into(),
        model,
        provider: options.provider,
        device: options.device,
        question: options.policy.question.clone(),
        encoding_policy: options.policy,
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
        encoding: options.include_encoding.then_some(encoding),
        onnx_audit: audit,
        native_metadata,
        profiling_file,
    };
    let encoded = facet_json::to_string(&output)?;
    cancellation.bail_if_cancelled()?;
    let mut stdout = io::stdout().lock();
    writeln!(stdout, "{encoded}")?;
    Ok(())
}

pub fn print_help() {
    println!(
        "teamy-llm decide --provider native|onnx --model julia-1 --model-dir <path> [--request '@choice.json'|-|<json>] [--device cuda|cpu]"
    );
    println!(
        "ONNX requires --runtime-library <explicit DLL path>. Requested CUDA never silently falls back to CPU."
    );
    println!(
        "ONNX CUDA: --allow-cpu-fallback explicitly permits mixed graph placement. --profile-prefix <path> writes an ORT node trace; JSON returns its actual timestamped filename."
    );
    println!(
        "Input: version 1 JSON containing input {{text, visuals: []}}, options [{{id, content: {{text, visuals: []}}}}], optional model and question."
    );
    println!(
        "Question default: Which option should be chosen?  Julia accepts text and 2–20 options; images are rejected."
    );
    println!(
        "[--question <text>] [--max-tokens 1024] [--head-tokens 256] [--repeat 1] [--include-encoding] [--output-format json]"
    );
    println!(
        "Without --request, read piped UTF-8 JSON from stdin (2 MiB maximum). Quote @file tokens in PowerShell."
    );
    println!(
        "One versioned JSON result is written to stdout. Diagnostics remain on stderr. Ctrl+C and stop-after options request cancellation; executing GPU kernels cannot be preempted."
    );
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
    fn provider_selection_and_no_positional_shortcut_are_enforced() {
        let base = ["--provider", "native", "--model-dir", "model"];
        assert!(parse(base.iter().map(|s| s.to_string()).collect()).is_ok());
        assert!(parse(vec!["src.txt".into(), "dest.txt".into()]).is_err());
        assert!(
            parse(
                ["--provider", "onnx", "--model-dir", "model"]
                    .into_iter()
                    .map(String::from)
                    .collect()
            )
            .is_err()
        );
        assert!(
            parse(
                [
                    "--provider",
                    "native",
                    "--model-dir",
                    "model",
                    "--repeat",
                    "0"
                ]
                .into_iter()
                .map(String::from)
                .collect()
            )
            .is_err()
        );
    }
}
