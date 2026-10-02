//! Resident, quantized GGUF generation using Makepad's native CUDA executor.
//!
//! A dedicated owner thread creates and retains the non-Send GPU session. Callers exchange data
//! and token events through bounded channels; no unsafe Send/Sync implementation is needed.
//! Cancellation observes hash/load/prefill/decode boundaries, never preempts a GPU kernel.

use std::fs::{File, OpenOptions};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use eyre::{Result, WrapErr, ensure, eyre};
use facet::Facet;
use makepad_ai_llm::{
    LlamaArchitecture, LlamaModel, LlamaSamplerState, LlamaSamplingParams, LlamaSession,
    LlamaSessionConfig, LlamaStopReason, MappedRegion,
};
use sha2::{Digest, Sha256};
use teamy_cancellation::CancellationToken;

pub mod chunk_verification;

#[cfg(test)]
mod mapped_weights_tests;

pub const MAKEPAD_REVISION: &str = "9e5e3d2b03214f8c2c37adffa0e308c815662060";
pub const ORCA_MODEL_NAME: &str = "qwen3.8-27b-uncensored-q4-k-m";
pub const ORCA_GGUF_REVISION: &str = "fc437a3374c9977bdc339a8ec106dcd9a0357001";
pub const ORCA_GGUF_BYTES: u64 = 16_810_714_496;
pub const ORCA_GGUF_SHA256: &str =
    "3445102e9cde5d562508642c100a2f5ac3368a5a3f748442811d7a95daee3bec";
// Generated from one guarded pass that also matched ORCA_GGUF_SHA256 and exact size.
// The canonical root binds version, chunk size, file size, whole SHA and ordered digests.
const ORCA_CHUNK_TABLE_SHA256: &str =
    "4c5382f01047942fa3258670a54cd842754fa832bcd15d8e434a9fa99d4ddfd8";
const ORCA_CHUNK_DIGESTS: &[u8] = include_bytes!("../assets/orca-q4-k-m.sha256-chunks.bin");
pub const DEFAULT_CONTEXT_TOKENS: u32 = 8192;
pub const MAX_CONTEXT_TOKENS: u32 = 32768;

/// Bound GPU allocation without silently truncating a caller's request.
pub fn validate_context_tokens(context: u32) -> Result<()> {
    ensure!(
        (128..=MAX_CONTEXT_TOKENS).contains(&context),
        "--context-tokens must be in 128..={MAX_CONTEXT_TOKENS}"
    );
    Ok(())
}

#[derive(Clone, Debug)]
pub struct GgufLoadOptions {
    pub model_path: PathBuf,
    pub max_context: u32,
    pub prefill_batch_size: usize,
    pub device_id: i32,
    pub expected_sha256: Option<String>,
}

impl GgufLoadOptions {
    #[must_use]
    pub fn new(model_path: PathBuf) -> Self {
        Self {
            model_path,
            max_context: DEFAULT_CONTEXT_TOKENS,
            prefill_batch_size: 32,
            device_id: 0,
            expected_sha256: None,
        }
    }

    #[must_use]
    pub fn orca_qwen_q4(model_path: PathBuf) -> Self {
        Self {
            expected_sha256: Some(ORCA_GGUF_SHA256.into()),
            ..Self::new(model_path)
        }
    }
}

/// CPU-only artifact preparation, movable to the CUDA owner after input validation.
///
/// Private fields bind the validated options and metadata to their live file guard and verified
/// view. Dropping a rejected preflight releases these resources without creating a GPU session.
#[derive(Debug)]
pub struct GgufPreflight {
    options: GgufLoadOptions,
    cancellation: CancellationToken,
    started: Instant,
    _model_file: File,
    model_bytes: u64,
    model_sha256: String,
    model_identity_verification: String,
    verified_mapping: Option<Arc<MappedRegion>>,
    model: LlamaModel,
    main_layers: u32,
    ignored_mtp_layers: u32,
    embedding_length: u32,
    feed_forward_length: u32,
}

impl GgufPreflight {
    #[must_use]
    pub fn max_context(&self) -> u32 {
        self.options.max_context
    }

    #[must_use]
    pub fn model_bytes(&self) -> u64 {
        self.model_bytes
    }

    fn check_loading_cancellation(&self, cancellation: &CancellationToken) -> Result<()> {
        self.cancellation.bail_if_cancelled()?;
        cancellation.bail_if_cancelled()
    }
}

#[derive(Clone, Debug, Facet)]
pub struct GgufAudit {
    pub provider: String,
    pub source_revision: String,
    pub model_sha256: String,
    /// The verification scheme used now; a chunk commitment is not a new serial whole SHA.
    pub model_identity_verification: String,
    pub model_bytes: u64,
    pub architecture: String,
    pub gguf_file_type: Option<u32>,
    pub main_layers: u32,
    pub ignored_mtp_layers: u32,
    pub embedding_length: u32,
    pub feed_forward_length: u32,
    pub requested_device_id: i32,
    pub actual_device_name: String,
    pub quantized_weight_residency: bool,
    pub cpu_fallback: bool,
    pub max_context: u32,
    pub prefill_batch_size: usize,
    pub load_ms: f64,
}

#[derive(Clone, Debug)]
pub struct GgufGenerationOptions {
    pub max_new_tokens: usize,
    pub timeout: Option<Duration>,
    pub cancellation: Option<CancellationToken>,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u64,
    /// Optional independently encoded prompt identity. Checked before any GPU prefill.
    pub expected_prompt_token_ids: Option<Vec<usize>>,
}

impl Default for GgufGenerationOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 64,
            timeout: None,
            cancellation: None,
            temperature: 0.0,
            top_p: 1.0,
            seed: 7,
            expected_prompt_token_ids: None,
        }
    }
}

impl GgufGenerationOptions {
    /// Check text and generation bounds without reading artifacts or preparing a GPU session.
    /// Context fit requires independently tokenizing the prompt and checking the load context.
    pub fn validate(&self, rendered_prompt: &str) -> Result<()> {
        validate_generation(rendered_prompt, self)
    }
}

#[derive(Clone, Debug, Facet)]
pub struct GgufGenerationReport {
    pub generated_token_ids: Vec<usize>,
    pub generated_text: String,
    pub prompt_token_count: usize,
    pub prompt_token_ids: Vec<usize>,
    pub stop_reason: String,
    pub prefill_ms: f64,
    pub decode_ms: f64,
}

enum Reply {
    Token { id: usize, text: String },
    Done(Result<GgufGenerationReport>),
}

struct GenerateJob {
    prompt: String,
    options: GgufGenerationOptions,
    stop: CancellationToken,
    replies: mpsc::SyncSender<Reply>,
}

enum Command {
    Generate(GenerateJob),
    Shutdown,
}

struct Owner {
    commands: mpsc::SyncSender<Command>,
    thread: Mutex<Option<thread::JoinHandle<()>>>,
    audit: GgufAudit,
}

#[derive(Debug)]
struct GenerationCancelled {
    reason: String,
}
impl std::fmt::Display for GenerationCancelled {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "GGUF generation cancelled: {}", self.reason)
    }
}
impl std::error::Error for GenerationCancelled {}

#[derive(Debug)]
struct GenerationTimedOut;
impl std::fmt::Display for GenerationTimedOut {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("GGUF generation timed out at a cooperative boundary")
    }
}
impl std::error::Error for GenerationTimedOut {}

#[must_use]
pub fn is_generation_cancelled_error(error: &eyre::Report) -> bool {
    error.downcast_ref::<GenerationCancelled>().is_some()
}

#[must_use]
pub fn is_generation_timeout_error(error: &eyre::Report) -> bool {
    error.downcast_ref::<GenerationTimedOut>().is_some()
}

impl Drop for Owner {
    fn drop(&mut self) {
        let _ = self.commands.send(Command::Shutdown);
        if let Ok(thread) = self.thread.get_mut()
            && let Some(thread) = thread.take()
        {
            let _ = thread.join();
        }
    }
}

/// Cloneable Send+Sync facade. GPU data never crosses its owner-thread boundary.
#[derive(Clone)]
pub struct GgufRuntime {
    owner: Arc<Owner>,
}

impl GgufRuntime {
    pub fn load(options: GgufLoadOptions, cancellation: CancellationToken) -> Result<Self> {
        validate_load(&options)?;
        cancellation.bail_if_cancelled()?;
        validate_cuda_startup(&options)?;
        let preflight = Self::preflight(options, &cancellation)?;
        Self::load_preflight(preflight, cancellation)
    }

    /// Read/hash/parse artifacts and validate their supported layout without initializing CUDA.
    /// Callers may overlap this CPU work with independent tokenizer/input validation, then join
    /// both before calling `load_preflight`. The original cancellation remains bound to the result.
    pub fn preflight(
        options: GgufLoadOptions,
        cancellation: &CancellationToken,
    ) -> Result<GgufPreflight> {
        prepare_preflight(options, cancellation)
    }

    /// Prepare CUDA on its dedicated owner only after the caller accepts the input/preflight.
    /// Both the preflight's original cancellation and this caller's cancellation are honored.
    pub fn load_preflight(
        preflight: GgufPreflight,
        cancellation: CancellationToken,
    ) -> Result<Self> {
        preflight.check_loading_cancellation(&cancellation)?;
        validate_cuda_startup(&preflight.options)?;
        let (commands, receiver) = mpsc::sync_channel(1);
        let (ready_sender, ready_receiver) = mpsc::sync_channel(1);
        let thread = thread::Builder::new()
            .name("teamy-gguf-cuda-owner".into())
            .spawn(move || {
                let prepared = prepare(preflight, cancellation);
                match prepared {
                    Ok((mut session, audit)) => {
                        if ready_sender.send(Ok(audit)).is_err() {
                            return;
                        }
                        while let Ok(command) = receiver.recv() {
                            match command {
                                Command::Shutdown => break,
                                Command::Generate(job) => {
                                    let report = generate(&mut session, &job);
                                    let _ = job.replies.send(Reply::Done(report));
                                }
                            }
                        }
                    }
                    Err(error) => {
                        let _ = ready_sender.send(Err(error.to_string()));
                    }
                }
            })
            .wrap_err("Starting native GGUF owner thread failed")?;
        let ready = ready_receiver
            .recv()
            .map_err(|_| eyre!("GGUF owner stopped during preparation"));
        match ready {
            Ok(Ok(audit)) => Ok(Self {
                owner: Arc::new(Owner {
                    commands,
                    thread: Mutex::new(Some(thread)),
                    audit,
                }),
            }),
            other => {
                let _ = thread.join();
                Err(match other {
                    Ok(Err(error)) => eyre!(error),
                    Err(error) => error,
                    Ok(Ok(_)) => unreachable!(),
                })
            }
        }
    }

    #[must_use]
    pub fn audit(&self) -> &GgufAudit {
        &self.owner.audit
    }

    /// `rendered_prompt` is a complete template-rendered prompt; no chat-template guess occurs here.
    /// Events are delivered on the caller's thread. They concatenate to the final decoded text;
    /// the final event is buffered until the next token or stop, preserving split UTF-8 sequences.
    pub fn generate_with_events(
        &self,
        rendered_prompt: &str,
        options: GgufGenerationOptions,
        sink: &mut dyn FnMut(usize, &str) -> Result<()>,
    ) -> Result<GgufGenerationReport> {
        options.validate(rendered_prompt)?;
        check(&options, &CancellationToken::new(), Instant::now())?;
        let stop = CancellationToken::new();
        let (replies, receiver) = mpsc::sync_channel(8);
        let job = GenerateJob {
            prompt: rendered_prompt.into(),
            options: options.clone(),
            stop: stop.clone(),
            replies,
        };
        self.owner
            .commands
            .try_send(Command::Generate(job))
            .map_err(|error| eyre!("GGUF owner queue is unavailable or full: {error}"))?;
        loop {
            if let Some(cancellation) = &options.cancellation
                && let Err(error) = check_cancellation(cancellation)
            {
                stop.request_cancel("Caller cancelled generation");
                return Err(error);
            }
            match receiver.recv_timeout(Duration::from_millis(100)) {
                Ok(Reply::Token { id, text }) => {
                    if let Err(error) = sink(id, &text) {
                        stop.request_cancel("Generated-token sink failed");
                        return Err(error);
                    }
                }
                Ok(Reply::Done(result)) => return result,
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(eyre!("GGUF owner stopped during generation"));
                }
            }
        }
    }
}

fn prepare_preflight(
    options: GgufLoadOptions,
    cancellation: &CancellationToken,
) -> Result<GgufPreflight> {
    let started = Instant::now();
    validate_load(&options)?;
    cancellation.bail_if_cancelled()?;
    // Hold the same ordinary-write exclusion from verification through mapping.
    // This does not protect against privileged writers or pre-existing mappings.
    let mut model_file = open_model_for_verification(&options.model_path)?;
    let model_bytes = model_file.metadata()?.len();
    let (model_sha256, model_identity_verification, verified_mapping) = {
        let _span = tracing::info_span!("gguf_verify_identity", model_bytes).entered();
        if let Some(expected) = &options.expected_sha256 {
            ensure!(
                expected.len() == 64 && expected.bytes().all(|byte| byte.is_ascii_hexdigit()),
                "Malformed expected GGUF SHA-256"
            );
        }
        if options
            .expected_sha256
            .as_deref()
            .is_some_and(|expected| expected.eq_ignore_ascii_case(ORCA_GGUF_SHA256))
        {
            ensure!(
                model_bytes == ORCA_GGUF_BYTES,
                "GGUF size differs from the frozen identity"
            );
            let mapping =
                Arc::new(MappedRegion::map_file(&options.model_path).map_err(makepad_error)?);
            let commitment = orca_chunk_commitment();
            let receipt =
                chunk_verification::verify_chunks(mapping.as_slice(), &commitment, cancellation)?;
            tracing::debug!(
                verification = "compiled-sha256-chunks-v1",
                verified_bytes = receipt.verified_bytes,
                verified_chunks = receipt.verified_chunks,
                workers = receipt.worker_count,
                elapsed_ms = receipt.elapsed_ms,
                table_sha256 = %receipt.table_sha256,
                "GGUF every-byte identity verification"
            );
            (
                receipt.expected_whole_sha256,
                "compiled-sha256-chunks-v1".into(),
                Some(mapping),
            )
        } else {
            let sha256 = hash_reader(&mut model_file, cancellation)?;
            if let Some(expected) = &options.expected_sha256 {
                ensure!(
                    sha256.eq_ignore_ascii_case(expected),
                    "GGUF SHA-256 differs from the frozen identity"
                );
            }
            (sha256, "full-sha256".into(), None)
        }
    };
    cancellation.bail_if_cancelled()?;
    let model = {
        let _span = tracing::info_span!("gguf_load_model", model_bytes).entered();
        LlamaModel::load(&options.model_path).map_err(makepad_error)?
    };
    cancellation.bail_if_cancelled()?;
    ensure!(
        model.architecture == LlamaArchitecture::Qwen35,
        "This slice supports dense qwen35 GGUF architecture only"
    );
    // llama.cpp's public GGUF file-type enumeration identifies Q4_K_M as 15. Mixed F32/F16
    // auxiliary tensors are retained as declared; this is not a claim that every tensor is 4-bit.
    ensure!(
        model.general.file_type == Some(15),
        "This slice requires GGUF Q4_K_M file type 15"
    );
    let config = model
        .qwen35
        .as_ref()
        .ok_or_else(|| eyre!("Missing dense Qwen configuration"))?;
    let main_layers = config.main_block_count().map_err(makepad_error)?;
    let (ignored_mtp_layers, embedding_length, feed_forward_length) = (
        config.nextn_predict_layers,
        config.embedding_length,
        config.feed_forward_length,
    );
    // MTP is intentionally disabled. Weight inventory and CUDA buffers retain GGUF quantization.
    ensure!(!model.load_mtp(), "Unexpected MTP execution policy");
    cancellation.bail_if_cancelled()?;
    Ok(GgufPreflight {
        options,
        cancellation: cancellation.clone(),
        started,
        _model_file: model_file,
        model_bytes,
        model_sha256,
        model_identity_verification,
        verified_mapping,
        model,
        main_layers,
        ignored_mtp_layers,
        embedding_length,
        feed_forward_length,
    })
}

fn prepare(
    preflight: GgufPreflight,
    cancellation: CancellationToken,
) -> Result<(LlamaSession, GgufAudit)> {
    preflight.check_loading_cancellation(&cancellation)?;
    let options = &preflight.options;
    let model = &preflight.model;
    let main_layers = preflight.main_layers;
    let session = {
        let _span = tracing::info_span!(
            "gguf_prepare_session",
            main_layers,
            max_context = options.max_context,
            prefill_batch_size = options.prefill_batch_size
        )
        .entered();
        let mut stage = "metadata clone and validation".to_owned();
        let mut stage_started = Instant::now();
        let mut progress = |name: &str, fraction: f64| {
            let name = if name.starts_with("load llm upload") || name.starts_with("load llm gguf") {
                "load llm upload"
            } else if name.starts_with("load llm compile") {
                "load llm compile"
            } else {
                name
            };
            if stage != name {
                if !stage.is_empty() {
                    tracing::debug!(
                        stage,
                        elapsed_ms = stage_started.elapsed().as_secs_f64() * 1000.0,
                        "GGUF session preparation phase"
                    );
                }
                stage = name.to_owned();
                stage_started = Instant::now();
                tracing::trace!(stage, fraction, "GGUF session preparation progress");
            }
        };
        let config = LlamaSessionConfig {
            max_context: Some(options.max_context),
            prefill_batch_size: options.prefill_batch_size,
            spec_draft_max: 0,
            ..LlamaSessionConfig::default()
        };
        let session = if let Some(mapping) = &preflight.verified_mapping {
            LlamaSession::from_model_with_mapped_weights_and_progress(
                model,
                config,
                mapping.clone(),
                &mut progress,
            )
        } else {
            LlamaSession::from_model_with_progress(model, config, &mut progress)
        }
        .map_err(makepad_error)?;
        tracing::debug!(
            stage,
            elapsed_ms = stage_started.elapsed().as_secs_f64() * 1000.0,
            "GGUF session preparation phase"
        );
        session
    };
    preflight.check_loading_cancellation(&cancellation)?;
    let actual_device_name = session.device_name();
    ensure!(
        actual_device_name != "metal",
        "CUDA was requested but Metal was selected"
    );
    let audit = GgufAudit {
        provider: "makepad-native-cuda-gguf".into(),
        source_revision: MAKEPAD_REVISION.into(),
        model_sha256: preflight.model_sha256.clone(),
        model_identity_verification: preflight.model_identity_verification.clone(),
        model_bytes: preflight.model_bytes,
        architecture: model.architecture.name().into(),
        gguf_file_type: model.general.file_type,
        main_layers,
        ignored_mtp_layers: preflight.ignored_mtp_layers,
        embedding_length: preflight.embedding_length,
        feed_forward_length: preflight.feed_forward_length,
        requested_device_id: options.device_id,
        actual_device_name,
        quantized_weight_residency: true,
        cpu_fallback: false,
        max_context: options.max_context,
        prefill_batch_size: options.prefill_batch_size,
        load_ms: preflight.started.elapsed().as_secs_f64() * 1000.0,
    };
    tracing::info!(device = %audit.actual_device_name, architecture = %audit.architecture,
        "Resident GGUF CUDA session prepared");
    Ok((session, audit))
}

fn generate(session: &mut LlamaSession, job: &GenerateJob) -> Result<GgufGenerationReport> {
    let started = Instant::now();
    check(&job.options, &job.stop, started)?;
    session.reset().map_err(makepad_error)?;
    let token_ids = session
        .vocab()
        .tokenize(&job.prompt, false, true)
        .map_err(makepad_error)?;
    ensure!(
        !token_ids.is_empty(),
        "Rendered prompt tokenized to an empty sequence"
    );
    let prompt_token_ids = token_ids
        .iter()
        .copied()
        .map(|id| usize::try_from(id).map_err(|_| eyre!("Prompt contains a negative token ID")))
        .collect::<Result<Vec<_>>>()?;
    validate_prompt_identity(
        &prompt_token_ids,
        job.options.expected_prompt_token_ids.as_deref(),
    )?;
    ensure!(
        token_ids
            .len()
            .checked_add(job.options.max_new_tokens)
            .is_some_and(|length| length <= session.max_context()),
        "Prompt and requested completion exceed the resident context limit"
    );
    let prefill_started = Instant::now();
    for batch in token_ids.chunks(session.config().prefill_batch_size) {
        check(&job.options, &job.stop, started)?;
        session.append_tokens(batch).map_err(makepad_error)?;
        check(&job.options, &job.stop, started)?;
    }
    let prefill_ms = prefill_started.elapsed().as_secs_f64() * 1000.0;
    let decode_started = Instant::now();
    let mut sampler = LlamaSamplerState::new(job.options.seed);
    let parameters = LlamaSamplingParams {
        temperature: job.options.temperature,
        top_p: job.options.top_p,
        seed: job.options.seed,
        ..LlamaSamplingParams::default()
    };
    let mut decoder = session.vocab().text_decoder();
    let mut pending: Option<(usize, String)> = None;
    let mut generated_token_ids = Vec::new();
    let mut generated_text = String::new();
    let mut stop_reason = "max-new-tokens".to_string();
    for _ in 0..job.options.max_new_tokens {
        check(&job.options, &job.stop, started)?;
        let result = session
            .continue_sampled_with(1, parameters, &mut sampler)
            .map_err(makepad_error)?;
        check(&job.options, &job.stop, started)?;
        for token in result.token_ids {
            let id =
                usize::try_from(token).map_err(|_| eyre!("Model emitted a negative token ID"))?;
            let text = decoder
                .push_token(session.vocab(), token)
                .ok_or_else(|| eyre!("Model token is outside vocabulary"))?;
            generated_text.push_str(&text);
            generated_token_ids.push(id);
            if let Some((id, text)) = pending.take() {
                send_token(job, id, text)?;
            }
            pending = Some((id, text));
        }
        match result.stop_reason {
            LlamaStopReason::MaxNewTokens => {}
            LlamaStopReason::EndOfSequence => {
                stop_reason = "end-of-sequence".into();
                break;
            }
            LlamaStopReason::PaddingToken => {
                stop_reason = "padding-token".into();
                break;
            }
        }
    }
    let tail = decoder.finish();
    generated_text.push_str(&tail);
    if let Some((id, mut text)) = pending {
        text.push_str(&tail);
        send_token(job, id, text)?;
    }
    check(&job.options, &job.stop, started)?;
    Ok(GgufGenerationReport {
        generated_token_ids,
        generated_text,
        prompt_token_count: token_ids.len(),
        prompt_token_ids,
        stop_reason,
        prefill_ms,
        decode_ms: decode_started.elapsed().as_secs_f64() * 1000.0,
    })
}

fn validate_prompt_identity(actual: &[usize], expected: Option<&[usize]>) -> Result<()> {
    if let Some(expected) = expected {
        ensure!(
            actual == expected,
            "GGUF prompt token IDs differ from the independently encoded prompt; GPU prefill rejected"
        );
    }
    Ok(())
}

fn send_token(job: &GenerateJob, id: usize, text: String) -> Result<()> {
    job.replies
        .send(Reply::Token { id, text })
        .map_err(|_| eyre!("Generated-token recipient disconnected"))
}

fn check(
    options: &GgufGenerationOptions,
    stop: &CancellationToken,
    started: Instant,
) -> Result<()> {
    check_cancellation(stop)?;
    if let Some(token) = &options.cancellation {
        check_cancellation(token)?;
    }
    if let Some(timeout) = options.timeout
        && started.elapsed() >= timeout
    {
        return Err(eyre::Report::new(GenerationTimedOut));
    }
    Ok(())
}

fn check_cancellation(token: &CancellationToken) -> Result<()> {
    if token.is_cancelled() {
        return Err(eyre::Report::new(GenerationCancelled {
            reason: token
                .cancellation_reason()
                .unwrap_or_else(|| "Cancellation requested".into()),
        }));
    }
    Ok(())
}

fn validate_load(options: &GgufLoadOptions) -> Result<()> {
    validate_context_tokens(options.max_context)?;
    ensure!(
        (1..=256).contains(&options.prefill_batch_size) && options.device_id >= 0,
        "Invalid GGUF context, prefill batch size or device ordinal"
    );
    Ok(())
}

fn validate_cuda_startup(options: &GgufLoadOptions) -> Result<()> {
    // Makepad reads device selection from process environment and would write a cuBLAS setting
    // when absent. Require startup configuration; this adapter never mutates that environment.
    ensure!(
        cfg!(any(target_os = "windows", target_os = "linux")),
        "This adapter requires native CUDA on Windows or Linux"
    );
    ensure!(
        configured_device()? == options.device_id,
        "Requested device differs from MAKEPAD_LLAMA_CUDA_DEVICE (default 0); configure it before process startup"
    );
    ensure!(
        matches!(
            std::env::var("CUBLAS_WORKSPACE_CONFIG").as_deref(),
            Ok(":4096:8" | ":16:8")
        ),
        "Set CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8 before process startup"
    );
    Ok(())
}

fn validate_generation(prompt: &str, options: &GgufGenerationOptions) -> Result<()> {
    ensure!(
        !prompt.is_empty()
            && prompt.len() <= 1024 * 1024
            && (1..=4096).contains(&options.max_new_tokens),
        "Empty/oversized prompt or invalid generation token bound"
    );
    ensure!(
        options.timeout.is_none_or(|duration| !duration.is_zero())
            && options.temperature.is_finite()
            && options.temperature >= 0.0
            && options.top_p.is_finite()
            && options.top_p > 0.0
            && options.top_p <= 1.0,
        "Invalid generation timeout or sampling parameters"
    );
    Ok(())
}

fn configured_device() -> Result<i32> {
    match std::env::var("MAKEPAD_LLAMA_CUDA_DEVICE") {
        Ok(value) => {
            let device: i32 = value
                .parse()
                .wrap_err("Invalid MAKEPAD_LLAMA_CUDA_DEVICE")?;
            ensure!(device >= 0, "CUDA ordinal must be nonnegative");
            Ok(device)
        }
        Err(std::env::VarError::NotPresent) => Ok(0),
        Err(error) => Err(eyre!(error)),
    }
}

fn makepad_error(error: impl std::fmt::Display) -> eyre::Report {
    eyre!("{error}")
}

fn open_model_for_verification(path: &Path) -> Result<File> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.share_mode(1); // FILE_SHARE_READ
    }
    options.open(path).wrap_err("Opening local GGUF failed")
}

fn orca_chunk_commitment() -> chunk_verification::TrustedChunkCommitment<'static> {
    chunk_verification::TrustedChunkCommitment {
        version: chunk_verification::CHUNK_TABLE_VERSION,
        chunk_bytes: chunk_verification::DEFAULT_CHUNK_BYTES,
        total_bytes: ORCA_GGUF_BYTES,
        whole_sha256: ORCA_GGUF_SHA256,
        table_sha256: ORCA_CHUNK_TABLE_SHA256,
        digests: ORCA_CHUNK_DIGESTS,
    }
}

fn hash_reader(file: &mut File, cancellation: &CancellationToken) -> Result<String> {
    let mut digest = Sha256::new();
    let mut bytes = vec![0; 1024 * 1024];
    let mut read_duration = Duration::ZERO;
    let mut digest_duration = Duration::ZERO;
    let mut total_bytes = 0u64;
    loop {
        cancellation.bail_if_cancelled()?;
        let read_started = Instant::now();
        let read = file.read(&mut bytes)?;
        read_duration += read_started.elapsed();
        if read == 0 {
            break;
        }
        let digest_started = Instant::now();
        digest.update(&bytes[..read]);
        digest_duration += digest_started.elapsed();
        total_bytes += read as u64;
    }
    tracing::debug!(
        total_bytes,
        read_ms = read_duration.as_secs_f64() * 1000.0,
        digest_ms = digest_duration.as_secs_f64() * 1000.0,
        "GGUF identity verification phases"
    );
    Ok(format!("{:x}", digest.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn frozen_orca_chunk_table_matches_the_compiled_canonical_commitment() {
        let table = orca_chunk_commitment();
        assert_eq!(ORCA_CHUNK_DIGESTS.len(), 1002 * 32);
        assert_eq!(
            chunk_verification::canonical_table_sha256(
                table.version,
                table.chunk_bytes,
                table.total_bytes,
                table.whole_sha256,
                table.digests,
            )
            .unwrap(),
            ORCA_CHUNK_TABLE_SHA256
        );
    }
    #[test]
    fn configured_context_has_explicit_allocation_bounds() {
        assert_eq!(
            GgufLoadOptions::new("example.gguf".into()).max_context,
            8192
        );
        for context in [128, 1024, 8192, 32768] {
            validate_context_tokens(context).unwrap();
        }
        for context in [0, 127, 32769, u32::MAX] {
            assert!(validate_context_tokens(context).is_err());
        }
    }
    #[test]
    fn facade_is_send_and_sync() {
        fn assert_traits<T: Send + Sync>() {}
        assert_traits::<GgufRuntime>();
        assert_traits::<GgufPreflight>();
    }

    #[test]
    fn artifact_preflight_rejects_invalid_bounds_before_file_access() {
        let directory = tempfile::tempdir().unwrap();
        let missing = directory.path().join("no-such-model.gguf");
        let mut options = GgufLoadOptions::new(missing.clone());
        options.max_context = 127;
        let error = GgufRuntime::preflight(options, &CancellationToken::new()).unwrap_err();
        assert!(error.to_string().contains("--context-tokens"));
        let mut options = GgufLoadOptions::new(missing.clone());
        options.prefill_batch_size = 0;
        let error = GgufRuntime::preflight(options, &CancellationToken::new()).unwrap_err();
        assert!(error.to_string().contains("prefill batch size"));
        let mut options = GgufLoadOptions::new(missing);
        options.device_id = -1;
        let error = GgufRuntime::preflight(options, &CancellationToken::new()).unwrap_err();
        assert!(error.to_string().contains("device ordinal"));
    }

    #[test]
    fn artifact_preflight_honors_cancellation_before_file_access_or_cuda() {
        let directory = tempfile::tempdir().unwrap();
        let options = GgufLoadOptions::new(directory.path().join("no-such-model.gguf"));
        let cancellation = CancellationToken::new();
        cancellation.request_cancel("artifact preflight fixture cancelled");
        let error = GgufRuntime::preflight(options, &cancellation).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("artifact preflight fixture cancelled")
        );
    }
    #[test]
    fn prompt_identity_rejects_any_token_or_order_mismatch() {
        assert!(validate_prompt_identity(&[1, 2], Some(&[1, 2])).is_ok());
        assert!(validate_prompt_identity(&[1, 2], None).is_ok());
        assert!(validate_prompt_identity(&[1, 2], Some(&[2, 1])).is_err());
        assert!(validate_prompt_identity(&[1, 2], Some(&[1])).is_err());
    }
    #[test]
    fn validates_bounds_without_loading_artifacts() {
        let mut options = GgufLoadOptions::new(PathBuf::from("fixture.gguf"));
        assert!(validate_load(&options).is_ok());
        options.max_context = 0;
        assert!(validate_load(&options).is_err());
        let mut generation = GgufGenerationOptions::default();
        assert!(generation.validate("fixture").is_ok());
        generation.temperature = f32::NAN;
        assert!(generation.validate("fixture").is_err());
    }
    #[test]
    fn public_preflight_rejects_invalid_requests_without_a_runtime() {
        let mut options = GgufGenerationOptions::default();
        assert!(options.validate("").is_err());
        options.timeout = Some(Duration::ZERO);
        assert!(options.validate("fixture").is_err());
        options.timeout = Some(Duration::from_millis(1));
        assert!(options.validate("fixture").is_ok());
        options.max_new_tokens = 4097;
        assert!(options.validate("fixture").is_err());
    }
    #[test]
    fn cancellation_stops_at_boundary() {
        let token = CancellationToken::new();
        token.request_cancel("fixture stop");
        let error = check(&GgufGenerationOptions::default(), &token, Instant::now()).unwrap_err();
        assert!(is_generation_cancelled_error(&error));
    }
    #[test]
    fn timeout_is_observed_without_sleeping() {
        let options = GgufGenerationOptions {
            timeout: Some(Duration::from_millis(1)),
            ..Default::default()
        };
        let error = check(
            &options,
            &CancellationToken::new(),
            Instant::now() - Duration::from_secs(1),
        )
        .unwrap_err();
        assert!(is_generation_timeout_error(&error));
    }

    #[test]
    fn typed_errors_survive_the_owner_message_boundary() {
        let (sender, receiver) = mpsc::sync_channel(1);
        thread::spawn(move || {
            sender
                .send(Reply::Done(Err(eyre::Report::new(GenerationCancelled {
                    reason: "fixture stop".into(),
                }))))
                .unwrap()
        })
        .join()
        .unwrap();
        let Reply::Done(result) = receiver.recv().unwrap() else {
            panic!("Expected completion")
        };
        assert!(is_generation_cancelled_error(&result.unwrap_err()));
    }
}
