use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use teamy_cancellation::CancellationToken;
use teamy_llm_burn_qwen35::burn_backend::{BurnLlmSupportState, inspect_burn_runtime_support};
use teamy_llm_burn_qwen35::burn_text::{
    BurnTextGenerationOptions, CachedBurnTextCudaRuntime,
    GenerationEventSink as BackendGenerationEventSink,
    generate_with_cached_burn_text_runtime_to_sink,
    inspect_top_tokens_with_cached_burn_text_runtime, is_generation_cancelled_error,
    is_generation_timeout_error, load_cached_burn_text_cuda_runtime,
    score_continuation_with_cached_burn_text_runtime, verify_required_cuda_runtime,
};
use teamy_llm_burn_qwen35::model::{
    DEFAULT_LLM_MODEL_NAME, LlmModelArtifacts, LlmModelPreparationState, inspect_model_dir,
    list_known_llm_model_statuses, prepare_burn_text_runtime_bundle, prepare_known_llm_model,
    render_model_report, resolve_llm_model_dir,
};
use teamy_llm_burn_qwen35::paths::{AppHome, CacheHome};
use teamy_llm_burn_qwen35::runtime::render_qwen_single_turn_prompt;
use teamy_llm_core::{
    GenerateEvent, GenerateRequest, GenerateResult, ModelLocationStatus, ModelPreparationState,
    ModelReport, ModelSelector, ModelStatus, PrepareEvent, PrepareModelRequest, PrepareModelResult,
    ServiceError, ServiceErrorKind, ServiceHealth, ThinkingMode,
};
pub use teamy_llm_makepad_gguf::ORCA_MODEL_NAME;
use teamy_llm_makepad_gguf::{GgufGenerationOptions, GgufLoadOptions, GgufRuntime};

mod model_defaults;
mod qwen_chat;
mod tokenizer_cache;

pub use model_defaults::ModelDefaultInfo;

pub trait GenerateEventSink {
    fn send(&mut self, event: GenerateEvent) -> Result<(), ServiceError>;
}

/// Parameters for a local CUDA next-token candidate inspection.
#[derive(Clone, Debug)]
pub struct TopTokensRequest {
    pub selector: ModelSelector,
    pub system_prompt: Option<String>,
    pub user_prompt: String,
    pub thinking_mode: ThinkingMode,
    /// Uses `user_prompt` unchanged rather than rendering the normal Qwen chat envelope.
    pub raw_prompt: bool,
    /// Preserves an already-tokenized prompt for diagnostic continuation without re-tokenizing.
    pub prompt_token_ids: Option<Vec<u32>>,
    pub top_k: usize,
}

/// One decoded candidate from a local CUDA next-token inspection.
#[derive(Clone, Debug, PartialEq)]
pub struct TopToken {
    pub token_id: u32,
    pub text: String,
    pub logit: f32,
}

/// Next-token candidates and the exact prompt rendered for their inspection.
#[derive(Clone, Debug, PartialEq)]
pub struct TopTokensResult {
    pub rendered_prompt: String,
    pub input_token_ids: Vec<u32>,
    pub input_token_count: usize,
    pub top_tokens: Vec<TopToken>,
}

/// Parameters for ranking candidate continuations by the local model's conditional likelihood.
#[derive(Clone, Debug)]
pub struct RankContinuationsRequest {
    pub selector: ModelSelector,
    pub system_prompt: Option<String>,
    pub user_prompt: String,
    pub thinking_mode: ThinkingMode,
    pub raw_prompt: bool,
    pub candidates: Vec<String>,
}

/// One candidate ranked by length-normalized causal likelihood.
#[derive(Clone, Debug, PartialEq)]
pub struct RankedContinuation {
    pub candidate: String,
    pub token_count: usize,
    pub total_log_probability: f64,
    pub mean_log_probability: f64,
    pub mean_negative_log_likelihood: f64,
    pub perplexity: f64,
}

/// Candidate rankings and the prompt used to score them.
#[derive(Clone, Debug, PartialEq)]
pub struct RankContinuationsResult {
    pub rendered_prompt: String,
    pub prompt_token_count: usize,
    pub rankings: Vec<RankedContinuation>,
}

impl<F> GenerateEventSink for F
where
    F: FnMut(GenerateEvent) -> Result<(), ServiceError>,
{
    fn send(&mut self, event: GenerateEvent) -> Result<(), ServiceError> {
        self(event)
    }
}

#[derive(Clone)]
pub struct TeamyLlmService {
    app_home: AppHome,
    cache_home: CacheHome,
    inspected_models: Arc<Mutex<HashMap<PathBuf, LlmModelArtifacts>>>,
    cached_burn_runtimes: Arc<Mutex<HashMap<PathBuf, Arc<CachedBurnTextCudaRuntime>>>>,
    cached_gguf_runtimes: Arc<Mutex<HashMap<PathBuf, Arc<GgufRuntime>>>>,
    cached_tokenizers: tokenizer_cache::TokenizerCache,
    // A 24 GB device cannot retain both the legacy FP16 and 27B Q4 models at once.
    // Serialize service GPU work, and release the other backend when selection changes.
    gpu_queue: Arc<Mutex<()>>,
    generation_queues: Arc<Mutex<HashMap<PathBuf, Arc<Mutex<()>>>>>,
}

impl TeamyLlmService {
    pub fn new(app_home: AppHome, cache_home: CacheHome) -> Self {
        Self {
            app_home,
            cache_home,
            inspected_models: Arc::new(Mutex::new(HashMap::new())),
            cached_burn_runtimes: Arc::new(Mutex::new(HashMap::new())),
            cached_gguf_runtimes: Arc::new(Mutex::new(HashMap::new())),
            cached_tokenizers: tokenizer_cache::TokenizerCache::default(),
            gpu_queue: Arc::new(Mutex::new(())),
            generation_queues: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn from_default_dirs() -> Result<Self, ServiceError> {
        let app_home = AppHome::resolve().map_err(map_internal_error)?;
        let cache_home = CacheHome::resolve().map_err(map_internal_error)?;
        Ok(Self::new(app_home, cache_home))
    }

    #[must_use]
    pub fn health(&self) -> ServiceHealth {
        ServiceHealth {
            service_name: "teamy-llm-service".to_owned(),
            version: env!("CARGO_PKG_VERSION").to_owned(),
            backend: "burn-qwen35,makepad-gguf-cuda".to_owned(),
            ready: true,
        }
    }

    pub fn list_models(&self) -> Result<Vec<ModelStatus>, ServiceError> {
        let statuses = list_known_llm_model_statuses(&self.app_home, &self.cache_home)
            .map_err(map_internal_error)?;
        Ok(statuses.into_iter().map(map_model_status).collect())
    }

    /// Report the explicit preference or the existing registered/built-in fallback.
    pub fn default_model(&self) -> Result<ModelDefaultInfo, ServiceError> {
        model_defaults::default_model(&self.app_home, &self.cache_home)
    }

    /// Select an existing, compatible managed model without changing registration order.
    pub fn set_default_model(&self, model: &str) -> Result<ModelDefaultInfo, ServiceError> {
        model_defaults::set_default_model(&self.app_home, &self.cache_home, model)
    }

    pub fn inspect_model(&self, selector: ModelSelector) -> Result<ModelReport, ServiceError> {
        let model_dir = self.resolve_model_dir(&selector)?;
        let artifacts = self.inspect_artifacts(&model_dir)?;
        let burn_support = inspect_burn_runtime_support(&artifacts).map_err(map_internal_error)?;
        Ok(ModelReport {
            model_root: artifacts.root.display().to_string(),
            display_name: artifacts.metadata.display_name.clone(),
            family: artifacts.metadata.family.clone(),
            architecture: artifacts.metadata.architecture.clone(),
            quantization: artifacts.metadata.quantization.clone(),
            supports_tool_calling: artifacts.metadata.supports_tool_calling,
            supports_vision: artifacts.metadata.supports_vision,
            burn_support_state: format!("{:?}", burn_support.state),
            details: render_model_report(&artifacts),
        })
    }

    pub async fn prepare_model(
        &self,
        request: PrepareModelRequest,
        mut events: impl FnMut(PrepareEvent) -> Result<(), ServiceError>,
    ) -> Result<PrepareModelResult, ServiceError> {
        events(PrepareEvent::Started {
            model: request.model.clone(),
        })?;
        let prepared = prepare_known_llm_model(
            &self.app_home,
            &self.cache_home,
            &request.model,
            request.overwrite,
            request.include_mmproj,
            request.download_main_model_file,
        )
        .await
        .map_err(map_internal_error)?;
        self.cached_tokenizers
            .invalidate(&prepared.artifacts.tokenizer_path)
            .map_err(map_internal_error)?;
        // Preparation changed the files even if a later observer or export fails.
        // Publish the new artifact snapshot and discard resident runtimes before callbacks.
        self.inspected_models
            .lock()
            .map_err(|_| {
                ServiceError::new(ServiceErrorKind::Internal, "model cache mutex poisoned")
            })?
            .insert(prepared.artifacts.root.clone(), prepared.artifacts.clone());
        self.cached_burn_runtimes
            .lock()
            .map_err(|_| {
                ServiceError::new(
                    ServiceErrorKind::Internal,
                    "Burn runtime cache mutex poisoned",
                )
            })?
            .remove(&prepared.artifacts.root);
        self.cached_gguf_runtimes
            .lock()
            .map_err(|_| {
                ServiceError::new(
                    ServiceErrorKind::Internal,
                    "GGUF runtime cache mutex poisoned",
                )
            })?
            .remove(&prepared.artifacts.root);
        events(PrepareEvent::ModelPrepared {
            model_dir: prepared.managed_dir.display().to_string(),
        })?;
        if request.export_burn_text {
            events(PrepareEvent::BurnTextExportStarted)?;
            let artifacts = prepared.artifacts.clone();
            let overwrite = request.overwrite;
            let burn_text_dtype = request.burn_text_dtype.clone();
            let message = tokio::task::spawn_blocking(move || {
                prepare_burn_text_runtime_bundle(&artifacts, overwrite, burn_text_dtype.as_deref())
            })
            .await
            .map_err(|error| {
                ServiceError::new(
                    ServiceErrorKind::Internal,
                    format!("Burn text export task failed: {error}"),
                )
            })?
            .map_err(map_internal_error)?;
            events(PrepareEvent::BurnTextExportCompleted { message })?;
        }
        let model_dir = prepared.managed_dir.display().to_string();
        events(PrepareEvent::Completed {
            model_dir: model_dir.clone(),
        })?;
        Ok(PrepareModelResult {
            model_dir,
            registered_model_dirs: prepared
                .registered_model_dirs
                .iter()
                .map(|path| path.display().to_string())
                .collect(),
            registration_warning: prepared.registration_warning,
        })
    }

    pub fn generate(
        &self,
        request: GenerateRequest,
        sink: &mut impl GenerateEventSink,
    ) -> Result<GenerateResult, ServiceError> {
        self.generate_with_cancellation(request, None, sink)
    }

    #[tracing::instrument(level = "info", name = "llm_generate", skip_all)]
    pub fn generate_with_cancellation(
        &self,
        request: GenerateRequest,
        cancellation_token: Option<CancellationToken>,
        sink: &mut impl GenerateEventSink,
    ) -> Result<GenerateResult, ServiceError> {
        check_cancelled(cancellation_token.as_ref())?;
        if request.max_new_tokens == 0 || request.timeout_ms == Some(0) {
            return Err(ServiceError::new(
                ServiceErrorKind::InvalidRequest,
                "Generation token limit and timeout must be greater than zero",
            ));
        }
        if let Some(context) = request.context_tokens {
            teamy_llm_makepad_gguf::validate_context_tokens(context)
                .map_err(map_invalid_request_error)?;
        }
        let _gpu_guard = self.lock_gpu_queue(cancellation_token.as_ref())?;
        check_cancelled(cancellation_token.as_ref())?;
        let model_dir = self.resolve_model_dir(&request.selector)?;
        check_cancelled(cancellation_token.as_ref())?;
        let queue = self.queue_for_model(&model_dir)?;
        let _queue_guard = {
            let _span = tracing::info_span!("llm_wait_model_queue").entered();
            queue.lock().map_err(|_| {
                ServiceError::new(
                    ServiceErrorKind::Internal,
                    "model generation queue mutex poisoned",
                )
            })?
        };

        check_cancelled(cancellation_token.as_ref())?;
        let artifacts = self.inspect_artifacts(&model_dir)?;
        check_cancelled(cancellation_token.as_ref())?;
        if artifacts.metadata.model_name == ORCA_MODEL_NAME {
            return self.generate_gguf(request, cancellation_token, &artifacts, sink);
        }
        if request.context_tokens.is_some() {
            return Err(ServiceError::new(
                ServiceErrorKind::InvalidRequest,
                "--context-tokens is supported by the Orca GGUF provider; legacy Burn context is unchanged",
            ));
        }
        {
            let _span = tracing::info_span!("llm_cuda_preflight").entered();
            verify_required_cuda_runtime().map_err(map_generation_error)?;
        }
        let burn_support = inspect_burn_runtime_support(&artifacts).map_err(map_internal_error)?;
        if burn_support.state != BurnLlmSupportState::Ready {
            return Err(ServiceError::new(
                ServiceErrorKind::ModelNotPrepared,
                format!(
                    "model at {} is not ready for Burn generation: {}",
                    artifacts.root.display(),
                    burn_support.reason
                ),
            ));
        }
        sink.send(GenerateEvent::Started {
            model_dir: artifacts.root.display().to_string(),
        })?;
        let rendered = render_qwen_single_turn_prompt(
            request.system_prompt.as_deref(),
            &request.user_prompt,
            request.thinking_mode,
        );
        sink.send(GenerateEvent::RenderedPrompt {
            prompt: rendered.prompt.clone(),
        })?;
        let token_ids = {
            let _span = tracing::info_span!("llm_tokenize").entered();
            self.tokenize_prompt(&artifacts.tokenizer_path, &rendered.prompt)?
        };
        tracing::debug!(prompt_tokens = token_ids.len(), "prompt tokenized");
        check_cancelled(cancellation_token.as_ref())?;
        let cached_runtime = self.cached_burn_runtime(&artifacts)?;
        check_cancelled(cancellation_token.as_ref())?;
        let options = BurnTextGenerationOptions {
            max_new_tokens: request.max_new_tokens,
            generation_timeout: request.timeout_ms.map(Duration::from_millis),
            cancellation_token,
        };
        let mut backend_sink = ServiceBackendSink { sink };
        let report = generate_with_cached_burn_text_runtime_to_sink(
            &cached_runtime,
            &artifacts,
            &token_ids,
            &options,
            &mut backend_sink,
        )
        .map_err(map_generation_error)?;
        sink.send(GenerateEvent::Completed {
            generated_text: report.generated_text.clone(),
        })?;
        Ok(GenerateResult {
            rendered_prompt: rendered.prompt,
            generated_token_ids: report.generated_token_ids,
            generated_text: report.generated_text,
        })
    }

    /// Return the highest next-token logits from the configured local CUDA model.
    ///
    /// Unless `raw_prompt` is set, the prompt is rendered with the same Qwen template and thinking
    /// mode as normal generation.
    pub fn inspect_top_tokens(
        &self,
        request: TopTokensRequest,
    ) -> Result<TopTokensResult, ServiceError> {
        let _gpu_guard = self.lock_gpu_queue(None)?;
        if request.top_k == 0 {
            return Err(ServiceError::new(
                ServiceErrorKind::InvalidRequest,
                "top-k must be at least 1",
            ));
        }
        verify_required_cuda_runtime().map_err(map_generation_error)?;
        let model_dir = self.resolve_model_dir(&request.selector)?;
        let queue = self.queue_for_model(&model_dir)?;
        let _queue_guard = queue.lock().map_err(|_| {
            ServiceError::new(
                ServiceErrorKind::Internal,
                "model generation queue mutex poisoned",
            )
        })?;

        let artifacts = self.inspect_artifacts(&model_dir)?;
        let burn_support = inspect_burn_runtime_support(&artifacts).map_err(map_internal_error)?;
        if burn_support.state != BurnLlmSupportState::Ready {
            return Err(ServiceError::new(
                ServiceErrorKind::ModelNotPrepared,
                format!(
                    "model at {} is not ready for Burn top-token inspection: {}",
                    artifacts.root.display(),
                    burn_support.reason
                ),
            ));
        }
        if request.raw_prompt && request.system_prompt.is_some() {
            return Err(ServiceError::new(
                ServiceErrorKind::InvalidRequest,
                "--raw-prompt cannot be combined with a system prompt",
            ));
        }
        let (rendered_prompt, token_ids) = if let Some(token_ids) = request.prompt_token_ids {
            if token_ids.is_empty() {
                return Err(ServiceError::new(
                    ServiceErrorKind::InvalidRequest,
                    "pre-tokenized top-token inspection requires at least one prompt token",
                ));
            }
            (request.user_prompt, token_ids)
        } else {
            let rendered_prompt = if request.raw_prompt {
                request.user_prompt
            } else {
                render_qwen_single_turn_prompt(
                    request.system_prompt.as_deref(),
                    &request.user_prompt,
                    request.thinking_mode,
                )
                .prompt
            };
            let token_ids = self.tokenize_prompt(&artifacts.tokenizer_path, &rendered_prompt)?;
            (rendered_prompt, token_ids)
        };
        let cached_runtime = self.cached_burn_runtime(&artifacts)?;
        let candidates = inspect_top_tokens_with_cached_burn_text_runtime(
            &cached_runtime,
            &token_ids,
            request.top_k,
        )
        .map_err(map_generation_error)?;
        let tokenizer = self
            .cached_tokenizers
            .get(&artifacts.tokenizer_path)
            .map_err(map_invalid_request_error)?;
        let top_tokens = candidates
            .into_iter()
            .map(|candidate| {
                let token_id = u32::try_from(candidate.token_id).map_err(|_| {
                    ServiceError::new(
                        ServiceErrorKind::Internal,
                        format!("Token id {} exceeded u32 range", candidate.token_id),
                    )
                })?;
                let text = tokenizer.decode(&[token_id], false).map_err(|error| {
                    ServiceError::new(ServiceErrorKind::Internal, error.to_string())
                })?;
                Ok(TopToken {
                    token_id,
                    text,
                    logit: candidate.logit,
                })
            })
            .collect::<Result<Vec<_>, ServiceError>>()?;
        Ok(TopTokensResult {
            rendered_prompt,
            input_token_count: token_ids.len(),
            input_token_ids: token_ids,
            top_tokens,
        })
    }

    /// Rank candidate continuations by their length-normalized conditional likelihood.
    pub fn rank_continuations(
        &self,
        request: RankContinuationsRequest,
    ) -> Result<RankContinuationsResult, ServiceError> {
        let _gpu_guard = self.lock_gpu_queue(None)?;
        if request.candidates.is_empty() {
            return Err(ServiceError::new(
                ServiceErrorKind::InvalidRequest,
                "rank command requires at least one candidate continuation",
            ));
        }
        if request.raw_prompt && request.system_prompt.is_some() {
            return Err(ServiceError::new(
                ServiceErrorKind::InvalidRequest,
                "--raw-prompt cannot be combined with a system prompt",
            ));
        }
        verify_required_cuda_runtime().map_err(map_generation_error)?;
        let model_dir = self.resolve_model_dir(&request.selector)?;
        let queue = self.queue_for_model(&model_dir)?;
        let _queue_guard = queue.lock().map_err(|_| {
            ServiceError::new(
                ServiceErrorKind::Internal,
                "model generation queue mutex poisoned",
            )
        })?;

        let artifacts = self.inspect_artifacts(&model_dir)?;
        let burn_support = inspect_burn_runtime_support(&artifacts).map_err(map_internal_error)?;
        if burn_support.state != BurnLlmSupportState::Ready {
            return Err(ServiceError::new(
                ServiceErrorKind::ModelNotPrepared,
                format!(
                    "model at {} is not ready for Burn continuation ranking: {}",
                    artifacts.root.display(),
                    burn_support.reason
                ),
            ));
        }
        let rendered_prompt = if request.raw_prompt {
            request.user_prompt
        } else {
            render_qwen_single_turn_prompt(
                request.system_prompt.as_deref(),
                &request.user_prompt,
                request.thinking_mode,
            )
            .prompt
        };
        let prompt_token_ids = self.tokenize_prompt(&artifacts.tokenizer_path, &rendered_prompt)?;
        let tokenizer = self
            .cached_tokenizers
            .get(&artifacts.tokenizer_path)
            .map_err(map_invalid_request_error)?;
        let cached_runtime = self.cached_burn_runtime(&artifacts)?;
        let mut rankings = Vec::with_capacity(request.candidates.len());
        for candidate in request.candidates {
            let candidate_ids = tokenizer
                .encode(candidate.as_str(), false)
                .map_err(|error| {
                    ServiceError::new(ServiceErrorKind::InvalidRequest, error.to_string())
                })?
                .get_ids()
                .to_vec();
            if candidate_ids.is_empty() {
                return Err(ServiceError::new(
                    ServiceErrorKind::InvalidRequest,
                    format!("candidate {:?} produced no tokens", candidate),
                ));
            }
            let score = score_continuation_with_cached_burn_text_runtime(
                &cached_runtime,
                &prompt_token_ids,
                &candidate_ids,
            )
            .map_err(map_generation_error)?;
            rankings.push(RankedContinuation {
                candidate,
                token_count: score.token_count,
                total_log_probability: score.total_log_probability,
                mean_log_probability: score.mean_log_probability,
                mean_negative_log_likelihood: score.mean_negative_log_likelihood,
                perplexity: score.perplexity,
            });
        }
        rankings.sort_by(|left, right| {
            left.mean_negative_log_likelihood
                .total_cmp(&right.mean_negative_log_likelihood)
                .then_with(|| left.candidate.cmp(&right.candidate))
        });
        Ok(RankContinuationsResult {
            rendered_prompt,
            prompt_token_count: prompt_token_ids.len(),
            rankings,
        })
    }

    fn resolve_model_dir(&self, selector: &ModelSelector) -> Result<PathBuf, ServiceError> {
        if selector.model.is_none()
            && selector.model_dir.is_none()
            && let Some(model_dir) =
                model_defaults::explicit_model_dir(&self.app_home, &self.cache_home)?
        {
            return Ok(model_dir);
        }
        resolve_llm_model_dir(
            &self.app_home,
            &self.cache_home,
            selector.model.as_deref(),
            selector.model_dir.as_deref().map(Path::new),
        )
        .map_err(map_internal_error)
    }

    fn lock_gpu_queue(
        &self,
        cancellation: Option<&CancellationToken>,
    ) -> Result<std::sync::MutexGuard<'_, ()>, ServiceError> {
        let _span = tracing::info_span!("llm_wait_gpu_queue").entered();
        loop {
            check_cancelled(cancellation)?;
            match self.gpu_queue.try_lock() {
                Ok(guard) => return Ok(guard),
                Err(std::sync::TryLockError::WouldBlock) => {
                    std::thread::sleep(Duration::from_millis(20))
                }
                Err(std::sync::TryLockError::Poisoned(_)) => {
                    return Err(ServiceError::new(
                        ServiceErrorKind::Internal,
                        "GPU queue mutex poisoned",
                    ));
                }
            }
        }
    }

    fn generate_gguf(
        &self,
        request: GenerateRequest,
        cancellation: Option<CancellationToken>,
        artifacts: &LlmModelArtifacts,
        sink: &mut impl GenerateEventSink,
    ) -> Result<GenerateResult, ServiceError> {
        let mut load_options = GgufLoadOptions::orca_qwen_q4(artifacts.model_path.clone());
        if let Some(context) = request.context_tokens {
            load_options.max_context = context;
        }
        sink.send(GenerateEvent::Started {
            model_dir: artifacts.root.display().to_string(),
        })?;
        let rendered = qwen_chat::render_orca_qwen_single_turn_prompt(
            request.system_prompt.as_deref(),
            &request.user_prompt,
            request.thinking_mode,
        );
        let mut generation_options = GgufGenerationOptions {
            max_new_tokens: request.max_new_tokens,
            timeout: request.timeout_ms.map(Duration::from_millis),
            cancellation: cancellation.clone(),
            ..Default::default()
        };
        generation_options
            .validate(&rendered)
            .map_err(map_invalid_request_error)?;
        sink.send(GenerateEvent::RenderedPrompt {
            prompt: rendered.clone(),
        })?;
        let expected_prompt_token_ids: Vec<usize> = {
            let _span =
                tracing::info_span!("llm_tokenize", provider = "makepad-gguf-cuda").entered();
            self.tokenize_prompt(&artifacts.tokenizer_path, &rendered)?
                .into_iter()
                .map(|id| id as usize)
                .collect()
        };
        if expected_prompt_token_ids
            .len()
            .checked_add(request.max_new_tokens)
            .is_none_or(|total| total > load_options.max_context as usize)
        {
            return Err(ServiceError::new(
                ServiceErrorKind::InvalidRequest,
                format!(
                    "Formatted prompt uses {} tokens and requests {} output tokens, exceeding the {}-token context; raise --context-tokens or reduce input/--max-new-tokens",
                    expected_prompt_token_ids.len(),
                    request.max_new_tokens,
                    load_options.max_context
                ),
            ));
        }
        generation_options.expected_prompt_token_ids = Some(expected_prompt_token_ids);
        // The global GPU guard held by our caller keeps cache replacement and generation serial.
        self.cached_burn_runtimes
            .lock()
            .map_err(|_| {
                ServiceError::new(
                    ServiceErrorKind::Internal,
                    "Burn runtime cache mutex poisoned",
                )
            })?
            .clear();
        let runtime = {
            let mut runtimes = self.cached_gguf_runtimes.lock().map_err(|_| {
                ServiceError::new(
                    ServiceErrorKind::Internal,
                    "GGUF runtime cache mutex poisoned",
                )
            })?;
            runtimes.retain(|root, _| root == &artifacts.root);
            // One resident arena per model. A capacity change must not retain a
            // second weight copy or silently reuse a smaller KV allocation.
            if runtimes
                .get(&artifacts.root)
                .is_some_and(|runtime| runtime.audit().max_context != load_options.max_context)
            {
                tracing::info!(
                    context_tokens = load_options.max_context,
                    "replacing GGUF context capacity"
                );
                runtimes.remove(&artifacts.root);
            }
            if let Some(runtime) = runtimes.get(&artifacts.root) {
                tracing::debug!(provider = "makepad-gguf-cuda", "reusing resident model");
                runtime.clone()
            } else {
                let _span = tracing::info_span!("llm_load_runtime", provider = "makepad-gguf-cuda")
                    .entered();
                let loaded = Arc::new(
                    GgufRuntime::load(load_options, cancellation.clone().unwrap_or_default())
                        .map_err(map_generation_error)?,
                );
                let audit = loaded.audit();
                tracing::info!(device = %audit.actual_device_name, model_bytes = audit.model_bytes,
                    load_ms = audit.load_ms, quantized = audit.quantized_weight_residency,
                    source_revision = %audit.source_revision, "resident GGUF model loaded");
                runtimes.insert(artifacts.root.clone(), loaded.clone());
                loaded
            }
        };
        let report = runtime
            .generate_with_events(&rendered, generation_options, &mut |token_id, text| {
                sink.send(GenerateEvent::Token {
                    token_id,
                    text: text.to_owned(),
                })
                .map_err(|error| eyre::eyre!(error.to_string()))
            })
            .map_err(map_generation_error)?;
        tracing::info!(prompt_tokens = report.prompt_token_count,
            generated_tokens = report.generated_token_ids.len(), prefill_ms = report.prefill_ms,
            decode_ms = report.decode_ms, stop_reason = %report.stop_reason, "GGUF generation completed");
        sink.send(GenerateEvent::Completed {
            generated_text: report.generated_text.clone(),
        })?;
        Ok(GenerateResult {
            rendered_prompt: rendered,
            generated_token_ids: report.generated_token_ids,
            generated_text: report.generated_text,
        })
    }

    fn tokenize_prompt(&self, path: &Path, prompt: &str) -> Result<Vec<u32>, ServiceError> {
        let tokenizer = self
            .cached_tokenizers
            .get(path)
            .map_err(map_invalid_request_error)?;
        let encoded = tokenizer.encode(prompt, true).map_err(|error| {
            ServiceError::new(
                ServiceErrorKind::InvalidRequest,
                format!("Failed to tokenize rendered prompt: {error}"),
            )
        })?;
        Ok(encoded.get_ids().to_vec())
    }

    fn cached_burn_runtime(
        &self,
        artifacts: &LlmModelArtifacts,
    ) -> Result<Arc<CachedBurnTextCudaRuntime>, ServiceError> {
        self.cached_gguf_runtimes
            .lock()
            .map_err(|_| {
                ServiceError::new(
                    ServiceErrorKind::Internal,
                    "GGUF runtime cache mutex poisoned",
                )
            })?
            .clear();
        self.cached_burn_runtimes
            .lock()
            .map_err(|_| {
                ServiceError::new(
                    ServiceErrorKind::Internal,
                    "Burn runtime cache mutex poisoned",
                )
            })?
            .retain(|root, _| root == &artifacts.root);
        let cached = self
            .cached_burn_runtimes
            .lock()
            .map_err(|_| {
                ServiceError::new(
                    ServiceErrorKind::Internal,
                    "Burn runtime cache mutex poisoned",
                )
            })?
            .get(&artifacts.root)
            .cloned();
        if let Some(cached) = cached {
            tracing::debug!(model_dir = %artifacts.root.display(), "reusing cached Burn CUDA runtime");
            return Ok(cached);
        }

        let _span = tracing::info_span!("llm_load_runtime").entered();
        tracing::info!(model_dir = %artifacts.root.display(), "loading Burn CUDA runtime into service cache");
        let loaded =
            Arc::new(load_cached_burn_text_cuda_runtime(artifacts).map_err(map_generation_error)?);
        let mut runtimes = self.cached_burn_runtimes.lock().map_err(|_| {
            ServiceError::new(
                ServiceErrorKind::Internal,
                "Burn runtime cache mutex poisoned",
            )
        })?;
        Ok(runtimes
            .entry(artifacts.root.clone())
            .or_insert_with(|| loaded.clone())
            .clone())
    }

    fn inspect_artifacts(&self, model_dir: &Path) -> Result<LlmModelArtifacts, ServiceError> {
        let cache_key = model_dir.to_path_buf();
        if let Some(artifacts) = self
            .inspected_models
            .lock()
            .map_err(|_| {
                ServiceError::new(ServiceErrorKind::Internal, "model cache mutex poisoned")
            })?
            .get(&cache_key)
            .cloned()
        {
            return Ok(artifacts);
        }
        let artifacts = inspect_model_dir(model_dir).map_err(|error| {
            ServiceError::new(ServiceErrorKind::ModelNotPrepared, error.to_string())
        })?;
        self.inspected_models
            .lock()
            .map_err(|_| {
                ServiceError::new(ServiceErrorKind::Internal, "model cache mutex poisoned")
            })?
            .insert(cache_key, artifacts.clone());
        Ok(artifacts)
    }

    fn queue_for_model(&self, model_dir: &Path) -> Result<Arc<Mutex<()>>, ServiceError> {
        let mut queues = self.generation_queues.lock().map_err(|_| {
            ServiceError::new(ServiceErrorKind::Internal, "queue map mutex poisoned")
        })?;
        Ok(queues
            .entry(model_dir.to_path_buf())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone())
    }
}

fn check_cancelled(cancellation_token: Option<&CancellationToken>) -> Result<(), ServiceError> {
    let Some(cancellation_token) = cancellation_token else {
        return Ok(());
    };
    cancellation_token
        .bail_if_cancelled()
        .map_err(|error| ServiceError::new(ServiceErrorKind::Cancelled, error.to_string()))
}

struct ServiceBackendSink<'a, S: GenerateEventSink> {
    sink: &'a mut S,
}

impl<S: GenerateEventSink> BackendGenerationEventSink for ServiceBackendSink<'_, S> {
    fn generated_token(&mut self, token_id: usize, text: &str) -> eyre::Result<()> {
        self.sink
            .send(GenerateEvent::Token {
                token_id,
                text: text.to_owned(),
            })
            .map_err(|error| eyre::eyre!(error.to_string()))
    }
}

fn map_model_status(
    status: teamy_llm_burn_qwen35::model::LlmModelPreparationStatus,
) -> ModelStatus {
    ModelStatus {
        model_name: status.model_name,
        state: match status.state {
            LlmModelPreparationState::Missing => ModelPreparationState::Missing,
            LlmModelPreparationState::DownloadedUnprocessed => {
                ModelPreparationState::DownloadedUnprocessed
            }
            LlmModelPreparationState::Compatible => ModelPreparationState::Compatible,
        },
        locations: status
            .locations
            .into_iter()
            .map(|location| ModelLocationStatus {
                label: location.label,
                path: location.path.display().to_string(),
                exists: location.exists,
                compatible: location.compatible,
            })
            .collect(),
    }
}

fn map_invalid_request_error(error: eyre::Report) -> ServiceError {
    ServiceError::new(ServiceErrorKind::InvalidRequest, error.to_string())
}

fn map_internal_error(error: eyre::Report) -> ServiceError {
    ServiceError::new(ServiceErrorKind::Internal, error.to_string())
}

fn map_generation_error(error: eyre::Report) -> ServiceError {
    if is_generation_timeout_error(&error)
        || teamy_llm_makepad_gguf::is_generation_timeout_error(&error)
    {
        ServiceError::new(ServiceErrorKind::Timeout, error.to_string())
    } else if is_generation_cancelled_error(&error)
        || teamy_llm_makepad_gguf::is_generation_cancelled_error(&error)
        || error.to_string().contains("Cancelled")
        || error.to_string().contains("cancelled")
    {
        ServiceError::new(ServiceErrorKind::Cancelled, error.to_string())
    } else {
        ServiceError::new(ServiceErrorKind::BackendUnavailable, error.to_string())
    }
}

pub fn default_prepare_request(model: Option<String>) -> PrepareModelRequest {
    PrepareModelRequest {
        model: model.unwrap_or_else(|| DEFAULT_LLM_MODEL_NAME.to_owned()),
        overwrite: false,
        include_mmproj: false,
        export_burn_text: false,
        burn_text_dtype: None,
        download_main_model_file: true,
    }
}

#[cfg(test)]
mod queue_tests {
    use super::*;
    #[test]
    fn invalid_generation_bounds_fail_before_model_or_gpu_access() {
        let temp = tempfile::tempdir().unwrap();
        let service = TeamyLlmService::new(
            AppHome(temp.path().join("app")),
            CacheHome(temp.path().join("cache")),
        );
        for (max_new_tokens, timeout_ms) in [(0, None), (1, Some(0))] {
            let request = GenerateRequest {
                selector: ModelSelector::default(),
                system_prompt: None,
                user_prompt: "Synthetic prompt".into(),
                thinking_mode: ThinkingMode::Direct,
                max_new_tokens,
                timeout_ms,
                context_tokens: None,
            };
            let error = service
                .generate(request, &mut |_| {
                    panic!("Invalid request reached model events")
                })
                .unwrap_err();
            assert_eq!(error.kind, ServiceErrorKind::InvalidRequest);
        }
        assert!(!temp.path().join("app").exists());
        assert!(!temp.path().join("cache").exists());
    }
    #[test]
    fn waiting_request_cancels_without_waiting_for_another_generation() {
        let temp = tempfile::tempdir().unwrap();
        let service = TeamyLlmService::new(
            AppHome(temp.path().join("app")),
            CacheHome(temp.path().join("cache")),
        );
        let guard = service.lock_gpu_queue(None).unwrap();
        let cancellation = CancellationToken::new();
        let queued_service = service.clone();
        let queued_token = cancellation.clone();
        let (sender, receiver) = std::sync::mpsc::channel();
        let waiting = std::thread::spawn(move || {
            let result = queued_service.lock_gpu_queue(Some(&queued_token)).map(drop);
            sender.send(result).unwrap();
        });
        std::thread::sleep(Duration::from_millis(40));
        cancellation.request_cancel("queue test");
        let outcome = receiver.recv_timeout(Duration::from_secs(1));
        // Always release before asserting, including the failure case, to avoid hanging this test.
        drop(guard);
        waiting.join().unwrap();
        assert_eq!(
            outcome.unwrap().unwrap_err().kind,
            ServiceErrorKind::Cancelled
        );
    }
}

#[cfg(feature = "vox")]
pub mod vox_api;
