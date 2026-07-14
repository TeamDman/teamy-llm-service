use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use teamy_cancellation::CancellationToken;
use teamy_llm_burn_qwen35::burn_backend::{BurnLlmSupportState, inspect_burn_runtime_support};
use teamy_llm_burn_qwen35::burn_text::{
    BurnTextGenerationOptions, GenerationEventSink as BackendGenerationEventSink,
    generate_with_burn_text_runtime_to_sink, is_generation_cancelled_error,
    is_generation_timeout_error, verify_required_cuda_runtime,
};
use teamy_llm_burn_qwen35::model::{
    DEFAULT_LLM_MODEL_NAME, LlmModelArtifacts, LlmModelPreparationState, inspect_model_dir,
    list_known_llm_model_statuses, prepare_burn_text_runtime_bundle, prepare_known_llm_model,
    render_model_report, resolve_llm_model_dir,
};
use teamy_llm_burn_qwen35::paths::{AppHome, CacheHome};
use teamy_llm_burn_qwen35::runtime::{render_qwen_single_turn_prompt, tokenize_rendered_prompt};
use teamy_llm_core::{
    GenerateEvent, GenerateRequest, GenerateResult, ModelLocationStatus, ModelPreparationState,
    ModelReport, ModelSelector, ModelStatus, PrepareEvent, PrepareModelRequest, PrepareModelResult,
    ServiceError, ServiceErrorKind, ServiceHealth,
};

pub trait GenerateEventSink {
    fn send(&mut self, event: GenerateEvent) -> Result<(), ServiceError>;
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
    generation_queues: Arc<Mutex<HashMap<PathBuf, Arc<Mutex<()>>>>>,
}

impl TeamyLlmService {
    pub fn new(app_home: AppHome, cache_home: CacheHome) -> Self {
        Self {
            app_home,
            cache_home,
            inspected_models: Arc::new(Mutex::new(HashMap::new())),
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
            backend: "burn-qwen35".to_owned(),
            ready: true,
        }
    }

    pub fn list_models(&self) -> Result<Vec<ModelStatus>, ServiceError> {
        let statuses = list_known_llm_model_statuses(&self.app_home, &self.cache_home)
            .map_err(map_internal_error)?;
        Ok(statuses.into_iter().map(map_model_status).collect())
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
        self.inspected_models
            .lock()
            .map_err(|_| {
                ServiceError::new(ServiceErrorKind::Internal, "model cache mutex poisoned")
            })?
            .insert(prepared.artifacts.root.clone(), prepared.artifacts.clone());
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

    pub fn generate_with_cancellation(
        &self,
        request: GenerateRequest,
        cancellation_token: Option<CancellationToken>,
        sink: &mut impl GenerateEventSink,
    ) -> Result<GenerateResult, ServiceError> {
        check_cancelled(cancellation_token.as_ref())?;
        verify_required_cuda_runtime().map_err(map_generation_error)?;
        check_cancelled(cancellation_token.as_ref())?;
        let model_dir = self.resolve_model_dir(&request.selector)?;
        check_cancelled(cancellation_token.as_ref())?;
        let queue = self.queue_for_model(&model_dir)?;
        let _queue_guard = queue.lock().map_err(|_| {
            ServiceError::new(
                ServiceErrorKind::Internal,
                "model generation queue mutex poisoned",
            )
        })?;

        check_cancelled(cancellation_token.as_ref())?;
        let artifacts = self.inspect_artifacts(&model_dir)?;
        check_cancelled(cancellation_token.as_ref())?;
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
        let rendered =
            render_qwen_single_turn_prompt(request.system_prompt.as_deref(), &request.user_prompt);
        sink.send(GenerateEvent::RenderedPrompt {
            prompt: rendered.prompt.clone(),
        })?;
        let token_ids = tokenize_rendered_prompt(&artifacts.tokenizer_path, &rendered.prompt)
            .map_err(map_invalid_request_error)?;
        check_cancelled(cancellation_token.as_ref())?;
        let options = BurnTextGenerationOptions {
            max_new_tokens: request.max_new_tokens,
            generation_timeout: request.timeout_ms.map(Duration::from_millis),
            cancellation_token,
        };
        let mut backend_sink = ServiceBackendSink { sink };
        let report = generate_with_burn_text_runtime_to_sink(
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

    fn resolve_model_dir(&self, selector: &ModelSelector) -> Result<PathBuf, ServiceError> {
        resolve_llm_model_dir(
            &self.app_home,
            &self.cache_home,
            selector.model.as_deref(),
            selector.model_dir.as_deref().map(Path::new),
        )
        .map_err(map_internal_error)
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
    if is_generation_timeout_error(&error) {
        ServiceError::new(ServiceErrorKind::Timeout, error.to_string())
    } else if is_generation_cancelled_error(&error)
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

#[cfg(feature = "vox")]
pub mod vox_api {
    use super::*;

    #[vox::service]
    pub trait TeamyLlm {
        async fn health(&self) -> ServiceHealth;
        async fn list_models(&self) -> Result<Vec<ModelStatus>, ServiceError>;
        async fn inspect_model(&self, selector: ModelSelector)
        -> Result<ModelReport, ServiceError>;
        async fn prepare_model(
            &self,
            request: PrepareModelRequest,
            events: vox::Tx<PrepareEvent>,
        ) -> Result<PrepareModelResult, ServiceError>;
        async fn generate(
            &self,
            request: GenerateRequest,
            events: vox::Tx<GenerateEvent>,
        ) -> Result<GenerateResult, ServiceError>;
    }

    impl TeamyLlm for TeamyLlmService {
        async fn health(&self) -> ServiceHealth {
            TeamyLlmService::health(self)
        }

        async fn list_models(&self) -> Result<Vec<ModelStatus>, ServiceError> {
            TeamyLlmService::list_models(self)
        }

        async fn inspect_model(
            &self,
            selector: ModelSelector,
        ) -> Result<ModelReport, ServiceError> {
            TeamyLlmService::inspect_model(self, selector)
        }

        async fn prepare_model(
            &self,
            request: PrepareModelRequest,
            events: vox::Tx<PrepareEvent>,
        ) -> Result<PrepareModelResult, ServiceError> {
            TeamyLlmService::prepare_model(self, request, |event| {
                events.try_send(event).map_err(|error| {
                    ServiceError::new(ServiceErrorKind::Cancelled, error.to_string())
                })
            })
            .await
        }

        async fn generate(
            &self,
            request: GenerateRequest,
            events: vox::Tx<GenerateEvent>,
        ) -> Result<GenerateResult, ServiceError> {
            TeamyLlmService::generate(self, request, &mut |event| {
                events.try_send(event).map_err(|error| {
                    ServiceError::new(ServiceErrorKind::Cancelled, error.to_string())
                })
            })
        }
    }

    pub async fn serve_local(
        address: &str,
        service: TeamyLlmService,
    ) -> Result<(), vox::ServeError> {
        vox::serve(address, TeamyLlmDispatcher::new(service)).await
    }
}
