use arbitrary::Arbitrary;
use facet::Facet;

#[derive(Clone, Debug, Default, PartialEq, Eq, Facet, Arbitrary)]
#[facet(rename_all = "kebab-case")]
pub struct ModelSelector {
    pub model: Option<String>,
    pub model_dir: Option<String>,
}

impl ModelSelector {
    #[must_use]
    pub fn managed(model: impl Into<String>) -> Self {
        Self {
            model: Some(model.into()),
            model_dir: None,
        }
    }

    #[must_use]
    pub fn explicit_dir(model_dir: impl Into<String>) -> Self {
        Self {
            model: None,
            model_dir: Some(model_dir.into()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Facet, Arbitrary)]
#[facet(rename_all = "kebab-case")]
pub struct ServiceHealth {
    pub service_name: String,
    pub version: String,
    pub backend: String,
    pub ready: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Facet, Arbitrary)]
#[facet(rename_all = "kebab-case")]
pub struct ModelStatus {
    pub model_name: String,
    pub state: ModelPreparationState,
    pub locations: Vec<ModelLocationStatus>,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Facet, Arbitrary)]
#[facet(rename_all = "kebab-case")]
pub enum ModelPreparationState {
    Missing,
    DownloadedUnprocessed,
    Compatible,
}

#[derive(Clone, Debug, PartialEq, Eq, Facet, Arbitrary)]
#[facet(rename_all = "kebab-case")]
pub struct ModelLocationStatus {
    pub label: String,
    pub path: String,
    pub exists: bool,
    pub compatible: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Facet, Arbitrary)]
#[facet(rename_all = "kebab-case")]
pub struct ModelReport {
    pub model_root: String,
    pub display_name: String,
    pub family: String,
    pub architecture: String,
    pub quantization: String,
    pub supports_tool_calling: bool,
    pub supports_vision: bool,
    pub burn_support_state: String,
    pub details: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Facet, Arbitrary)]
#[facet(rename_all = "kebab-case")]
pub struct PrepareModelRequest {
    pub model: String,
    pub overwrite: bool,
    pub include_mmproj: bool,
    pub export_burn_text: bool,
    pub burn_text_dtype: Option<String>,
    pub download_main_model_file: bool,
}

#[repr(u8)]
#[derive(Clone, Debug, PartialEq, Eq, Facet, Arbitrary)]
#[facet(rename_all = "kebab-case")]
pub enum PrepareEvent {
    Started { model: String },
    ModelPrepared { model_dir: String },
    BurnTextExportStarted,
    BurnTextExportCompleted { message: String },
    Completed { model_dir: String },
    Failed { message: String },
}

#[derive(Clone, Debug, PartialEq, Eq, Facet, Arbitrary)]
#[facet(rename_all = "kebab-case")]
pub struct PrepareModelResult {
    pub model_dir: String,
    pub registered_model_dirs: Vec<String>,
    pub registration_warning: Option<String>,
}

/// Controls whether a Qwen-style prompt reserves output tokens for an explicit reasoning pass.
///
/// Direct mode is the default because it gives ordinary short answers the lowest latency. Thinking
/// mode remains available for tasks where a visible reasoning pass is worth the extra tokens.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Facet, Arbitrary)]
#[facet(rename_all = "kebab-case")]
pub enum ThinkingMode {
    #[default]
    Direct,
    Thinking,
}

#[derive(Clone, Debug, PartialEq, Eq, Facet, Arbitrary)]
#[facet(rename_all = "kebab-case")]
pub struct GenerateRequest {
    pub selector: ModelSelector,
    pub system_prompt: Option<String>,
    pub user_prompt: String,
    #[facet(default)]
    pub thinking_mode: ThinkingMode,
    pub max_new_tokens: usize,
    pub timeout_ms: Option<u64>,
}

#[repr(u8)]
#[derive(Clone, Debug, PartialEq, Eq, Facet, Arbitrary)]
#[facet(rename_all = "kebab-case")]
pub enum GenerateEvent {
    Started { model_dir: String },
    RenderedPrompt { prompt: String },
    Token { token_id: usize, text: String },
    Completed { generated_text: String },
    Failed { message: String },
}

#[derive(Clone, Debug, PartialEq, Eq, Facet, Arbitrary)]
#[facet(rename_all = "kebab-case")]
pub struct GenerateResult {
    pub rendered_prompt: String,
    pub generated_token_ids: Vec<usize>,
    pub generated_text: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Facet, Arbitrary)]
#[facet(rename_all = "kebab-case")]
pub struct ServiceError {
    pub kind: ServiceErrorKind,
    pub message: String,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Facet, Arbitrary)]
#[facet(rename_all = "kebab-case")]
pub enum ServiceErrorKind {
    InvalidRequest,
    ModelNotPrepared,
    BackendUnavailable,
    Timeout,
    Cancelled,
    Internal,
}

impl ServiceError {
    #[must_use]
    pub fn new(kind: ServiceErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for ServiceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for ServiceError {}
