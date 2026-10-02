use eyre::{Context, bail};
use facet::Facet;
use facet_json::RawJson;
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;

use crate::paths::{AppHome, CacheHome};

use crate::burn_backend::{inspect_burn_runtime_support, render_burn_runtime_support_report};
use crate::burn_text::{
    DEFAULT_BURN_TEXT_EXPORT_DTYPE, burn_text_manifest_path, export_burn_text_weights,
    inspect_burn_text_runtime_status, load_burn_text_manifest,
};
use crate::source_config::load_llm_source_config_summary;

pub const MODEL_DIRS_FILE_NAME: &str = "llm-model-dirs.txt";
pub const MANAGED_MODELS_DIR_NAME: &str = "models";
pub const LLM_MODELS_DIR_NAME: &str = "llm";
pub const MANAGED_MODEL_DOWNLOADS_DIR_NAME: &str = "downloads";
pub const MODEL_FILE_NAME: &str = "model.gguf";
pub const MMPROJ_FILE_NAME: &str = "mmproj.gguf";
pub const TOKENIZER_FILE_NAME: &str = "tokenizer.json";
pub const TOKENIZER_CONFIG_FILE_NAME: &str = "tokenizer_config.json";
pub const HF_CONFIG_FILE_NAME: &str = "config.json";
pub const MODEL_METADATA_FILE_NAME: &str = "model-metadata.json";
pub const DEFAULT_LLM_MODEL_NAME: &str = "qwopus-3.5-9b-coder-q4-k-m";
pub const ORCA_QWEN_MODEL_NAME: &str = "qwen3.8-27b-uncensored-q4-k-m";
pub const ORCA_QWEN_GGUF_FILE: &str = "Qwen3.8-27B-Uncensored-Q4_K_M.gguf";
pub const ORCA_QWEN_GGUF_REVISION: &str = "fc437a3374c9977bdc339a8ec106dcd9a0357001";
pub const ORCA_QWEN_SOURCE_REVISION: &str = "8cb32d72080f6a47bf34dc5adf8067daa6c63a31";
const BURN_TEXT_ONLY_MODEL_PLACEHOLDER: &str = "This placeholder marks a Teamy Burn-text-only model directory. The Rust runtime uses the converted burn-text bundle rather than this GGUF file.\n";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KnownLlmModel {
    pub name: &'static str,
    pub family: &'static str,
    pub display_name: &'static str,
    pub source_repo_id: &'static str,
    pub model_repo_id: &'static str,
    pub tokenizer_repo_id: &'static str,
    pub architecture: &'static str,
    pub quantization: &'static str,
    pub model_file_name: &'static str,
    pub parameter_count: &'static str,
    pub size_estimate: &'static str,
    pub supports_vision: bool,
    pub supports_tool_calling: bool,
}

pub const KNOWN_LLM_MODELS: [KnownLlmModel; 2] = [
    KnownLlmModel {
        name: DEFAULT_LLM_MODEL_NAME,
        family: "qwopus",
        display_name: "Jackrong/Qwopus3.5-9B-Coder-GGUF (Q4_K_M)",
        source_repo_id: "Jackrong/Qwopus3.5-9B-Coder",
        model_repo_id: "Jackrong/Qwopus3.5-9B-Coder-GGUF",
        tokenizer_repo_id: "Jackrong/Qwopus3.5-9B-Coder",
        architecture: "qwen35",
        quantization: "Q4_K_M",
        model_file_name: "Qwopus3.5-9B-coder-Exp-Q4_K_M.gguf",
        parameter_count: "9B",
        size_estimate: "5.63 GiB",
        supports_vision: true,
        supports_tool_calling: true,
    },
    KnownLlmModel {
        name: ORCA_QWEN_MODEL_NAME,
        family: "qwen3.8",
        display_name: "OrcaRouter/Qwen3.8-27B-Uncensored-GGUF (Q4_K_M)",
        source_repo_id: "orcarouter/Qwen3.8-27B-Uncensored",
        model_repo_id: "orcarouter/Qwen3.8-27B-Uncensored-GGUF",
        tokenizer_repo_id: "orcarouter/Qwen3.8-27B-Uncensored",
        architecture: "qwen35",
        quantization: "Q4_K_M",
        model_file_name: ORCA_QWEN_GGUF_FILE,
        parameter_count: "27B",
        size_estimate: "15.66 GiB (16,810,714,496 bytes)",
        // This runtime slice accepts text only; the upstream model's vision tower is separate.
        supports_vision: false,
        supports_tool_calling: false,
    },
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LlmModelPreparationState {
    Missing,
    DownloadedUnprocessed,
    Compatible,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LlmModelLocationStatus {
    pub label: String,
    pub path: PathBuf,
    pub exists: bool,
    pub compatible: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LlmModelPreparationStatus {
    pub model_name: String,
    pub state: LlmModelPreparationState,
    pub locations: Vec<LlmModelLocationStatus>,
}

impl LlmModelPreparationStatus {
    #[must_use]
    pub const fn is_compatible(&self) -> bool {
        matches!(self.state, LlmModelPreparationState::Compatible)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Facet)]
pub struct LlmManagedModelMetadata {
    pub model_name: String,
    pub family: String,
    pub display_name: String,
    pub source_repo_id: String,
    pub model_repo_id: String,
    pub tokenizer_repo_id: String,
    pub architecture: String,
    pub quantization: String,
    pub model_file_name: String,
    pub mmproj_file_name: Option<String>,
    pub hf_config_file_name: String,
    pub tokenizer_file_name: String,
    pub tokenizer_config_file_name: String,
    pub parameter_count: String,
    pub size_estimate: String,
    pub supports_vision: bool,
    pub supports_tool_calling: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LlmModelArtifacts {
    pub root: PathBuf,
    pub model_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub tokenizer_config_path: PathBuf,
    pub hf_config_path: PathBuf,
    pub mmproj_path: Option<PathBuf>,
    pub metadata_path: PathBuf,
    pub metadata: LlmManagedModelMetadata,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedLlmModel {
    pub managed_dir: PathBuf,
    pub artifacts: LlmModelArtifacts,
    pub registered_model_dirs: Vec<PathBuf>,
    pub registration_warning: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenizerConfigSummary {
    pub path: PathBuf,
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
    pub chat_template: Option<String>,
}

#[derive(Clone, Debug, Facet, PartialEq, Eq)]
struct TokenizerConfigFile {
    #[facet(default)]
    bos_token: Option<RawJson<'static>>,
    #[facet(default)]
    eos_token: Option<RawJson<'static>>,
    #[facet(default)]
    chat_template: Option<String>,
}

#[must_use]
pub fn known_llm_model(model_name: &str) -> Option<&'static KnownLlmModel> {
    KNOWN_LLM_MODELS
        .iter()
        .find(|known| known.name.eq_ignore_ascii_case(model_name.trim()))
}

#[must_use]
pub fn managed_models_dir(cache_home: &CacheHome) -> PathBuf {
    cache_home
        .join(MANAGED_MODELS_DIR_NAME)
        .join(LLM_MODELS_DIR_NAME)
}

#[must_use]
pub fn managed_model_downloads_dir(cache_home: &CacheHome) -> PathBuf {
    managed_models_dir(cache_home).join(MANAGED_MODEL_DOWNLOADS_DIR_NAME)
}

#[must_use]
pub fn managed_model_dir(cache_home: &CacheHome, model_name: &str) -> PathBuf {
    managed_models_dir(cache_home).join(model_name)
}

#[must_use]
pub fn model_file_url(known: &KnownLlmModel) -> String {
    if known.name == ORCA_QWEN_MODEL_NAME {
        return format!(
            "https://huggingface.co/{}/resolve/{ORCA_QWEN_GGUF_REVISION}/{}",
            known.model_repo_id, known.model_file_name
        );
    }
    hugging_face_resolve_url(known.model_repo_id, known.model_file_name)
}

#[must_use]
pub fn mmproj_file_url(known: &KnownLlmModel) -> String {
    hugging_face_resolve_url(known.model_repo_id, MMPROJ_FILE_NAME)
}

#[must_use]
pub fn tokenizer_file_url(known: &KnownLlmModel) -> String {
    if known.name == ORCA_QWEN_MODEL_NAME {
        return format!(
            "https://huggingface.co/{}/resolve/{ORCA_QWEN_SOURCE_REVISION}/{TOKENIZER_FILE_NAME}",
            known.tokenizer_repo_id
        );
    }
    hugging_face_resolve_url(known.tokenizer_repo_id, TOKENIZER_FILE_NAME)
}

#[must_use]
pub fn tokenizer_config_file_url(known: &KnownLlmModel) -> String {
    if known.name == ORCA_QWEN_MODEL_NAME {
        return format!(
            "https://huggingface.co/{}/resolve/{ORCA_QWEN_SOURCE_REVISION}/{TOKENIZER_CONFIG_FILE_NAME}",
            known.tokenizer_repo_id
        );
    }
    hugging_face_resolve_url(known.tokenizer_repo_id, TOKENIZER_CONFIG_FILE_NAME)
}

#[must_use]
pub fn hf_config_file_url(known: &KnownLlmModel) -> String {
    if known.name == ORCA_QWEN_MODEL_NAME {
        return format!(
            "https://huggingface.co/{}/resolve/{ORCA_QWEN_SOURCE_REVISION}/{HF_CONFIG_FILE_NAME}",
            known.source_repo_id
        );
    }
    hugging_face_resolve_url(known.source_repo_id, HF_CONFIG_FILE_NAME)
}

#[must_use]
pub fn hugging_face_resolve_url(repo_id: &str, file_name: &str) -> String {
    format!("https://huggingface.co/{repo_id}/resolve/main/{file_name}")
}

/// Discover a locally prepared Teamy Studio LLM model directory.
///
/// # Errors
///
/// This function will return an error if the directory is incomplete or incompatible.
pub fn inspect_model_dir(root: &Path) -> eyre::Result<LlmModelArtifacts> {
    inspect_model_dir_with_tokenizer_validation(root, |tokenizer_path| {
        tokenizers::Tokenizer::from_file(tokenizer_path)
            .map(|_| ())
            .map_err(|error| {
                eyre::eyre!(
                    "Failed to load tokenizer from {}: {}",
                    tokenizer_path.display(),
                    error
                )
            })
    })
}

/// Inspect artifacts while letting a resident service retain its validated tokenizer.
///
/// The callback must load or validate the tokenizer at the supplied path and reject invalid
/// contents. It is called exactly once after [`inspect_model_files`] succeeds. This is a
/// validated inspection API; callers that defer tokenizer parsing must use the explicitly
/// structural [`inspect_model_files`] API instead. Tokenization options are unchanged.
///
/// # Errors
///
/// Returns the callback's validation error or an incomplete/incompatible artifact error.
pub fn inspect_model_dir_with_tokenizer_validation(
    root: &Path,
    validate_tokenizer: impl FnOnce(&Path) -> eyre::Result<()>,
) -> eyre::Result<LlmModelArtifacts> {
    let artifacts = inspect_model_files(root)?;
    validate_tokenizer(&artifacts.tokenizer_path)?;
    Ok(artifacts)
}

/// Inspect the model directory's files and declared metadata without parsing the tokenizer.
///
/// This structural inspection performs the same artifact presence and metadata/path checks
/// as [`inspect_model_dir`]. It does not validate tokenizer contents, verify model weight
/// identity or construct a GPU runtime. A service may use it to overlap CPU preparation,
/// but must successfully validate the tokenizer before caching validated artifacts or
/// beginning generation. Ordinary callers should use [`inspect_model_dir`] or
/// [`inspect_model_dir_with_tokenizer_validation`].
///
/// # Errors
///
/// Returns an incomplete directory, malformed metadata or incompatible declared-path error.
pub fn inspect_model_files(root: &Path) -> eyre::Result<LlmModelArtifacts> {
    ensure_existing_dir(root)?;

    let tokenizer_path = root.join(TOKENIZER_FILE_NAME);
    if !tokenizer_path.is_file() {
        bail!(
            "LLM model directory is missing {}: {}",
            TOKENIZER_FILE_NAME,
            tokenizer_path.display()
        );
    }

    let tokenizer_config_path = root.join(TOKENIZER_CONFIG_FILE_NAME);
    if !tokenizer_config_path.is_file() {
        bail!(
            "LLM model directory is missing {}: {}",
            TOKENIZER_CONFIG_FILE_NAME,
            tokenizer_config_path.display()
        );
    }

    let hf_config_path = root.join(HF_CONFIG_FILE_NAME);
    if !hf_config_path.is_file() {
        bail!(
            "LLM model directory is missing {}: {}",
            HF_CONFIG_FILE_NAME,
            hf_config_path.display()
        );
    }

    let metadata_path = root.join(MODEL_METADATA_FILE_NAME);
    if !metadata_path.is_file() {
        bail!(
            "LLM model directory is missing {}: {}",
            MODEL_METADATA_FILE_NAME,
            metadata_path.display()
        );
    }
    let metadata: LlmManagedModelMetadata =
        facet_json::from_slice(&std::fs::read(&metadata_path).wrap_err_with(|| {
            format!(
                "Failed to read LLM metadata file {}",
                metadata_path.display()
            )
        })?)
        .wrap_err_with(|| format!("Failed to parse {}", metadata_path.display()))?;
    let model_path = declared_model_path(root, &metadata)?;
    if !model_path.is_file() {
        bail!(
            "LLM model directory is missing declared model file: {}",
            model_path.display()
        );
    }

    let mmproj_path = root.join(MMPROJ_FILE_NAME);
    let mmproj_path = if mmproj_path.is_file() {
        Some(mmproj_path)
    } else {
        None
    };

    Ok(LlmModelArtifacts {
        root: root.to_path_buf(),
        model_path,
        tokenizer_path,
        tokenizer_config_path,
        hf_config_path,
        mmproj_path,
        metadata_path,
        metadata,
    })
}

/// Keep old Burn artifact layouts unchanged; the new GGUF adapter consumes the exact publisher
/// filename directly. Metadata cannot select arbitrary relative paths or traverse the model root.
fn declared_model_path(root: &Path, metadata: &LlmManagedModelMetadata) -> eyre::Result<PathBuf> {
    if metadata.model_name == ORCA_QWEN_MODEL_NAME {
        if metadata.model_file_name != ORCA_QWEN_GGUF_FILE {
            bail!("Frozen Orca Qwen variant requires its exact published GGUF filename");
        }
        return Ok(root.join(ORCA_QWEN_GGUF_FILE));
    }
    Ok(root.join(MODEL_FILE_NAME))
}

#[must_use]
pub fn render_model_report(artifacts: &LlmModelArtifacts) -> String {
    let mut lines = vec![
        format!("Model root: {}", artifacts.root.display()),
        format!("Display name: {}", artifacts.metadata.display_name),
        format!("Family: {}", artifacts.metadata.family),
        format!("Architecture: {}", artifacts.metadata.architecture),
        format!("Quantization: {}", artifacts.metadata.quantization),
        format!("Source repo: {}", artifacts.metadata.source_repo_id),
        format!("Model repo: {}", artifacts.metadata.model_repo_id),
        format!("Tokenizer repo: {}", artifacts.metadata.tokenizer_repo_id),
        format!("Model file: {}", artifacts.model_path.display()),
        format!("Tokenizer file: {}", artifacts.tokenizer_path.display()),
        format!(
            "Tokenizer config file: {}",
            artifacts.tokenizer_config_path.display()
        ),
        format!(
            "Hugging Face config file: {}",
            artifacts.hf_config_path.display()
        ),
        format!("Metadata file: {}", artifacts.metadata_path.display()),
        format!("Parameter count: {}", artifacts.metadata.parameter_count),
        format!("Size estimate: {}", artifacts.metadata.size_estimate),
        format!(
            "Supports tool calling: {}",
            artifacts.metadata.supports_tool_calling
        ),
        format!("Supports vision: {}", artifacts.metadata.supports_vision),
    ];
    if let Some(mmproj_path) = &artifacts.mmproj_path {
        lines.push(format!("Vision mmproj file: {}", mmproj_path.display()));
    } else {
        lines.push("Vision mmproj file: not installed".to_owned());
    }
    let burn_text_status = inspect_burn_text_runtime_status(&artifacts.root);
    lines.push(format!(
        "Burn text manifest: {}",
        if burn_text_status.exists {
            burn_text_status.manifest_path.clone()
        } else {
            "missing".to_owned()
        }
    ));
    if let Ok(summary) = load_tokenizer_config_summary(&artifacts.tokenizer_config_path) {
        lines.push(format!(
            "Tokenizer bos token: {}",
            summary.bos_token.unwrap_or_else(|| "<none>".to_owned())
        ));
        lines.push(format!(
            "Tokenizer eos token: {}",
            summary.eos_token.unwrap_or_else(|| "<none>".to_owned())
        ));
        lines.push(format!(
            "Tokenizer chat template: {}",
            if summary
                .chat_template
                .as_deref()
                .is_some_and(|value| !value.trim().is_empty())
            {
                "present"
            } else {
                "missing"
            }
        ));
    }
    if let Ok(summary) = load_llm_source_config_summary(&artifacts.hf_config_path) {
        if let Some(model_type) = summary.model_type {
            lines.push(format!("HF model type: {}", model_type));
        }
        if let Some(text_model_type) = summary.text_model_type {
            lines.push(format!("HF text model type: {}", text_model_type));
        }
        if let Some(model_name) = summary.model_name {
            lines.push(format!("HF model name: {}", model_name));
        }
        if !summary.architectures.is_empty() {
            lines.push(format!(
                "HF architectures: {}",
                summary.architectures.join(", ")
            ));
        }
        if let Some(num_hidden_layers) = summary.text_num_hidden_layers {
            lines.push(format!("HF text layer count: {}", num_hidden_layers));
        }
        if let Some(hidden_size) = summary.text_hidden_size {
            lines.push(format!("HF text hidden size: {}", hidden_size));
        }
        if let Some(intermediate_size) = summary.text_intermediate_size {
            lines.push(format!("HF text intermediate size: {}", intermediate_size));
        }
        if let Some(num_attention_heads) = summary.text_num_attention_heads {
            lines.push(format!("HF text attention heads: {}", num_attention_heads));
        }
        if let Some(num_key_value_heads) = summary.text_num_key_value_heads {
            lines.push(format!("HF text kv heads: {}", num_key_value_heads));
        }
        if let Some(head_dim) = summary.text_head_dim {
            lines.push(format!("HF text head dim: {}", head_dim));
        }
        if let Some(hidden_act) = summary.text_hidden_act {
            lines.push(format!("HF text hidden act: {}", hidden_act));
        }
        if let Some(partial_rotary_factor) = summary.text_partial_rotary_factor {
            lines.push(format!(
                "HF text partial rotary factor: {}",
                partial_rotary_factor
            ));
        }
        if let Some(rope_theta) = summary.text_rope_theta {
            lines.push(format!("HF text rope theta: {}", rope_theta));
        }
        if let Some(full_attention_interval) = summary.text_full_attention_interval {
            lines.push(format!(
                "HF full attention interval: {}",
                full_attention_interval
            ));
        }
        if let Some(linear_num_key_heads) = summary.text_linear_num_key_heads {
            lines.push(format!(
                "HF linear-attn key heads: {}",
                linear_num_key_heads
            ));
        }
        if let Some(linear_num_value_heads) = summary.text_linear_num_value_heads {
            lines.push(format!(
                "HF linear-attn value heads: {}",
                linear_num_value_heads
            ));
        }
        if let Some(linear_key_head_dim) = summary.text_linear_key_head_dim {
            lines.push(format!(
                "HF linear-attn key head dim: {}",
                linear_key_head_dim
            ));
        }
        if let Some(linear_value_head_dim) = summary.text_linear_value_head_dim {
            lines.push(format!(
                "HF linear-attn value head dim: {}",
                linear_value_head_dim
            ));
        }
        if let Some(linear_conv_kernel_dim) = summary.text_linear_conv_kernel_dim {
            lines.push(format!(
                "HF linear-attn conv kernel: {}",
                linear_conv_kernel_dim
            ));
        }
        if !summary.text_layer_type_counts.is_empty() {
            let histogram = summary
                .text_layer_type_counts
                .iter()
                .map(|(layer_type, count)| format!("{layer_type}={count}"))
                .collect::<Vec<_>>()
                .join(", ");
            lines.push(format!("HF text layer histogram: {}", histogram));
        }
        if !summary.text_layer_types_preview.is_empty() {
            lines.push(format!(
                "HF text layer preview: {}",
                summary.text_layer_types_preview.join(", ")
            ));
        }
    }
    if let Ok(report) = inspect_burn_runtime_support(artifacts) {
        lines.push(render_burn_runtime_support_report(&report));
    }
    lines.join("\n")
}

/// Load a tokenizer config summary from a Hugging Face tokenizer config file.
///
/// # Errors
///
/// This function will return an error if the tokenizer config cannot be read or parsed.
pub fn load_tokenizer_config_summary(path: &Path) -> eyre::Result<TokenizerConfigSummary> {
    let parsed: TokenizerConfigFile = facet_json::from_slice(
        &std::fs::read(path)
            .wrap_err_with(|| format!("Failed to read tokenizer config {}", path.display()))?,
    )
    .wrap_err_with(|| format!("Failed to parse tokenizer config {}", path.display()))?;
    Ok(TokenizerConfigSummary {
        path: path.to_path_buf(),
        bos_token: parsed
            .bos_token
            .map(parse_tokenizer_config_token_value)
            .transpose()?,
        eos_token: parsed
            .eos_token
            .map(parse_tokenizer_config_token_value)
            .transpose()?,
        chat_template: parsed.chat_template,
    })
}

#[derive(Clone, Debug, Facet, PartialEq, Eq)]
struct TokenizerConfigTokenObject {
    content: String,
}

fn parse_tokenizer_config_token_value(value: RawJson<'static>) -> eyre::Result<String> {
    if let Ok(token) = facet_json::from_str::<String>(value.as_ref()) {
        return Ok(token);
    }
    if let Ok(token) = facet_json::from_str::<TokenizerConfigTokenObject>(value.as_ref()) {
        return Ok(token.content);
    }
    bail!(
        "Tokenizer token value must be a JSON string or an object with `content`, found {}",
        value.as_ref()
    )
}

/// Prepare a known Qwopus/Qwen-style model artifact bundle into Teamy's managed cache.
///
/// # Errors
///
/// This function will return an error if the model cannot be downloaded, validated, or registered.
pub async fn prepare_known_llm_model(
    app_home: &AppHome,
    cache_home: &CacheHome,
    model_name: &str,
    overwrite: bool,
    include_mmproj: bool,
    download_main_model_file: bool,
) -> eyre::Result<PreparedLlmModel> {
    let known = known_llm_model(model_name).ok_or_else(|| {
        eyre::eyre!(
            "Unknown Teamy Studio LLM model {:?}. Known models: {}",
            model_name,
            KNOWN_LLM_MODELS
                .iter()
                .map(|known| known.name)
                .collect::<Vec<_>>()
                .join(", ")
        )
    })?;

    let managed_dir = managed_model_dir(cache_home, known.name);
    if managed_dir.is_dir() && !overwrite {
        let artifacts = inspect_model_dir(&managed_dir).wrap_err_with(|| {
            format!(
                "Managed LLM model directory {} already exists but could not be inspected cleanly",
                managed_dir.display()
            )
        })?;
        let (registered_model_dirs, registration_warning) =
            try_add_registered_model_dir(app_home, &managed_dir);
        return Ok(PreparedLlmModel {
            managed_dir,
            artifacts,
            registered_model_dirs,
            registration_warning,
        });
    }

    if known.name == ORCA_QWEN_MODEL_NAME {
        bail!(
            "The pinned Orca Qwen GGUF must be acquired directly into its managed model directory with approved Hugging Face tooling, then inspected and registered. This first slice does not overwrite it or convert it into Burn FP16 weights."
        );
    }

    let staging_dir = managed_dir.with_extension("staging");
    remove_existing_path(&staging_dir)
        .wrap_err_with(|| format!("Failed to clear staging path {}", staging_dir.display()))?;
    std::fs::create_dir_all(&staging_dir).wrap_err_with(|| {
        format!(
            "Failed to create staging directory {}",
            staging_dir.display()
        )
    })?;

    if download_main_model_file {
        download_to_file(&model_file_url(known), &staging_dir.join(MODEL_FILE_NAME)).await?;
    } else {
        write_burn_text_only_model_placeholder(&staging_dir.join(MODEL_FILE_NAME))?;
    }
    download_to_file(
        &tokenizer_file_url(known),
        &staging_dir.join(TOKENIZER_FILE_NAME),
    )
    .await?;
    download_to_file(
        &tokenizer_config_file_url(known),
        &staging_dir.join(TOKENIZER_CONFIG_FILE_NAME),
    )
    .await?;
    download_to_file(
        &hf_config_file_url(known),
        &staging_dir.join(HF_CONFIG_FILE_NAME),
    )
    .await?;
    if include_mmproj {
        download_to_file(&mmproj_file_url(known), &staging_dir.join(MMPROJ_FILE_NAME)).await?;
    }
    write_model_metadata(&staging_dir, known, include_mmproj)?;

    let _artifacts = inspect_model_dir(&staging_dir)?;
    if managed_dir.exists() {
        remove_existing_path(&managed_dir).wrap_err_with(|| {
            format!(
                "Failed to replace existing managed LLM model path {}",
                managed_dir.display()
            )
        })?;
    } else if let Some(parent) = managed_dir.parent() {
        std::fs::create_dir_all(parent).wrap_err_with(|| {
            format!(
                "Failed to create LLM model parent directory {}",
                parent.display()
            )
        })?;
    }
    std::fs::rename(&staging_dir, &managed_dir).wrap_err_with(|| {
        format!(
            "Failed to move staged LLM model directory {} into {}",
            staging_dir.display(),
            managed_dir.display()
        )
    })?;

    let artifacts = inspect_model_dir(&managed_dir)?;
    let (registered_model_dirs, registration_warning) =
        try_add_registered_model_dir(app_home, &managed_dir);
    Ok(PreparedLlmModel {
        managed_dir,
        artifacts,
        registered_model_dirs,
        registration_warning,
    })
}

/// Export a Burn text runtime bundle beside a prepared Teamy LLM model directory.
///
/// # Errors
///
/// This function will return an error if the Python export helper cannot prepare the Burn bundle.
pub fn prepare_burn_text_runtime_bundle(
    artifacts: &LlmModelArtifacts,
    overwrite: bool,
    dtype: Option<&str>,
) -> eyre::Result<String> {
    if artifacts.metadata.model_name == ORCA_QWEN_MODEL_NAME {
        bail!(
            "This Orca Qwen Q4 GGUF uses the separate Makepad quantized runtime; Burn FP16 conversion is not supported for this variant"
        );
    }
    let manifest_path = burn_text_manifest_path(&artifacts.root);
    if manifest_path.is_file() && !overwrite {
        load_burn_text_manifest(&manifest_path).wrap_err_with(|| {
            format!(
                "Existing Burn text manifest at {} was not readable; rerun with --overwrite to replace it",
                manifest_path.display()
            )
        })?;
        return Ok(format!(
            "Burn text runtime bundle already exists at {}; skipping export because --overwrite was not set",
            manifest_path.display()
        ));
    }
    let report = export_burn_text_weights(
        artifacts,
        overwrite,
        dtype.unwrap_or(DEFAULT_BURN_TEXT_EXPORT_DTYPE),
    )?;
    Ok(format!(
        "Burn text export complete: {} tensors written to {} as {}",
        report.tensor_count, report.output_dir, report.dtype
    ))
}

#[must_use]
pub fn render_registered_model_dirs(model_dirs: &[PathBuf]) -> String {
    if model_dirs.is_empty() {
        return "No registered LLM model directories.".to_owned();
    }
    model_dirs
        .iter()
        .enumerate()
        .map(|(index, path)| {
            let suffix = if index == 0 { " (default)" } else { "" };
            format!("{}{}", path.display(), suffix)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// List known Teamy LLM model statuses and registered directories.
///
/// # Errors
///
/// This function will return an error if the app-home registry cannot be read.
pub fn list_known_llm_model_statuses(
    app_home: &AppHome,
    cache_home: &CacheHome,
) -> eyre::Result<Vec<LlmModelPreparationStatus>> {
    let registered = list_registered_model_dirs(app_home)?;
    let statuses = KNOWN_LLM_MODELS
        .iter()
        .map(|known| {
            let managed_dir = managed_model_dir(cache_home, known.name);
            let managed_compatible = inspect_model_dir(&managed_dir)
                .is_ok_and(|artifacts| artifacts.metadata.model_name == known.name);
            let mut locations = vec![LlmModelLocationStatus {
                label: "managed".to_owned(),
                path: managed_dir.clone(),
                exists: managed_dir.exists(),
                compatible: managed_compatible,
            }];
            for path in &registered {
                let Ok(artifacts) = inspect_model_dir(path) else {
                    // An unreadable inventory has no verified identity to attribute to this row.
                    continue;
                };
                if artifacts.metadata.model_name != known.name {
                    continue;
                }
                locations.push(LlmModelLocationStatus {
                    label: "registered".to_owned(),
                    path: path.clone(),
                    exists: path.exists(),
                    compatible: true,
                });
            }
            let state = if managed_compatible {
                LlmModelPreparationState::Compatible
            } else if managed_dir.exists() {
                LlmModelPreparationState::DownloadedUnprocessed
            } else {
                LlmModelPreparationState::Missing
            };
            LlmModelPreparationStatus {
                model_name: known.name.to_owned(),
                state,
                locations,
            }
        })
        .collect();
    Ok(statuses)
}

/// Resolve a Teamy Studio LLM model directory from an explicit path, model name, or default registry.
///
/// # Errors
///
/// This function will return an error if the app-home registry cannot be read.
pub fn resolve_llm_model_dir(
    app_home: &AppHome,
    cache_home: &CacheHome,
    managed_model_name: Option<&str>,
    explicit_model_dir: Option<&Path>,
) -> eyre::Result<PathBuf> {
    if let Some(explicit_model_dir) = explicit_model_dir {
        return Ok(explicit_model_dir.to_path_buf());
    }
    if let Some(model_name) = managed_model_name.filter(|value| !value.trim().is_empty()) {
        return Ok(managed_model_dir(cache_home, model_name.trim()));
    }
    if let Some(default_model_dir) = resolve_default_model_dir(app_home)? {
        return Ok(default_model_dir);
    }
    Ok(managed_model_dir(cache_home, DEFAULT_LLM_MODEL_NAME))
}

/// Add a model directory to the Teamy LLM model registry.
///
/// # Errors
///
/// This function will return an error if the app-home registry cannot be updated.
pub fn add_registered_model_dir(app_home: &AppHome, root: &Path) -> eyre::Result<Vec<PathBuf>> {
    let root = root.to_path_buf();
    app_home.ensure_dir()?;
    let mut model_dirs = list_registered_model_dirs(app_home)?;
    model_dirs.retain(|path| path != &root);
    model_dirs.insert(0, root);
    write_registered_model_dirs(app_home, &model_dirs)?;
    Ok(model_dirs)
}

fn try_add_registered_model_dir(app_home: &AppHome, root: &Path) -> (Vec<PathBuf>, Option<String>) {
    match add_registered_model_dir(app_home, root) {
        Ok(model_dirs) => (model_dirs, None),
        Err(error) => (
            vec![root.to_path_buf()],
            Some(format!(
                "Prepared the model successfully, but Teamy could not persist the registered model directory list: {}. You can still use this model via `--model-dir {}`.",
                error,
                root.display()
            )),
        ),
    }
}

/// List all registered Teamy LLM model directories.
///
/// # Errors
///
/// This function will return an error if the app-home registry file cannot be read.
pub fn list_registered_model_dirs(app_home: &AppHome) -> eyre::Result<Vec<PathBuf>> {
    let registry_path = app_home.file_path(MODEL_DIRS_FILE_NAME);
    if !registry_path.is_file() {
        return Ok(Vec::new());
    }
    let registry = std::fs::read_to_string(&registry_path).wrap_err_with(|| {
        format!(
            "Failed to read LLM model registry {}",
            registry_path.display()
        )
    })?;
    let mut model_dirs = Vec::new();
    for line in registry.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        model_dirs.push(PathBuf::from(trimmed));
    }
    Ok(model_dirs)
}

/// Resolve the default Teamy LLM model directory from the registry.
///
/// # Errors
///
/// This function will return an error if the app-home registry file cannot be read.
pub fn resolve_default_model_dir(app_home: &AppHome) -> eyre::Result<Option<PathBuf>> {
    Ok(list_registered_model_dirs(app_home)?.into_iter().next())
}

fn write_registered_model_dirs(app_home: &AppHome, model_dirs: &[PathBuf]) -> eyre::Result<()> {
    app_home.ensure_dir()?;
    let registry_path = app_home.file_path(MODEL_DIRS_FILE_NAME);
    let mut output = String::new();
    for path in model_dirs {
        output.push_str(&path.display().to_string());
        output.push('\n');
    }
    std::fs::write(&registry_path, output)
        .wrap_err_with(|| format!("Failed to write {}", registry_path.display()))
}

pub fn write_model_metadata(
    root: &Path,
    known: &KnownLlmModel,
    include_mmproj: bool,
) -> eyre::Result<()> {
    let metadata = LlmManagedModelMetadata {
        model_name: known.name.to_owned(),
        family: known.family.to_owned(),
        display_name: known.display_name.to_owned(),
        source_repo_id: known.source_repo_id.to_owned(),
        model_repo_id: known.model_repo_id.to_owned(),
        tokenizer_repo_id: known.tokenizer_repo_id.to_owned(),
        architecture: known.architecture.to_owned(),
        quantization: known.quantization.to_owned(),
        model_file_name: known.model_file_name.to_owned(),
        mmproj_file_name: include_mmproj.then(|| MMPROJ_FILE_NAME.to_owned()),
        hf_config_file_name: HF_CONFIG_FILE_NAME.to_owned(),
        tokenizer_file_name: TOKENIZER_FILE_NAME.to_owned(),
        tokenizer_config_file_name: TOKENIZER_CONFIG_FILE_NAME.to_owned(),
        parameter_count: known.parameter_count.to_owned(),
        size_estimate: known.size_estimate.to_owned(),
        supports_vision: known.supports_vision,
        supports_tool_calling: known.supports_tool_calling,
    };
    let metadata_path = root.join(MODEL_METADATA_FILE_NAME);
    std::fs::write(
        &metadata_path,
        facet_json::to_vec_pretty(&metadata).context("Failed to encode LLM model metadata")?,
    )
    .wrap_err_with(|| format!("Failed to write {}", metadata_path.display()))
}

fn ensure_existing_dir(root: &Path) -> eyre::Result<()> {
    if !root.exists() {
        bail!(
            "LLM model directory does not exist: {}. Prepare the model first with `teamy-studio llm model prepare --burn-text-only --with-burn-text`, or pass `--model-dir <prepared-model-dir>`.",
            root.display()
        );
    }
    if !root.is_dir() {
        bail!(
            "Expected LLM model directory but found a non-directory path: {}. If this is a stale placeholder, rerun model preparation with `teamy-studio llm model prepare --overwrite --burn-text-only --with-burn-text`.",
            root.display()
        );
    }
    Ok(())
}

fn remove_existing_path(path: &Path) -> eyre::Result<()> {
    if path.is_dir() {
        std::fs::remove_dir_all(path)
    } else if path.exists() {
        std::fs::remove_file(path)
    } else {
        Ok(())
    }
    .wrap_err_with(|| format!("Failed to remove existing path {}", path.display()))
}

async fn download_to_file(url: &str, destination: &Path) -> eyre::Result<()> {
    if let Some(parent) = destination.parent() {
        tokio::fs::create_dir_all(parent).await.wrap_err_with(|| {
            format!("Failed to create download directory {}", parent.display())
        })?;
    }
    if let Some((repo_id, file_name)) = parse_hugging_face_resolve_url(url)
        && let Some(cached_source) = resolve_local_hugging_face_repo_file(repo_id, file_name)?
    {
        tokio::fs::copy(&cached_source, destination)
            .await
            .wrap_err_with(|| {
                format!(
                    "Failed to copy cached Hugging Face file {} into {}",
                    cached_source.display(),
                    destination.display()
                )
            })?;
        return Ok(());
    }
    let client = reqwest::Client::builder()
        .build()
        .wrap_err("Failed to build HTTP client for model download")?;
    let mut response = client
        .get(url)
        .send()
        .await
        .wrap_err_with(|| format!("Failed to start download from {url}"))?;
    if !response.status().is_success() {
        bail!("Download failed from {url} with HTTP {}", response.status());
    }
    let mut output = tokio::fs::File::create(destination)
        .await
        .wrap_err_with(|| format!("Failed to create {}", destination.display()))?;
    while let Some(chunk) = response.chunk().await.wrap_err_with(|| {
        format!(
            "Failed to stream download body from {url} into {}",
            destination.display()
        )
    })? {
        output.write_all(&chunk).await.wrap_err_with(|| {
            format!(
                "Failed to write download body from {url} into {}",
                destination.display()
            )
        })?;
    }
    output
        .flush()
        .await
        .wrap_err_with(|| format!("Failed to flush downloaded file {}", destination.display()))?;
    Ok(())
}

fn parse_hugging_face_resolve_url(url: &str) -> Option<(&str, &str)> {
    let prefix = "https://huggingface.co/";
    let suffix = "/resolve/main/";
    let remainder = url.strip_prefix(prefix)?;
    let split_index = remainder.find(suffix)?;
    let repo_id = &remainder[..split_index];
    let file_name = &remainder[split_index + suffix.len()..];
    Some((repo_id, file_name))
}

fn resolve_local_hugging_face_repo_file(
    repo_id: &str,
    file_name: &str,
) -> eyre::Result<Option<PathBuf>> {
    for snapshot_dir in local_hugging_face_snapshot_dirs(repo_id) {
        let candidate = snapshot_dir.join(file_name);
        if candidate.is_file() {
            return Ok(Some(candidate));
        }
    }
    Ok(None)
}

fn local_hugging_face_snapshot_dirs(repo_id: &str) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(hf_home) = std::env::var_os("HF_HOME") {
        roots.push(PathBuf::from(hf_home).join("hub"));
    }
    if let Some(hf_hub_cache) = std::env::var_os("HF_HUB_CACHE") {
        roots.push(PathBuf::from(hf_hub_cache));
    }
    if let Some(huggingface_hub_cache) = std::env::var_os("HUGGINGFACE_HUB_CACHE") {
        roots.push(PathBuf::from(huggingface_hub_cache));
    }
    if let Some(user_profile) = std::env::var_os("USERPROFILE") {
        roots.push(
            PathBuf::from(user_profile)
                .join(".cache")
                .join("huggingface")
                .join("hub"),
        );
    }

    let repo_cache_dir_name = format!("models--{}", repo_id.replace('/', "--"));
    let mut snapshots = Vec::new();
    for root in roots {
        let repo_dir = root.join(&repo_cache_dir_name);
        let refs_main = repo_dir.join("refs").join("main");
        let snapshots_dir = repo_dir.join("snapshots");
        if refs_main.is_file()
            && let Ok(revision) = std::fs::read_to_string(&refs_main)
        {
            let snapshot_dir = snapshots_dir.join(revision.trim());
            if snapshot_dir.is_dir() {
                snapshots.push(snapshot_dir);
            }
        }
    }
    snapshots
}

fn write_burn_text_only_model_placeholder(path: &Path) -> eyre::Result<()> {
    std::fs::write(path, BURN_TEXT_ONLY_MODEL_PLACEHOLDER).wrap_err_with(|| {
        format!(
            "Failed to write Burn-text-only model placeholder {}",
            path.display()
        )
    })
}

#[cfg(test)]
mod tests {
    use super::{
        HF_CONFIG_FILE_NAME, KNOWN_LLM_MODELS, MODEL_FILE_NAME, MODEL_METADATA_FILE_NAME,
        TOKENIZER_CONFIG_FILE_NAME, TOKENIZER_FILE_NAME, inspect_model_dir,
        inspect_model_dir_with_tokenizer_validation, inspect_model_files,
        load_tokenizer_config_summary, remove_existing_path, write_model_metadata,
    };
    use tokenizers::{Tokenizer, models::bpe::BPE};

    #[test]
    fn tokenizer_config_summary_reads_string_tokens() {
        let temp = tempfile::tempdir().expect("temp dir");
        let path = temp.path().join(TOKENIZER_CONFIG_FILE_NAME);
        std::fs::write(
            &path,
            r#"{"bos_token":"<s>","eos_token":{"content":"</s>"},"chat_template":"hello"}"#,
        )
        .expect("tokenizer config fixture");
        let summary = load_tokenizer_config_summary(&path).expect("summary should load");
        assert_eq!(summary.bos_token.as_deref(), Some("<s>"));
        assert_eq!(summary.eos_token.as_deref(), Some("</s>"));
        assert_eq!(summary.chat_template.as_deref(), Some("hello"));
    }

    #[test]
    fn inspect_model_dir_accepts_teamy_llm_layout() {
        let temp = tempfile::tempdir().expect("temp dir");
        let root = temp.path();
        std::fs::write(root.join(MODEL_FILE_NAME), b"gguf-fixture").expect("model fixture");
        Tokenizer::new(BPE::default())
            .save(root.join(TOKENIZER_FILE_NAME), false)
            .expect("tokenizer fixture");
        std::fs::write(
            root.join(TOKENIZER_CONFIG_FILE_NAME),
            r#"{"bos_token":"<s>","eos_token":"</s>","chat_template":"template"}"#,
        )
        .expect("tokenizer config fixture");
        std::fs::write(
            root.join(HF_CONFIG_FILE_NAME),
            r#"{"model_type":"qwen3_5","text_config":{"model_type":"qwen3_5_text","layer_types":["linear_attention","full_attention"]}}"#,
        )
        .expect("hf config fixture");
        write_model_metadata(root, &KNOWN_LLM_MODELS[0], false).expect("metadata fixture");

        let artifacts = inspect_model_dir(root).expect("artifacts should load");
        assert_eq!(artifacts.metadata.model_name, KNOWN_LLM_MODELS[0].name);
        assert_eq!(
            artifacts
                .model_path
                .file_name()
                .and_then(|value| value.to_str()),
            Some(MODEL_FILE_NAME)
        );
        assert!(root.join(MODEL_METADATA_FILE_NAME).is_file());
    }

    #[test]
    fn inspection_callback_validates_once_and_standalone_inspection_still_rejects_bad_tokenizer() {
        let temp = tempfile::tempdir().expect("fixture directory");
        let root = temp.path();
        std::fs::write(root.join(MODEL_FILE_NAME), b"model-fixture").unwrap();
        Tokenizer::new(BPE::default())
            .save(root.join(TOKENIZER_FILE_NAME), false)
            .unwrap();
        std::fs::write(root.join(TOKENIZER_CONFIG_FILE_NAME), "{}").unwrap();
        std::fs::write(root.join(HF_CONFIG_FILE_NAME), "{}").unwrap();
        write_model_metadata(root, &KNOWN_LLM_MODELS[0], false).unwrap();
        let calls = std::cell::Cell::new(0);
        let artifacts = inspect_model_dir_with_tokenizer_validation(root, |path| {
            calls.set(calls.get() + 1);
            assert_eq!(path, root.join(TOKENIZER_FILE_NAME));
            Tokenizer::from_file(path)
                .map(|_| ())
                .map_err(|error| eyre::eyre!(error.to_string()))
        })
        .expect("callback validated artifacts");
        assert_eq!(calls.get(), 1);
        assert_eq!(artifacts.metadata.model_name, KNOWN_LLM_MODELS[0].name);

        std::fs::write(root.join(TOKENIZER_FILE_NAME), "invalid tokenizer").unwrap();
        let error = inspect_model_dir(root).expect_err("standalone validation remains mandatory");
        assert!(error.to_string().contains("Failed to load tokenizer"));
        let error = inspect_model_dir_with_tokenizer_validation(root, |_| {
            eyre::bail!("retained tokenizer rejected replacement")
        })
        .expect_err("callback failure must stop inspection");
        assert!(
            error
                .to_string()
                .contains("retained tokenizer rejected replacement")
        );
    }

    fn write_inspection_fixture(root: &std::path::Path) {
        std::fs::write(root.join(MODEL_FILE_NAME), b"model-fixture").unwrap();
        Tokenizer::new(BPE::default())
            .save(root.join(TOKENIZER_FILE_NAME), false)
            .unwrap();
        std::fs::write(root.join(TOKENIZER_CONFIG_FILE_NAME), "{}").unwrap();
        std::fs::write(root.join(HF_CONFIG_FILE_NAME), "{}").unwrap();
        write_model_metadata(root, &KNOWN_LLM_MODELS[0], false).unwrap();
    }

    #[test]
    fn structural_inspection_defers_tokenizer_contents_but_validated_apis_reject_them() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        write_inspection_fixture(root);
        std::fs::write(root.join(TOKENIZER_FILE_NAME), b"invalid tokenizer").unwrap();
        let artifacts = inspect_model_files(root).expect("structural files are complete");
        assert_eq!(artifacts.metadata.model_name, KNOWN_LLM_MODELS[0].name);
        assert_eq!(artifacts.tokenizer_path, root.join(TOKENIZER_FILE_NAME));
        assert!(
            inspect_model_dir(root)
                .unwrap_err()
                .to_string()
                .contains("Failed to load tokenizer")
        );
        let calls = std::cell::Cell::new(0);
        let error = inspect_model_dir_with_tokenizer_validation(root, |path| {
            calls.set(calls.get() + 1);
            Tokenizer::from_file(path)
                .map(|_| ())
                .map_err(|error| eyre::eyre!(error.to_string()))
        })
        .expect_err("validated callback API must reject malformed tokenizer contents");
        assert_eq!(calls.get(), 1);
        assert!(!error.to_string().is_empty());
    }

    #[test]
    fn structural_and_validated_inspection_share_artifacts_and_required_file_checks() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        write_inspection_fixture(root);
        assert_eq!(
            inspect_model_files(root).unwrap(),
            inspect_model_dir(root).unwrap()
        );
        for file in [
            MODEL_FILE_NAME,
            TOKENIZER_FILE_NAME,
            TOKENIZER_CONFIG_FILE_NAME,
            HF_CONFIG_FILE_NAME,
            MODEL_METADATA_FILE_NAME,
        ] {
            let path = root.join(file);
            let bytes = std::fs::read(&path).unwrap();
            std::fs::remove_file(&path).unwrap();
            let structural = inspect_model_files(root).unwrap_err().to_string();
            assert_eq!(structural, inspect_model_dir(root).unwrap_err().to_string());
            let calls = std::cell::Cell::new(0);
            let validated = inspect_model_dir_with_tokenizer_validation(root, |_| {
                calls.set(calls.get() + 1);
                Ok(())
            })
            .unwrap_err();
            assert_eq!(structural, validated.to_string());
            assert_eq!(calls.get(), 0, "incomplete artifacts must not be validated");
            std::fs::write(path, bytes).unwrap();
        }
        std::fs::write(root.join(MODEL_METADATA_FILE_NAME), "invalid metadata").unwrap();
        assert_eq!(
            inspect_model_files(root).unwrap_err().to_string(),
            inspect_model_dir(root).unwrap_err().to_string()
        );
    }

    #[test]
    fn new_gguf_layout_keeps_publisher_filename_and_rejects_metadata_traversal() {
        let temp = tempfile::tempdir().expect("temp dir");
        write_model_metadata(temp.path(), &KNOWN_LLM_MODELS[1], false).unwrap();
        let mut metadata: super::LlmManagedModelMetadata = facet_json::from_slice(
            &std::fs::read(temp.path().join(MODEL_METADATA_FILE_NAME)).unwrap(),
        )
        .unwrap();
        assert_eq!(
            super::declared_model_path(temp.path(), &metadata).unwrap(),
            temp.path().join(super::ORCA_QWEN_GGUF_FILE)
        );
        metadata.model_file_name = "../outside.gguf".into();
        assert!(super::declared_model_path(temp.path(), &metadata).is_err());
        metadata.model_file_name = "model.gguf".into();
        assert!(super::declared_model_path(temp.path(), &metadata).is_err());
        let known = &KNOWN_LLM_MODELS[1];
        assert!(super::model_file_url(known).contains(super::ORCA_QWEN_GGUF_REVISION));
        assert!(super::tokenizer_file_url(known).contains(super::ORCA_QWEN_SOURCE_REVISION));
    }

    #[test]
    fn registered_locations_are_attributed_only_to_their_verified_model() {
        let temp = tempfile::tempdir().unwrap();
        let app = super::AppHome(temp.path().join("app"));
        let cache = super::CacheHome(temp.path().join("cache"));
        let old_root = temp.path().join("old-model");
        let new_root = temp.path().join("new-model");
        for (root, known) in [
            (&old_root, &KNOWN_LLM_MODELS[0]),
            (&new_root, &KNOWN_LLM_MODELS[1]),
        ] {
            std::fs::create_dir(root).unwrap();
            let model_file = if known.name == super::ORCA_QWEN_MODEL_NAME {
                super::ORCA_QWEN_GGUF_FILE
            } else {
                MODEL_FILE_NAME
            };
            std::fs::write(root.join(model_file), b"inventory-fixture").unwrap();
            Tokenizer::new(BPE::default())
                .save(root.join(TOKENIZER_FILE_NAME), false)
                .unwrap();
            std::fs::write(
                root.join(TOKENIZER_CONFIG_FILE_NAME),
                r#"{"bos_token":"<s>","eos_token":"</s>","chat_template":"template"}"#,
            )
            .unwrap();
            std::fs::write(root.join(HF_CONFIG_FILE_NAME),
                r#"{"model_type":"qwen3_5","text_config":{"model_type":"qwen3_5_text","layer_types":["linear_attention","full_attention"]}}"#).unwrap();
            write_model_metadata(root, known, false).unwrap();
            super::add_registered_model_dir(&app, root).unwrap();
        }
        let unknown_root = temp.path().join("unknown-model");
        std::fs::create_dir(&unknown_root).unwrap();
        super::add_registered_model_dir(&app, &unknown_root).unwrap();
        let statuses = super::list_known_llm_model_statuses(&app, &cache).unwrap();
        for (known, expected_root) in [
            (&KNOWN_LLM_MODELS[0], old_root),
            (&KNOWN_LLM_MODELS[1], new_root),
        ] {
            let status = statuses
                .iter()
                .find(|status| status.model_name == known.name)
                .unwrap();
            let registered: Vec<_> = status
                .locations
                .iter()
                .filter(|location| location.label == "registered")
                .collect();
            assert_eq!(registered.len(), 1);
            assert_eq!(registered[0].path, expected_root);
            assert!(registered[0].compatible);
        }
    }

    #[test]
    fn inspect_model_dir_reports_prepare_hint_for_missing_model_dir() {
        let temp = tempfile::tempdir().expect("temp dir");
        let missing = temp.path().join("missing-model");
        let error = inspect_model_dir(&missing).expect_err("missing model dir should fail");
        let message = error.to_string();
        assert!(message.contains("LLM model directory does not exist"));
        assert!(message.contains("llm model prepare"));
    }

    #[test]
    fn remove_existing_path_removes_files_and_directories() {
        let temp = tempfile::tempdir().expect("temp dir");
        let file_path = temp.path().join("stale-file");
        std::fs::write(&file_path, b"stale").expect("stale file");
        remove_existing_path(&file_path).expect("remove stale file");
        assert!(!file_path.exists());

        let dir_path = temp.path().join("stale-dir");
        std::fs::create_dir_all(&dir_path).expect("stale dir");
        std::fs::write(dir_path.join("child"), b"stale").expect("stale child");
        remove_existing_path(&dir_path).expect("remove stale dir");
        assert!(!dir_path.exists());
    }
}
