use eyre::{Context, bail};
use facet::Facet;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

pub const LLM_REFERENCE_SOURCE_PARENT_DIR: &str = "python";
pub const LLM_REFERENCE_SOURCE_DIR_NAME: &str = "llm-reference";
pub const DEFAULT_REFERENCE_MODEL_ID: &str = "Jackrong/Qwopus3.5-9B-Coder";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LlmReferenceCommandOutput {
    pub stdout: String,
    pub stderr: String,
}

#[derive(Clone, Debug, Facet, PartialEq, Eq)]
pub struct LlmReferenceImportReport {
    pub ok: bool,
    pub python: String,
    pub torch: String,
    pub transformers: String,
    pub tokenizers: String,
    pub cuda_available: bool,
}

#[derive(Clone, Debug, Facet, PartialEq)]
pub struct LlmReferencePromptReport {
    pub ok: bool,
    pub model_id: String,
    pub tokenizer_source: String,
    pub device: String,
    pub rendered_prompt: String,
    pub input_token_count: usize,
    pub input_token_ids: Vec<u32>,
    pub top_token_ids: Vec<u32>,
    pub top_token_text: Vec<String>,
    pub top_logits: Vec<f32>,
    pub generated_text: String,
}

#[derive(Clone, Debug, Facet, PartialEq)]
pub struct LlmReferenceLayerReport {
    pub ok: bool,
    pub model_id: String,
    pub tokenizer_source: String,
    pub device: String,
    pub rendered_prompt: String,
    pub input_token_count: usize,
    pub input_token_ids: Vec<u32>,
    pub top_token_ids: Vec<u32>,
    pub top_token_text: Vec<String>,
    pub top_logits: Vec<f32>,
    pub layer_last_hidden_states: Vec<Vec<f32>>,
    pub final_norm_last_hidden: Vec<f32>,
}

#[derive(Clone, Debug, Facet, PartialEq)]
pub struct LlmReferenceConfigReport {
    pub ok: bool,
    pub model_id: String,
    pub config_class: String,
    pub architectures: Vec<String>,
    pub model_name: Option<String>,
    pub model_type: Option<String>,
    pub text_model_type: Option<String>,
    pub text_num_hidden_layers: Option<usize>,
    pub text_hidden_size: Option<usize>,
    pub text_intermediate_size: Option<usize>,
    pub text_num_attention_heads: Option<usize>,
    pub text_num_key_value_heads: Option<usize>,
    pub text_head_dim: Option<usize>,
    pub text_partial_rotary_factor: Option<f32>,
    pub text_full_attention_interval: Option<usize>,
    pub text_linear_num_key_heads: Option<usize>,
    pub text_linear_num_value_heads: Option<usize>,
    pub text_linear_key_head_dim: Option<usize>,
    pub text_linear_value_head_dim: Option<usize>,
    pub text_linear_conv_kernel_dim: Option<usize>,
    pub text_layer_histogram: BTreeMap<String, usize>,
    pub text_layer_types_preview: Vec<String>,
}

#[derive(Clone, Debug, Facet, PartialEq, Eq)]
pub struct LlmReferenceBurnTextExportReport {
    pub ok: bool,
    pub model_id: String,
    pub output_dir: String,
    pub dtype: String,
    pub tensor_count: usize,
}

#[must_use]
pub fn llm_reference_source_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(LLM_REFERENCE_SOURCE_PARENT_DIR)
        .join(LLM_REFERENCE_SOURCE_DIR_NAME)
}

/// # Errors
///
/// This function will return an error if the Python LLM reference harness cannot be launched.
pub fn read_llm_reference_import_report() -> eyre::Result<LlmReferenceImportReport> {
    let output = run_llm_reference(["--check-imports"])?;
    parse_reference_json(&output.stdout, "LLM reference import report")
}

/// # Errors
///
/// This function will return an error if the Python LLM config report cannot be produced.
pub fn read_llm_reference_config_report(model_id: &str) -> eyre::Result<LlmReferenceConfigReport> {
    let output = run_llm_reference(["--config-report", "--model-id", model_id])?;
    parse_reference_json(&output.stdout, "LLM reference config report")
}

/// # Errors
///
/// This function will return an error if the Python LLM prompt report cannot be produced.
pub fn read_llm_reference_prompt_report(
    model_id: &str,
    tokenizer_path: Option<&Path>,
    device: &str,
    system_prompt: Option<&str>,
    user_prompt: &str,
    max_new_tokens: usize,
    top_k: usize,
) -> eyre::Result<LlmReferencePromptReport> {
    let mut args = vec![
        "--prompt-report".to_owned(),
        "--model-id".to_owned(),
        model_id.to_owned(),
        "--device".to_owned(),
        device.to_owned(),
        "--user-prompt".to_owned(),
        user_prompt.to_owned(),
        "--max-new-tokens".to_owned(),
        max_new_tokens.to_string(),
        "--top-k".to_owned(),
        top_k.to_string(),
    ];
    if let Some(tokenizer_path) = tokenizer_path {
        args.push("--tokenizer-path".to_owned());
        args.push(tokenizer_path.display().to_string());
    }
    if let Some(system_prompt) = system_prompt.filter(|value| !value.trim().is_empty()) {
        args.push("--system-prompt".to_owned());
        args.push(system_prompt.to_owned());
    }
    let output = run_llm_reference(args)?;
    parse_reference_json(&output.stdout, "LLM reference prompt report")
}

/// # Errors
///
/// This function will return an error if the Python LLM layer report cannot be produced.
pub fn read_llm_reference_layer_report(
    model_id: &str,
    tokenizer_path: Option<&Path>,
    device: &str,
    system_prompt: Option<&str>,
    user_prompt: &str,
    top_k: usize,
) -> eyre::Result<LlmReferenceLayerReport> {
    let mut args = vec![
        "--layer-report".to_owned(),
        "--model-id".to_owned(),
        model_id.to_owned(),
        "--device".to_owned(),
        device.to_owned(),
        "--user-prompt".to_owned(),
        user_prompt.to_owned(),
        "--top-k".to_owned(),
        top_k.to_string(),
    ];
    if let Some(tokenizer_path) = tokenizer_path {
        args.push("--tokenizer-path".to_owned());
        args.push(tokenizer_path.display().to_string());
    }
    if let Some(system_prompt) = system_prompt.filter(|value| !value.trim().is_empty()) {
        args.push("--system-prompt".to_owned());
        args.push(system_prompt.to_owned());
    }
    let output = run_llm_reference(args)?;
    parse_reference_json(&output.stdout, "LLM reference layer report")
}

/// # Errors
///
/// This function will return an error if the Python LLM Burn text export cannot be produced.
pub fn export_llm_reference_burn_text_model(
    model_id: &str,
    output_dir: &Path,
    dtype: &str,
    overwrite: bool,
) -> eyre::Result<LlmReferenceBurnTextExportReport> {
    let mut args = vec![
        "--export-burn-text".to_owned(),
        "--model-id".to_owned(),
        model_id.to_owned(),
        "--output-dir".to_owned(),
        output_dir.display().to_string(),
        "--dtype".to_owned(),
        dtype.to_owned(),
    ];
    if overwrite {
        args.push("--overwrite".to_owned());
    }
    let output = run_llm_reference(args)?;
    parse_reference_json(&output.stdout, "LLM reference Burn text export report")
}

fn run_llm_reference<I, S>(args: I) -> eyre::Result<LlmReferenceCommandOutput>
where
    I: IntoIterator<Item = S>,
    S: AsRef<std::ffi::OsStr>,
{
    let source_dir = llm_reference_source_dir();
    ensure_reference_source_dir(&source_dir)?;
    let uv_cache_dir = source_dir.join("uv-cache");
    std::fs::create_dir_all(&uv_cache_dir).with_context(|| {
        format!(
            "Failed to create LLM reference uv cache directory {}",
            uv_cache_dir.display()
        )
    })?;
    let output = Command::new("uv")
        .arg("run")
        .arg("teamy-llm-reference")
        .args(args)
        .env("UV_CACHE_DIR", &uv_cache_dir)
        .env("UV_LINK_MODE", "copy")
        .current_dir(&source_dir)
        .output()
        .wrap_err("failed to launch LLM reference harness through uv")?;
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    if !output.status.success() {
        bail!(
            "LLM reference harness exited with {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            stdout,
            stderr
        );
    }
    Ok(LlmReferenceCommandOutput { stdout, stderr })
}

fn ensure_reference_source_dir(path: &Path) -> eyre::Result<()> {
    if !path.is_dir() {
        bail!(
            "LLM reference source directory does not exist: {}",
            path.display()
        );
    }
    Ok(())
}

fn parse_reference_json<T>(stdout: &str, label: &str) -> eyre::Result<T>
where
    T: Facet<'static>,
{
    facet_json::from_str(stdout).with_context(|| {
        format!(
            "Failed to parse {label} from LLM reference harness stdout:\n{}",
            stdout
        )
    })
}
