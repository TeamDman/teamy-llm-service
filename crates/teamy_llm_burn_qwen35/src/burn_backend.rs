use facet::Facet;
use std::collections::BTreeMap;

use crate::burn_text::inspect_burn_text_runtime_status;
use crate::model::LlmModelArtifacts;
use crate::source_config::load_llm_source_config_summary;

#[repr(u8)]
#[derive(Clone, Copy, Debug, Facet, PartialEq, Eq)]
pub enum BurnLlmSupportState {
    InventoryOnly,
    TextArchitectureCaptured,
    Ready,
}

#[derive(Clone, Debug, Facet, PartialEq, Eq)]
pub struct BurnLlmRuntimeSupportReport {
    pub backend: String,
    pub state: BurnLlmSupportState,
    pub model_family: String,
    pub architecture: String,
    pub text_model_type: Option<String>,
    pub layer_type_histogram: BTreeMap<String, usize>,
    pub reason: String,
}

/// Inspect the Teamy Burn runtime support level for a prepared LLM model.
///
/// # Errors
///
/// This function will return an error if the captured Hugging Face config cannot be parsed.
pub fn inspect_burn_runtime_support(
    artifacts: &LlmModelArtifacts,
) -> eyre::Result<BurnLlmRuntimeSupportReport> {
    let source_config = load_llm_source_config_summary(&artifacts.hf_config_path)?;
    let text_model_type = source_config.text_model_type.clone();
    let layer_type_histogram = source_config.text_layer_type_counts.clone();
    let burn_text_status = inspect_burn_text_runtime_status(&artifacts.root);

    let (state, reason) = match text_model_type.as_deref() {
        Some("qwen3_5_text") if burn_text_status.exists => (
            BurnLlmSupportState::Ready,
            format!(
                "Teamy found a converted Burn text runtime bundle at {} and can execute Rust-only text generation through the lazy-load Qwen3.5 backend.",
                burn_text_status.manifest_path
            ),
        ),
        Some("qwen3_5_text") => (
            BurnLlmSupportState::TextArchitectureCaptured,
            format!(
                "Teamy has captured the authoritative Qwen3.5 text architecture and can execute it through Burn once a converted text bundle is present. No Burn text manifest was found at {} yet.",
                burn_text_status.manifest_path
            ),
        ),
        Some(other) => (
            BurnLlmSupportState::InventoryOnly,
            format!(
                "Teamy has not implemented Burn runtime support for Hugging Face text model type `{other}` yet."
            ),
        ),
        None => (
            BurnLlmSupportState::InventoryOnly,
            "Teamy could not find a text-model subtype in the captured Hugging Face config, so Burn runtime support remains inventory-only.".to_owned(),
        ),
    };

    Ok(BurnLlmRuntimeSupportReport {
        backend: "burn".to_owned(),
        state,
        model_family: artifacts.metadata.family.clone(),
        architecture: artifacts.metadata.architecture.clone(),
        text_model_type,
        layer_type_histogram,
        reason,
    })
}

#[must_use]
pub fn render_burn_runtime_support_report(report: &BurnLlmRuntimeSupportReport) -> String {
    let histogram = if report.layer_type_histogram.is_empty() {
        "<none>".to_owned()
    } else {
        report
            .layer_type_histogram
            .iter()
            .map(|(layer_type, count)| format!("{layer_type}={count}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    format!(
        "Burn backend: {}\nBurn support state: {:?}\nBurn text model type: {}\nBurn layer histogram: {}\nBurn support note: {}",
        report.backend,
        report.state,
        report.text_model_type.as_deref().unwrap_or("<unknown>"),
        histogram,
        report.reason
    )
}

#[cfg(test)]
mod tests {
    use super::{BurnLlmSupportState, inspect_burn_runtime_support};
    use crate::model::{
        HF_CONFIG_FILE_NAME, LlmManagedModelMetadata, LlmModelArtifacts, MODEL_FILE_NAME,
        MODEL_METADATA_FILE_NAME, TOKENIZER_CONFIG_FILE_NAME, TOKENIZER_FILE_NAME,
    };

    fn fixture_artifacts(root: &std::path::Path) -> LlmModelArtifacts {
        LlmModelArtifacts {
            root: root.to_path_buf(),
            model_path: root.join(MODEL_FILE_NAME),
            tokenizer_path: root.join(TOKENIZER_FILE_NAME),
            tokenizer_config_path: root.join(TOKENIZER_CONFIG_FILE_NAME),
            hf_config_path: root.join(HF_CONFIG_FILE_NAME),
            mmproj_path: None,
            metadata_path: root.join(MODEL_METADATA_FILE_NAME),
            metadata: LlmManagedModelMetadata {
                model_name: "qwopus-3.5-9b-coder-q4-k-m".to_owned(),
                family: "qwopus".to_owned(),
                display_name: "fixture".to_owned(),
                source_repo_id: "Jackrong/Qwopus3.5-9B-Coder".to_owned(),
                model_repo_id: "Jackrong/Qwopus3.5-9B-Coder-GGUF".to_owned(),
                tokenizer_repo_id: "Jackrong/Qwopus3.5-9B-Coder".to_owned(),
                architecture: "qwen35".to_owned(),
                quantization: "Q4_K_M".to_owned(),
                model_file_name: "model.gguf".to_owned(),
                mmproj_file_name: None,
                hf_config_file_name: "config.json".to_owned(),
                tokenizer_file_name: "tokenizer.json".to_owned(),
                tokenizer_config_file_name: "tokenizer_config.json".to_owned(),
                parameter_count: "9B".to_owned(),
                size_estimate: "5.63 GiB".to_owned(),
                supports_vision: true,
                supports_tool_calling: true,
            },
        }
    }

    #[test]
    fn burn_support_reports_qwen35_as_text_architecture_captured() {
        let temp = tempfile::tempdir().expect("temp dir");
        std::fs::write(
            temp.path().join(HF_CONFIG_FILE_NAME),
            r#"{"model_type":"qwen3_5","text_config":{"model_type":"qwen3_5_text","layer_types":["linear_attention","full_attention"]}}"#,
        )
        .expect("hf config fixture");

        let report = inspect_burn_runtime_support(&fixture_artifacts(temp.path()))
            .expect("burn support report should load");
        assert_eq!(report.state, BurnLlmSupportState::TextArchitectureCaptured);
        assert_eq!(
            report.layer_type_histogram.get("linear_attention"),
            Some(&1)
        );
        assert_eq!(report.layer_type_histogram.get("full_attention"), Some(&1));
    }
}
