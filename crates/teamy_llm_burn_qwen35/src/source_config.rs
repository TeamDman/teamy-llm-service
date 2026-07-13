use facet::Facet;
use facet_json::RawJson;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LlmSourceConfigSummary {
    pub path: PathBuf,
    pub architectures: Vec<String>,
    pub model_type: Option<String>,
    pub model_name: Option<String>,
    pub text_model_type: Option<String>,
    pub text_num_hidden_layers: Option<usize>,
    pub text_hidden_size: Option<usize>,
    pub text_intermediate_size: Option<usize>,
    pub text_num_attention_heads: Option<usize>,
    pub text_num_key_value_heads: Option<usize>,
    pub text_head_dim: Option<usize>,
    pub text_hidden_act: Option<String>,
    pub text_partial_rotary_factor: Option<String>,
    pub text_rope_theta: Option<String>,
    pub text_full_attention_interval: Option<usize>,
    pub text_linear_num_key_heads: Option<usize>,
    pub text_linear_num_value_heads: Option<usize>,
    pub text_linear_key_head_dim: Option<usize>,
    pub text_linear_value_head_dim: Option<usize>,
    pub text_linear_conv_kernel_dim: Option<usize>,
    pub text_layer_type_counts: BTreeMap<String, usize>,
    pub text_layer_types_preview: Vec<String>,
}

#[derive(Clone, Debug, Facet, PartialEq)]
struct LlmSourceConfigFile {
    #[facet(default)]
    architectures: Vec<String>,
    #[facet(default)]
    model_name: Option<String>,
    #[facet(default)]
    model_type: Option<String>,
    #[facet(default)]
    text_config: Option<LlmSourceTextConfigFile>,
}

#[derive(Clone, Debug, Facet, PartialEq)]
struct LlmSourceTextConfigFile {
    #[facet(default)]
    model_type: Option<String>,
    #[facet(default)]
    num_hidden_layers: Option<usize>,
    #[facet(default)]
    hidden_size: Option<usize>,
    #[facet(default)]
    intermediate_size: Option<usize>,
    #[facet(default)]
    num_attention_heads: Option<usize>,
    #[facet(default)]
    num_key_value_heads: Option<usize>,
    #[facet(default)]
    head_dim: Option<usize>,
    #[facet(default)]
    hidden_act: Option<String>,
    #[facet(default)]
    partial_rotary_factor: Option<RawJson<'static>>,
    #[facet(default)]
    rope_theta: Option<RawJson<'static>>,
    #[facet(default)]
    full_attention_interval: Option<usize>,
    #[facet(default)]
    linear_num_key_heads: Option<usize>,
    #[facet(default)]
    linear_num_value_heads: Option<usize>,
    #[facet(default)]
    linear_key_head_dim: Option<usize>,
    #[facet(default)]
    linear_value_head_dim: Option<usize>,
    #[facet(default)]
    linear_conv_kernel_dim: Option<usize>,
    #[facet(default)]
    layer_types: Vec<String>,
}

/// # Errors
///
/// This function will return an error if the Hugging Face config file cannot be read or parsed.
pub fn load_llm_source_config_summary(path: &Path) -> eyre::Result<LlmSourceConfigSummary> {
    let parsed: LlmSourceConfigFile =
        facet_json::from_slice(&std::fs::read(path).map_err(|error| {
            eyre::eyre!(
                "Failed to read Hugging Face config {}: {}",
                path.display(),
                error
            )
        })?)
        .map_err(|error| {
            eyre::eyre!(
                "Failed to parse Hugging Face config {}: {}",
                path.display(),
                error
            )
        })?;

    let mut text_layer_type_counts = BTreeMap::new();
    let mut text_layer_types_preview = Vec::new();
    if let Some(text_config) = &parsed.text_config {
        for layer_type in &text_config.layer_types {
            *text_layer_type_counts
                .entry(layer_type.clone())
                .or_insert(0) += 1;
        }
        text_layer_types_preview = text_config.layer_types.iter().take(8).cloned().collect();
    }

    Ok(LlmSourceConfigSummary {
        path: path.to_path_buf(),
        architectures: parsed.architectures,
        model_type: parsed.model_type,
        model_name: parsed.model_name,
        text_model_type: parsed
            .text_config
            .as_ref()
            .and_then(|text_config| text_config.model_type.clone()),
        text_num_hidden_layers: parsed
            .text_config
            .as_ref()
            .and_then(|text_config| text_config.num_hidden_layers),
        text_hidden_size: parsed
            .text_config
            .as_ref()
            .and_then(|text_config| text_config.hidden_size),
        text_intermediate_size: parsed
            .text_config
            .as_ref()
            .and_then(|text_config| text_config.intermediate_size),
        text_num_attention_heads: parsed
            .text_config
            .as_ref()
            .and_then(|text_config| text_config.num_attention_heads),
        text_num_key_value_heads: parsed
            .text_config
            .as_ref()
            .and_then(|text_config| text_config.num_key_value_heads),
        text_head_dim: parsed
            .text_config
            .as_ref()
            .and_then(|text_config| text_config.head_dim),
        text_hidden_act: parsed
            .text_config
            .as_ref()
            .and_then(|text_config| text_config.hidden_act.clone()),
        text_partial_rotary_factor: parsed.text_config.as_ref().and_then(|text_config| {
            text_config
                .partial_rotary_factor
                .clone()
                .map(render_json_value)
        }),
        text_rope_theta: parsed
            .text_config
            .as_ref()
            .and_then(|text_config| text_config.rope_theta.clone().map(render_json_value)),
        text_full_attention_interval: parsed
            .text_config
            .as_ref()
            .and_then(|text_config| text_config.full_attention_interval),
        text_linear_num_key_heads: parsed
            .text_config
            .as_ref()
            .and_then(|text_config| text_config.linear_num_key_heads),
        text_linear_num_value_heads: parsed
            .text_config
            .as_ref()
            .and_then(|text_config| text_config.linear_num_value_heads),
        text_linear_key_head_dim: parsed
            .text_config
            .as_ref()
            .and_then(|text_config| text_config.linear_key_head_dim),
        text_linear_value_head_dim: parsed
            .text_config
            .as_ref()
            .and_then(|text_config| text_config.linear_value_head_dim),
        text_linear_conv_kernel_dim: parsed
            .text_config
            .as_ref()
            .and_then(|text_config| text_config.linear_conv_kernel_dim),
        text_layer_type_counts,
        text_layer_types_preview,
    })
}

fn render_json_value(value: RawJson<'static>) -> String {
    let raw = value.as_ref();
    if raw == "null" {
        return "null".to_owned();
    }
    if let Ok(parsed) = facet_json::from_str::<bool>(raw) {
        return parsed.to_string();
    }
    if let Ok(parsed) = facet_json::from_str::<f64>(raw) {
        return parsed.to_string();
    }
    if let Ok(parsed) = facet_json::from_str::<String>(raw) {
        return parsed;
    }
    raw.to_owned()
}

#[cfg(test)]
mod tests {
    use super::load_llm_source_config_summary;

    #[test]
    fn source_config_summary_reads_hybrid_qwen35_fields() {
        let temp = tempfile::tempdir().expect("temp dir");
        let path = temp.path().join("config.json");
        std::fs::write(
            &path,
            r#"{
  "architectures":["Qwen3_5ForConditionalGeneration"],
  "model_name":"Jackrong/Qwopus3.5-9B-v3.5",
  "model_type":"qwen3_5",
  "text_config":{
    "model_type":"qwen3_5_text",
    "num_hidden_layers":32,
    "hidden_size":4096,
    "intermediate_size":12288,
    "num_attention_heads":16,
    "num_key_value_heads":4,
    "head_dim":256,
    "hidden_act":"silu",
    "partial_rotary_factor":0.25,
    "rope_theta":10000000.0,
    "full_attention_interval":4,
    "linear_num_key_heads":16,
    "linear_num_value_heads":32,
    "linear_key_head_dim":128,
    "linear_value_head_dim":128,
    "linear_conv_kernel_dim":4,
    "layer_types":["linear_attention","linear_attention","linear_attention","full_attention"]
  }
}"#,
        )
        .expect("fixture config");
        let summary = load_llm_source_config_summary(&path).expect("summary should load");
        assert_eq!(summary.model_type.as_deref(), Some("qwen3_5"));
        assert_eq!(summary.text_model_type.as_deref(), Some("qwen3_5_text"));
        assert_eq!(summary.text_num_hidden_layers, Some(32));
        assert_eq!(summary.text_hidden_act.as_deref(), Some("silu"));
        assert_eq!(summary.text_rope_theta.as_deref(), Some("10000000"));
        assert_eq!(summary.text_full_attention_interval, Some(4));
        assert_eq!(
            summary.text_layer_type_counts.get("linear_attention"),
            Some(&3)
        );
        assert_eq!(
            summary.text_layer_type_counts.get("full_attention"),
            Some(&1)
        );
    }
}
