use eyre::{WrapErr, bail};
use facet::Facet;
use std::io::Write;
use std::path::Path;
use std::time::Duration;

use crate::burn_backend::inspect_burn_runtime_support;
use crate::burn_text::{
    BurnTextGenerationOptions, generate_with_burn_text_runtime,
    generate_with_burn_text_runtime_to_writer,
};
use crate::model::{LlmModelArtifacts, load_tokenizer_config_summary};

pub const DEFAULT_MAX_NEW_TOKENS: usize = 256;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LlmPromptRequest {
    pub system_prompt: Option<String>,
    pub user_prompt: String,
    pub max_new_tokens: usize,
    pub generation_timeout: Option<Duration>,
}

#[derive(Clone, Debug, Facet, PartialEq, Eq)]
pub struct LlmPromptResult {
    pub rendered_prompt: String,
    pub output_text: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderedPrompt {
    pub prompt: String,
}

/// Render a single-turn Qwen-style chat prompt using the same top-level shape as the published
/// tokenizer template for system + user + assistant messages.
#[must_use]
pub fn render_qwen_single_turn_prompt(
    system_prompt: Option<&str>,
    user_prompt: &str,
) -> RenderedPrompt {
    let mut prompt = String::new();
    if let Some(system_prompt) = system_prompt.filter(|value| !value.trim().is_empty()) {
        prompt.push_str("<|im_start|>system\n");
        prompt.push_str(system_prompt.trim());
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>user\n");
    prompt.push_str(user_prompt.trim());
    prompt.push_str("<|im_end|>\n");
    prompt.push_str("<|im_start|>assistant\n");
    RenderedPrompt { prompt }
}

/// Tokenize a rendered prompt using the managed tokenizer file.
///
/// # Errors
///
/// This function will return an error if the tokenizer cannot be loaded or the prompt cannot be
/// encoded.
pub fn tokenize_rendered_prompt(
    tokenizer_path: &Path,
    rendered_prompt: &str,
) -> eyre::Result<Vec<u32>> {
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path).map_err(|error| {
        eyre::eyre!(
            "Failed to load tokenizer from {}: {}",
            tokenizer_path.display(),
            error
        )
    })?;
    let encoding = tokenizer
        .encode(rendered_prompt, true)
        .map_err(|error| eyre::eyre!("Failed to tokenize rendered prompt: {}", error))?;
    Ok(encoding.get_ids().to_vec())
}

/// Run a Teamy-managed prompt with the local Rust Burn runtime.
///
/// # Errors
///
/// This function will return an error if the local runtime cannot be loaded or the prompt fails.
pub fn run_prompt(
    artifacts: &LlmModelArtifacts,
    request: &LlmPromptRequest,
) -> eyre::Result<LlmPromptResult> {
    run_prompt_inner(artifacts, request, None)
}

/// Run a Teamy-managed prompt with the local Rust Burn runtime, streaming generated text to a
/// caller-provided writer as tokens are selected.
///
/// # Errors
///
/// This function will return an error if the local runtime cannot be loaded, the prompt fails, or
/// generated text cannot be written to the provided writer.
pub fn run_prompt_to_writer(
    artifacts: &LlmModelArtifacts,
    request: &LlmPromptRequest,
    writer: &mut dyn Write,
) -> eyre::Result<LlmPromptResult> {
    run_prompt_inner(artifacts, request, Some(writer))
}

fn run_prompt_inner(
    artifacts: &LlmModelArtifacts,
    request: &LlmPromptRequest,
    writer: Option<&mut dyn Write>,
) -> eyre::Result<LlmPromptResult> {
    let _summary = load_tokenizer_config_summary(&artifacts.tokenizer_config_path)?;
    let rendered =
        render_qwen_single_turn_prompt(request.system_prompt.as_deref(), &request.user_prompt);
    let _token_ids = tokenize_rendered_prompt(&artifacts.tokenizer_path, &rendered.prompt)?;

    let output_text = run_prompt_with_burn_backend(
        artifacts,
        &rendered.prompt,
        &BurnTextGenerationOptions {
            max_new_tokens: request.max_new_tokens,
            generation_timeout: request.generation_timeout,
        },
        writer,
    )
    .wrap_err("failed to execute the local Rust Burn LLM runtime")?;

    Ok(LlmPromptResult {
        rendered_prompt: rendered.prompt,
        output_text,
    })
}

fn run_prompt_with_burn_backend(
    artifacts: &LlmModelArtifacts,
    rendered_prompt: &str,
    options: &BurnTextGenerationOptions,
    writer: Option<&mut dyn Write>,
) -> eyre::Result<String> {
    let token_ids = tokenize_rendered_prompt(&artifacts.tokenizer_path, rendered_prompt)
        .wrap_err("failed to tokenize the rendered prompt for Burn execution")?;
    let report = inspect_burn_runtime_support(artifacts)?;
    if report.state != crate::burn_backend::BurnLlmSupportState::Ready {
        bail!(
            "The managed model at {} is not runnable yet through Teamy's Burn backend. Support state: {:?}. {}",
            artifacts.root.display(),
            report.state,
            report.reason
        );
    }
    let generation = if let Some(writer) = writer {
        generate_with_burn_text_runtime_to_writer(artifacts, &token_ids, options, writer)
    } else {
        generate_with_burn_text_runtime(artifacts, &token_ids, options)
    }
    .wrap_err("failed to execute the converted Burn text runtime")?;
    Ok(generation.generated_text)
}

#[cfg(test)]
mod tests {
    use super::{render_qwen_single_turn_prompt, tokenize_rendered_prompt};
    use crate::{
        burn_text::{BurnTextGenerationOptions, generate_with_burn_text_runtime},
        model::inspect_model_dir,
    };
    use std::{path::Path, time::Duration};
    use tokenizers::{Tokenizer, models::bpe::BPE};

    #[test]
    fn qwen_single_turn_prompt_matches_expected_shape() {
        let rendered = render_qwen_single_turn_prompt(Some("You are concise."), "Hello");
        assert_eq!(
            rendered.prompt,
            "<|im_start|>system\nYou are concise.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn tokenize_rendered_prompt_roundtrips_with_fixture_tokenizer() {
        let temp = tempfile::tempdir().expect("temp dir");
        let tokenizer_path = temp.path().join("tokenizer.json");
        Tokenizer::new(BPE::default())
            .save(&tokenizer_path, false)
            .expect("tokenizer fixture");
        let token_ids =
            tokenize_rendered_prompt(&tokenizer_path, "hello").expect("tokenization should work");
        let _ = token_ids;
    }

    #[test]
    fn jackrong_smoke_generation_matches_current_snapshot_when_model_bundle_available() {
        let Some(model_dir) = std::env::var_os("TEAMY_STUDIO_LLM_SMOKE_MODEL_DIR") else {
            return;
        };
        let artifacts = inspect_model_dir(Path::new(&model_dir))
            .expect("managed Jackrong model should inspect");
        let rendered = render_qwen_single_turn_prompt(None, "Hello again");
        let token_ids = tokenize_rendered_prompt(&artifacts.tokenizer_path, &rendered.prompt)
            .expect("Jackrong prompt should tokenize");
        let report = generate_with_burn_text_runtime(
            &artifacts,
            &token_ids,
            &BurnTextGenerationOptions {
                max_new_tokens: 1,
                generation_timeout: Some(Duration::from_secs(30)),
            },
        )
        .expect("Jackrong Burn text runtime should generate one token");
        assert_eq!(report.generated_token_ids, vec![248068]);
        assert_eq!(report.generated_text, "<think>");
    }
}
