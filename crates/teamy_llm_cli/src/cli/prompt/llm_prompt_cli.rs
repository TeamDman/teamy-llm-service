use crate::cli::{
    model_selection::ModelSelectionArgs,
    output::{OutputFormat, emit_report},
};
use eyre::{Result, ensure};
use facet::Facet;
use figue as args;
use std::io::{self, Write};
use std::time::Instant;
use teamy_cancellation::CancellationToken;
use teamy_llm_core::{GenerateEvent, GenerateRequest, GenerateResult, ThinkingMode};
use teamy_llm_service::{ORCA_MODEL_NAME, TeamyLlmService};

/// Generate text from the local model and stream it to stdout.
///
/// Orca is text-only with a 1024-token prompt-plus-output context. Named Orca
/// thinking mode defaults to 512 new tokens; lower the limit for longer prompts.
/// --timeout-ms bounds generation excluding queue/model loading; global
/// --stop-after-duration cooperatively cancels from CLI startup.
#[derive(Debug, Facet)]
pub struct PromptCli {
    #[facet(flatten, default)]
    pub selection: ModelSelectionArgs,
    /// Optional text-generation system prompt.
    #[facet(args::named)]
    pub system_prompt: Option<String>,
    /// Enable reasoning; --no-thinking selects the default direct mode.
    #[facet(args::named, default)]
    pub thinking: bool,
    /// Positive generation limit. Direct default 256; thinking model-dependent.
    #[facet(args::named)]
    pub max_new_tokens: Option<usize>,
    /// Positive generation timeout excluding model loading and queueing.
    #[facet(args::named)]
    pub timeout_ms: Option<u64>,
    /// Prompt text. Put literal flag-like text after --.
    #[facet(args::positional)]
    pub prompt: Vec<String>,
}
#[derive(Debug, Facet)]
#[facet(rename_all = "kebab-case")]
struct PromptOutput {
    version: u32,
    elapsed_ms: f64,
    result: GenerateResult,
}

pub fn generation_request(
    service: &TeamyLlmService,
    selection: &ModelSelectionArgs,
    system_prompt: Option<String>,
    thinking: bool,
    max_new_tokens: Option<usize>,
    timeout_ms: Option<u64>,
    user_prompt: String,
) -> Result<GenerateRequest> {
    ensure!(
        !user_prompt.is_empty(),
        "prompt command requires prompt text"
    );
    ensure!(timeout_ms != Some(0), "--timeout-ms must be at least 1");
    let selector = selection.to_selector()?;
    let thinking_mode = if thinking {
        ThinkingMode::Thinking
    } else {
        ThinkingMode::Direct
    };
    let max_new_tokens = match max_new_tokens {
        Some(limit) => limit,
        None if !thinking => 256,
        None => {
            let is_orca = match selection.model.as_deref() {
                Some(name) => name.trim().eq_ignore_ascii_case(ORCA_MODEL_NAME),
                None if selection.model_dir.is_none() => {
                    service.default_model()?.model_name == ORCA_MODEL_NAME
                }
                None => false,
            };
            if is_orca { 512 } else { 1024 }
        }
    };
    ensure!(max_new_tokens > 0, "--max-new-tokens must be at least 1");
    Ok(GenerateRequest {
        selector,
        system_prompt,
        user_prompt,
        thinking_mode,
        max_new_tokens,
        timeout_ms,
    })
}

impl PromptCli {
    pub fn into_request(self, service: &TeamyLlmService) -> Result<GenerateRequest> {
        generation_request(
            service,
            &self.selection,
            self.system_prompt,
            self.thinking,
            self.max_new_tokens,
            self.timeout_ms,
            self.prompt.join(" "),
        )
    }
    pub fn invoke(
        self,
        service: &TeamyLlmService,
        output: Option<OutputFormat>,
        cancellation: CancellationToken,
    ) -> Result<()> {
        ensure!(
            output != Some(OutputFormat::Csv),
            "prompt supports text or JSON output"
        );
        let _span = tracing::info_span!("llm_prompt").entered();
        if self.selection.model.is_none() && self.selection.model_dir.is_none() {
            let default = service.default_model()?;
            tracing::warn!(model = %default.model_name, "No --model supplied; using the configured default. Specify --model for reproducible selection. Run `teamy-llm model list` to inspect available models.");
            for available in service.list_models()? {
                tracing::warn!(model = %available.model_name, state = ?available.state, "model selection");
            }
        }
        let request = self.into_request(service)?;
        let stream_text = output != Some(OutputFormat::Json);
        let mut stdout = io::stdout();
        let requested_at = Instant::now();
        let mut first_token = true;
        let result =
            service.generate_with_cancellation(request, Some(cancellation), &mut |event| {
                match event {
                    GenerateEvent::Token { text, .. } => {
                        if first_token {
                            tracing::info!(
                                elapsed_ms = requested_at.elapsed().as_secs_f64() * 1000.0,
                                "first generated token"
                            );
                            first_token = false;
                        }
                        if stream_text {
                            write!(stdout, "{text}")
                                .and_then(|()| stdout.flush())
                                .map_err(|error| {
                                    teamy_llm_core::ServiceError::new(
                                        teamy_llm_core::ServiceErrorKind::Internal,
                                        error.to_string(),
                                    )
                                })?;
                        }
                    }
                    GenerateEvent::Failed { message } => {
                        tracing::error!(message, "generation failed")
                    }
                    _ => {}
                }
                Ok(())
            })?;
        if stream_text {
            writeln!(stdout)?;
            stdout.flush()?;
        } else {
            emit_report(
                &PromptOutput {
                    version: 1,
                    elapsed_ms: requested_at.elapsed().as_secs_f64() * 1000.0,
                    result,
                },
                Some(OutputFormat::Json),
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use teamy_llm_burn_qwen35::paths::{AppHome, CacheHome};

    #[test]
    fn named_orca_thinking_budget_matches_case_insensitive_inventory_selection() {
        // Explicit names resolve their budget without reading model artifacts or
        // the current user's default-model record.
        let service = TeamyLlmService::new(
            AppHome("synthetic-app-home".into()),
            CacheHome("synthetic-cache-home".into()),
        );
        for name in [
            ORCA_MODEL_NAME.to_owned(),
            format!("  {}  ", ORCA_MODEL_NAME.to_ascii_uppercase()),
        ] {
            let request = generation_request(
                &service,
                &ModelSelectionArgs {
                    model: Some(name),
                    model_dir: None,
                },
                None,
                true,
                None,
                None,
                "example".into(),
            )
            .unwrap();
            assert_eq!(request.max_new_tokens, 512);
        }
    }
}
