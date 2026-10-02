use crate::cli::{
    model_selection::ModelSelectionArgs,
    output::{OutputFormat, emit_report},
};
use eyre::{Result, WrapErr, ensure};
use facet::Facet;
use figue as args;
use std::io::{self, Write};
use std::time::Instant;
use teamy_cancellation::CancellationToken;
use teamy_llm_core::{GenerateEvent, GenerateRequest, GenerateResult, ThinkingMode};
use teamy_llm_service::{ORCA_MODEL_NAME, TeamyLlmService};

/// Generate text from the local model and stream it to stdout.
///
/// Orca is text-only. --context-tokens sets the total prompt-plus-output capacity.
/// Named Orca thinking mode defaults to 512 new tokens.
/// --timeout-ms bounds generation excluding queue/model loading; global
/// --stop-after-duration cooperatively cancels from CLI startup.
#[derive(Debug, Facet)]
pub struct PromptCli {
    #[facet(flatten, default)]
    pub selection: ModelSelectionArgs,
    /// Attach to a resident local:// service instead of loading a model here.
    /// Requires an explicit --model; connection failure never falls back locally.
    #[facet(args::named)]
    pub address: Option<String>,
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
    /// Positive total formatted input and output token capacity.
    #[facet(args::named)]
    pub context_tokens: Option<u32>,
    /// Prompt text, or a solitary - for bounded UTF-8 stdin.
    /// Put literal flag-like text after --.
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
        context_tokens: None,
    })
}

impl PromptCli {
    pub fn into_request(self, service: &TeamyLlmService) -> Result<GenerateRequest> {
        ensure!(
            self.context_tokens != Some(0),
            "--context-tokens must be at least 1"
        );
        let mut request = generation_request(
            service,
            &self.selection,
            self.system_prompt,
            self.thinking,
            self.max_new_tokens,
            self.timeout_ms,
            self.prompt.join(" "),
        )?;
        request.context_tokens = self.context_tokens;
        Ok(request)
    }
    #[tracing::instrument(name = "llm_prompt", skip_all)]
    pub async fn invoke(
        mut self,
        service: &TeamyLlmService,
        output: Option<OutputFormat>,
        cancellation: CancellationToken,
    ) -> Result<()> {
        ensure!(
            output != Some(OutputFormat::Csv),
            "prompt supports text or JSON output"
        );
        let requested_at = Instant::now();
        let address = self.address.take();
        if let Some(address) = address.as_deref() {
            crate::cli::serve::validate_local_address(address)?;
            ensure!(
                self.selection.model.is_some() && self.selection.model_dir.is_none(),
                "--address requires an explicit --model and does not accept --model-dir"
            );
        }
        ensure!(
            self.context_tokens != Some(0),
            "--context-tokens must be at least 1"
        );
        if self.selection.model.is_none() && self.selection.model_dir.is_none() {
            tracing::warn!(
                "No --model supplied; using the configured default. Specify --model for reproducible selection. Run `teamy-llm model default show` to inspect the default and `teamy-llm model list` to inspect available models."
            );
        }
        self.prompt = vec![super::prompt_input::resolve_prompt(self.prompt, &cancellation).await?];
        let request = self.into_request(service)?;
        let stream_text = output != Some(OutputFormat::Json);
        let mut stdout = io::stdout();
        let mut observer = PromptObserver {
            writer: &mut stdout,
            stream_text,
            requested_at,
            first_token: true,
            first_visible: true,
        };
        let result = if let Some(address) = address.as_deref() {
            super::streaming::generate_resident(address, request, &cancellation, &mut |event| {
                observer
                    .observe(event)
                    .wrap_err("cannot write prompt output")
            })
            .await?
        } else {
            service.generate_with_cancellation(request, Some(cancellation), &mut |event| {
                observer.observe(event).map_err(|error| {
                    teamy_llm_core::ServiceError::new(
                        teamy_llm_core::ServiceErrorKind::Internal,
                        error.to_string(),
                    )
                })
            })?
        };
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

struct PromptObserver<'a, W> {
    writer: &'a mut W,
    stream_text: bool,
    requested_at: Instant,
    first_token: bool,
    first_visible: bool,
}
impl<W: Write> PromptObserver<'_, W> {
    fn observe(&mut self, event: GenerateEvent) -> io::Result<()> {
        match event {
            GenerateEvent::Token { text, .. } => {
                if self.first_token {
                    tracing::info!(
                        elapsed_ms = self.requested_at.elapsed().as_secs_f64() * 1000.0,
                        "first generated token"
                    );
                    self.first_token = false;
                }
                if self.stream_text {
                    self.writer.write_all(text.as_bytes())?;
                    self.writer.flush()?;
                }
                // Empty tokens, whitespace and format controls do not qualify
                // as the first visible model text. JSON records are emitted
                // only after the complete result; this span records readiness.
                if self.first_visible && has_visible_text(&text) {
                    tracing::info!(
                        elapsed_ms = self.requested_at.elapsed().as_secs_f64() * 1000.0,
                        streamed_to_stdout = self.stream_text,
                        "first visible model text"
                    );
                    self.first_visible = false;
                }
            }
            GenerateEvent::Failed { message } => tracing::error!(message, "generation failed"),
            _ => {}
        }
        Ok(())
    }
}
fn has_visible_text(text: &str) -> bool {
    text.chars().any(|character| {
        !character.is_whitespace()
            && !character.is_control()
            && !matches!(character, '\u{200b}'..='\u{200f}' | '\u{202a}'..='\u{202e}' | '\u{2060}'..='\u{206f}' | '\u{feff}')
    })
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

    #[test]
    fn empty_tokens_do_not_count_as_visible_and_json_does_not_stream() {
        let mut bytes = Vec::new();
        let mut observer = PromptObserver {
            writer: &mut bytes,
            stream_text: false,
            requested_at: Instant::now(),
            first_token: true,
            first_visible: true,
        };
        for text in ["", "\n \t", "\u{200b}\u{202e}"] {
            observer
                .observe(GenerateEvent::Token {
                    token_id: 1,
                    text: text.into(),
                })
                .unwrap();
            assert!(observer.first_visible);
        }
        observer
            .observe(GenerateEvent::Token {
                token_id: 2,
                text: "café".into(),
            })
            .unwrap();
        assert!(!observer.first_visible);
        assert!(bytes.is_empty());
    }

    #[test]
    fn streamed_unicode_is_preserved_and_writer_failures_are_errors() {
        let mut bytes = Vec::new();
        let mut observer = PromptObserver {
            writer: &mut bytes,
            stream_text: true,
            requested_at: Instant::now(),
            first_token: true,
            first_visible: true,
        };
        observer
            .observe(GenerateEvent::Token {
                token_id: 1,
                text: "café\n".into(),
            })
            .unwrap();
        assert_eq!(bytes, "café\n".as_bytes());
        struct Broken;
        impl Write for Broken {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                Err(io::Error::other("closed sink"))
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let mut broken = Broken;
        let mut observer = PromptObserver {
            writer: &mut broken,
            stream_text: true,
            requested_at: Instant::now(),
            first_token: true,
            first_visible: true,
        };
        assert!(
            observer
                .observe(GenerateEvent::Token {
                    token_id: 1,
                    text: "x".into()
                })
                .is_err()
        );
        assert!(observer.first_visible);
    }
}
