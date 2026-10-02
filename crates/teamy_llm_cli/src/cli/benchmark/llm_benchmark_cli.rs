use crate::cli::{
    model_selection::ModelSelectionArgs,
    output::{OutputFormat, emit_report},
};
use eyre::{Result, ensure};
use facet::Facet;
use figue as args;
use std::time::Instant;
use teamy_cancellation::CancellationToken;
use teamy_llm_core::{GenerateEvent, GenerateRequest, ThinkingMode};
use teamy_llm_service::TeamyLlmService;

/// Measure cold and resident text generation in one process.
///
/// First-token latency includes validation, loading, prefill and the first token
/// event. Emission rate includes callback overhead and UTF-8 buffering; it is
/// not kernel timing. Omitted thinking selection runs both direct and thinking.
#[derive(Debug, Facet)]
pub struct BenchmarkCli {
    #[facet(flatten, default)]
    pub selection: ModelSelectionArgs,
    #[facet(args::named)]
    pub system_prompt: Option<String>,
    /// Run only this mode; --no-thinking runs only direct mode.
    #[facet(args::named)]
    pub thinking: Option<bool>,
    /// Explicitly request the default pair of modes. Conflicts with --thinking/--no-thinking.
    #[facet(args::named, default)]
    pub both_thinking_modes: bool,
    #[facet(args::named, default = 200)]
    pub max_new_tokens: usize,
    #[facet(args::named)]
    pub timeout_ms: Option<u64>,
    #[facet(args::named, default = 1)]
    pub repeat: usize,
    /// Default: Why is the sky blue?
    #[facet(args::positional, default)]
    pub prompt: Vec<String>,
}
#[derive(Debug, Facet)]
#[facet(rename_all = "kebab-case")]
struct BenchmarkRow {
    mode: ThinkingMode,
    repetition: usize,
    token_count: usize,
    time_to_first_token_ms: Option<f64>,
    time_to_completion_ms: f64,
    tokens_per_second: Option<f64>,
}
#[derive(Debug, Facet)]
#[facet(rename_all = "kebab-case")]
struct BenchmarkOutput {
    version: u32,
    requests: Vec<BenchmarkRow>,
}
impl BenchmarkCli {
    pub fn thinking_modes(&self) -> Result<Vec<ThinkingMode>> {
        ensure!(
            !(self.both_thinking_modes && self.thinking.is_some()),
            "--both-thinking-modes cannot be combined with --thinking or --no-thinking"
        );
        Ok(match self.thinking {
            Some(true) => vec![ThinkingMode::Thinking],
            Some(false) => vec![ThinkingMode::Direct],
            None => vec![ThinkingMode::Direct, ThinkingMode::Thinking],
        })
    }
    pub fn invoke(
        self,
        service: &TeamyLlmService,
        output: Option<OutputFormat>,
        cancellation: CancellationToken,
    ) -> Result<()> {
        let selector = self.selection.to_selector()?;
        let modes = self.thinking_modes()?;
        ensure!(
            self.max_new_tokens > 0,
            "--max-new-tokens must be at least 1"
        );
        ensure!(
            self.timeout_ms != Some(0),
            "--timeout-ms must be at least 1"
        );
        ensure!(
            (1..=1000).contains(&self.repeat),
            "--repeat must be in 1..=1000"
        );
        let user_prompt = if self.prompt.is_empty() {
            "Why is the sky blue?".into()
        } else {
            self.prompt.join(" ")
        };
        let mut requests = Vec::with_capacity(modes.len() * self.repeat);
        for thinking_mode in modes {
            for repetition in 0..self.repeat {
                tracing::info!(repetition, mode = ?thinking_mode, "benchmark request");
                let request = GenerateRequest {
                    selector: selector.clone(),
                    system_prompt: self.system_prompt.clone(),
                    user_prompt: user_prompt.clone(),
                    thinking_mode,
                    max_new_tokens: self.max_new_tokens,
                    timeout_ms: self.timeout_ms,
                };
                let start = Instant::now();
                let mut first_token = None;
                let result = service.generate_with_cancellation(
                    request,
                    Some(cancellation.clone()),
                    &mut |event| {
                        if matches!(event, GenerateEvent::Token { .. }) && first_token.is_none() {
                            first_token = Some(Instant::now());
                        }
                        Ok(())
                    },
                )?;
                let elapsed = start.elapsed();
                let token_count = result.generated_token_ids.len();
                let first_token_elapsed = first_token.map(|time| time - start);
                let tokens_per_second = first_token_elapsed.and_then(|first| {
                    let decode = elapsed - first;
                    (token_count > 1 && !decode.is_zero())
                        .then(|| (token_count - 1) as f64 / decode.as_secs_f64())
                });
                requests.push(BenchmarkRow {
                    mode: thinking_mode,
                    repetition,
                    token_count,
                    time_to_first_token_ms: first_token_elapsed
                        .map(|duration| duration.as_secs_f64() * 1000.0),
                    time_to_completion_ms: elapsed.as_secs_f64() * 1000.0,
                    tokens_per_second,
                });
            }
        }
        emit_report(
            &BenchmarkOutput {
                version: 1,
                requests,
            },
            output,
        )
    }
}
