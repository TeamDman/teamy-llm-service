use crate::cli::{
    output::{OutputFormat, emit_report},
    top_tokens::TopTokensCli,
};
use eyre::{Result, ensure, eyre};
use facet::Facet;
use figue as args;
use teamy_llm_service::TeamyLlmService;

/// Inspect bounded candidate completions from the legacy Burn backend.
#[derive(Debug, Facet)]
pub struct TopCompletionsCli {
    #[facet(flatten)]
    pub common: TopTokensCli,
    /// Continue through decoded whitespace instead of stopping at its first occurrence.
    #[facet(args::named, default)]
    pub full_continuation: bool,
    #[facet(args::named, default = 12)]
    pub max_continuation_tokens: usize,
}
#[derive(Debug, Facet)]
#[facet(rename_all = "kebab-case")]
struct Completion {
    initial_logit: f32,
    token_ids: Vec<u32>,
    text: String,
    stopped_at_whitespace: bool,
}
#[derive(Debug, Facet)]
#[facet(rename_all = "kebab-case")]
struct CompletionsOutput {
    version: u32,
    input_token_count: usize,
    max_continuation_tokens: usize,
    completions: Vec<Completion>,
}
fn continuation_reaches_whitespace(text: &str) -> bool {
    text.chars().any(char::is_whitespace)
}
impl TopCompletionsCli {
    pub fn invoke(self, service: &TeamyLlmService, output: Option<OutputFormat>) -> Result<()> {
        ensure!(
            self.max_continuation_tokens > 0,
            "--max-continuation-tokens must be at least 1"
        );
        let request = self.common.into_request()?;
        let first_report = service.inspect_top_tokens(request.clone())?;
        let input_token_count = first_report.input_token_count;
        let initial_prompt_token_ids = first_report.input_token_ids;
        let mut completions = Vec::with_capacity(first_report.top_tokens.len());
        for first_token in first_report.top_tokens {
            let mut model_token_ids = initial_prompt_token_ids.clone();
            let mut token_ids = vec![first_token.token_id];
            model_token_ids.push(first_token.token_id);
            let mut text = first_token.text;
            while (self.full_continuation || !continuation_reaches_whitespace(&text))
                && token_ids.len() < self.max_continuation_tokens
            {
                let mut continuation_request = request.clone();
                continuation_request.prompt_token_ids = Some(model_token_ids.clone());
                continuation_request.top_k = 1;
                let next = service
                    .inspect_top_tokens(continuation_request)?
                    .top_tokens
                    .into_iter()
                    .next()
                    .ok_or_else(|| {
                        eyre!("top-token inspection returned no continuation candidate")
                    })?;
                model_token_ids.push(next.token_id);
                token_ids.push(next.token_id);
                text.push_str(&next.text);
            }
            let stopped_at_whitespace = continuation_reaches_whitespace(&text);
            completions.push(Completion {
                initial_logit: first_token.logit,
                token_ids,
                text,
                stopped_at_whitespace,
            });
        }
        emit_report(
            &CompletionsOutput {
                version: 1,
                input_token_count,
                max_continuation_tokens: self.max_continuation_tokens,
                completions,
            },
            output,
        )
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn whitespace_ends_a_short_completion() {
        assert!(!continuation_reaches_whitespace("Python"));
        assert!(continuation_reaches_whitespace("Python "));
        assert!(continuation_reaches_whitespace("Python\n"));
    }
}
