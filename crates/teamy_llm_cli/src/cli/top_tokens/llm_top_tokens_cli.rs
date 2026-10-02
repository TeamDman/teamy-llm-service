use crate::cli::{
    model_selection::ModelSelectionArgs,
    output::{OutputFormat, emit_report},
};
use eyre::{Result, ensure};
use facet::Facet;
use figue as args;
use teamy_llm_core::ThinkingMode;
use teamy_llm_service::{TeamyLlmService, TopTokensRequest};

/// Inspect next-token candidates from a legacy Burn text model.
#[derive(Debug, Clone, Facet)]
pub struct TopTokensCli {
    #[facet(flatten, default)]
    pub selection: ModelSelectionArgs,
    #[facet(args::named)]
    pub system_prompt: Option<String>,
    #[facet(args::named, default)]
    pub thinking: bool,
    /// Preserve prompt text exactly instead of adding the chat envelope.
    #[facet(args::named, default)]
    pub raw_prompt: bool,
    /// Positive number of candidates.
    #[facet(args::named, default = 5)]
    pub top_k: usize,
    #[facet(args::positional)]
    pub prompt: Vec<String>,
}
#[derive(Debug, Facet)]
#[facet(rename_all = "kebab-case")]
struct TopTokenRow {
    token_id: u32,
    text: String,
    logit: f32,
}
#[derive(Debug, Facet)]
#[facet(rename_all = "kebab-case")]
struct TopTokensOutput {
    version: u32,
    input_token_count: usize,
    input_token_ids: Vec<u32>,
    top_tokens: Vec<TopTokenRow>,
}
impl TopTokensCli {
    pub fn into_request(self) -> Result<TopTokensRequest> {
        ensure!(
            !self.prompt.is_empty(),
            "top-tokens command requires prompt text"
        );
        ensure!(self.top_k > 0, "--top-k must be at least 1");
        Ok(TopTokensRequest {
            selector: self.selection.to_selector()?,
            system_prompt: self.system_prompt,
            user_prompt: self.prompt.join(" "),
            thinking_mode: if self.thinking {
                ThinkingMode::Thinking
            } else {
                ThinkingMode::Direct
            },
            raw_prompt: self.raw_prompt,
            prompt_token_ids: None,
            top_k: self.top_k,
        })
    }
    pub fn invoke(self, service: &TeamyLlmService, output: Option<OutputFormat>) -> Result<()> {
        let report = service.inspect_top_tokens(self.into_request()?)?;
        emit_report(
            &TopTokensOutput {
                version: 1,
                input_token_count: report.input_token_count,
                input_token_ids: report.input_token_ids,
                top_tokens: report
                    .top_tokens
                    .into_iter()
                    .map(|row| TopTokenRow {
                        token_id: row.token_id,
                        text: row.text,
                        logit: row.logit,
                    })
                    .collect(),
            },
            output,
        )
    }
}
