use crate::cli::{
    model_selection::ModelSelectionArgs,
    output::{OutputFormat, emit_report},
};
use eyre::{Result, ensure};
use facet::Facet;
use figue as args;
use teamy_llm_core::ThinkingMode;
use teamy_llm_service::{RankContinuationsRequest, TeamyLlmService};
/// Rank candidate continuations by conditional likelihood in the legacy Burn backend.
#[derive(Debug, Facet)]
pub struct RankCli {
    #[facet(flatten, default)]
    pub selection: ModelSelectionArgs,
    #[facet(args::named)]
    pub system_prompt: Option<String>,
    #[facet(args::named, default)]
    pub thinking: bool,
    #[facet(args::named, default)]
    pub raw_prompt: bool,
    /// Repeat --candidate for each continuation to score.
    #[facet(args::named, rename = "candidate", default)]
    pub candidates: Vec<String>,
    #[facet(args::positional)]
    pub prompt: Vec<String>,
}
#[derive(Debug, Facet)]
#[facet(rename_all = "kebab-case")]
struct RankedCandidate {
    candidate: String,
    token_count: usize,
    total_log_probability: f64,
    mean_log_probability: f64,
    mean_negative_log_likelihood: f64,
    perplexity: f64,
}
#[derive(Debug, Facet)]
#[facet(rename_all = "kebab-case")]
struct RankingsOutput {
    version: u32,
    prompt_token_count: usize,
    rankings: Vec<RankedCandidate>,
}
impl RankCli {
    pub fn invoke(self, service: &TeamyLlmService, output: Option<OutputFormat>) -> Result<()> {
        ensure!(!self.prompt.is_empty(), "rank command requires prompt text");
        ensure!(
            !self.candidates.is_empty(),
            "rank command requires at least one --candidate"
        );
        let report = service.rank_continuations(RankContinuationsRequest {
            selector: self.selection.to_selector()?,
            system_prompt: self.system_prompt,
            user_prompt: self.prompt.join(" "),
            thinking_mode: if self.thinking {
                ThinkingMode::Thinking
            } else {
                ThinkingMode::Direct
            },
            raw_prompt: self.raw_prompt,
            candidates: self.candidates,
        })?;
        emit_report(
            &RankingsOutput {
                version: 1,
                prompt_token_count: report.prompt_token_count,
                rankings: report
                    .rankings
                    .into_iter()
                    .map(|row| RankedCandidate {
                        candidate: row.candidate,
                        token_count: row.token_count,
                        total_log_probability: row.total_log_probability,
                        mean_log_probability: row.mean_log_probability,
                        mean_negative_log_likelihood: row.mean_negative_log_likelihood,
                        perplexity: row.perplexity,
                    })
                    .collect(),
            },
            output,
        )
    }
}
