use crate::cli::{
    benchmark::BenchmarkCli, decide::DecideCli, health::HealthCli, interactive::InteractiveCli,
    model::ModelCli, prompt::PromptCli, rank::RankCli, serve::ServeCli,
    top_completions::TopCompletionsCli, top_tokens::TopTokensCli,
};
use facet::Facet;
#[derive(Debug, Facet)]
#[repr(u8)]
pub enum Command {
    /// Report service identity and backend availability without loading a model.
    Health(HealthCli),
    /// Serve local clients over Vox IPC.
    Serve(ServeCli),
    /// Inspect, prepare and select models.
    Model(ModelCli),
    /// Stream a generated answer from a local model.
    Prompt(PromptCli),
    /// Choose among versioned finite-choice options.
    Decide(DecideCli),
    /// Use resident models through an interactive terminal session.
    Interactive(InteractiveCli),
    /// Inspect next-token candidates from the legacy Burn backend.
    TopTokens(TopTokensCli),
    /// Inspect short candidate completions from the legacy Burn backend.
    TopCompletions(TopCompletionsCli),
    /// Rank candidate continuations with the legacy Burn backend.
    Rank(RankCli),
    /// Measure cold and resident generation requests.
    Benchmark(BenchmarkCli),
}
