use crate::cli::decide::{DecisionDevice, DecisionProvider};
use crate::cli::model_selection::ModelSelectionArgs;
use facet::Facet;
use figue as args;
use std::path::PathBuf;

/// Prompt and make finite choices repeatedly with resident local models.
///
/// The TUI requires a terminal. --script uses the same engine for a repeatable
/// non-terminal session. Generation and finite-choice models are distinct.
#[derive(Debug, Facet)]
pub struct InteractiveCli {
    #[facet(flatten, default)]
    pub selection: ModelSelectionArgs,
    /// Optional text-generation system prompt.
    #[facet(args::named)]
    pub system_prompt: Option<String>,
    /// Enable visible reasoning; --no-thinking selects direct generation.
    #[facet(args::named, default)]
    pub thinking: bool,
    /// Generation token limit; selected model and thinking mode determine the default.
    #[facet(args::named)]
    pub max_new_tokens: Option<usize>,
    /// Generation timeout excluding queue/model loading; must be positive.
    #[facet(args::named)]
    pub timeout_ms: Option<u64>,
    /// Orca formatted prompt plus output capacity; default 8192 tokens.
    #[facet(args::named)]
    pub context_tokens: Option<u32>,
    /// Explicit finite-choice backend: native or onnx.
    #[facet(args::named)]
    pub decision_provider: Option<DecisionProvider>,
    /// Existing Julia finite-choice model artifacts.
    #[facet(args::named)]
    pub decision_model_dir: Option<PathBuf>,
    /// Required ONNX runtime library for an ONNX decision provider.
    #[facet(args::named)]
    pub runtime_library: Option<PathBuf>,
    /// Requested finite-choice device; CUDA never silently falls back.
    #[facet(args::named, default = DecisionDevice::Cuda)]
    pub device: DecisionDevice,
    /// Explicitly permit mixed CPU/CUDA ONNX graph placement.
    #[facet(args::named, default)]
    pub allow_cpu_fallback: bool,
    /// Julia encoding limit, including options and question.
    #[facet(args::named, default = 1024)]
    pub max_tokens: usize,
    /// Julia input head-token budget.
    #[facet(args::named, default = 256)]
    pub head_tokens: usize,
    /// Run a repeatable session script through the interactive engine.
    #[facet(args::named, rename = "script")]
    pub script_file: Option<PathBuf>,
}
