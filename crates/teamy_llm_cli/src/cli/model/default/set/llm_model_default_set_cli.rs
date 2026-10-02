use crate::cli::output::{OutputFormat, emit_report};
use eyre::Result;
use facet::Facet;
use figue as args;
use teamy_llm_service::TeamyLlmService;
/// Persist the named model as the default after validating its local inventory.
#[derive(Debug, Facet)]
pub struct ModelDefaultSetCli {
    /// Known managed model name, e.g. qwen3.8-27b-uncensored-q4-k-m.
    #[facet(args::positional)]
    pub model: String,
}
impl ModelDefaultSetCli {
    pub fn invoke(self, service: &TeamyLlmService, output: Option<OutputFormat>) -> Result<()> {
        emit_report(&service.set_default_model(&self.model)?, output)
    }
}
