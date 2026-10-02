use crate::cli::output::{OutputFormat, emit_report};
use eyre::Result;
use facet::Facet;
use teamy_llm_service::TeamyLlmService;
/// Show the effective default without loading model weights.
#[derive(Debug, Facet)]
pub struct ModelDefaultShowCli;
impl ModelDefaultShowCli {
    pub fn invoke(self, service: &TeamyLlmService, output: Option<OutputFormat>) -> Result<()> {
        emit_report(&service.default_model()?, output)
    }
}
