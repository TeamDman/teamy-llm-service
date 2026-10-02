use crate::cli::output::{OutputFormat, emit_report};
use eyre::Result;
use facet::Facet;
use teamy_llm_service::TeamyLlmService;
/// Report service identity without loading weights.
#[derive(Debug, Facet)]
pub struct HealthCli;
impl HealthCli {
    pub fn invoke(self, service: &TeamyLlmService, output: Option<OutputFormat>) -> Result<()> {
        emit_report(&service.health(), output)
    }
}
