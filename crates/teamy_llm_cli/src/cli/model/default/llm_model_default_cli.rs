use super::{set::ModelDefaultSetCli, show::ModelDefaultShowCli};
use crate::cli::output::OutputFormat;
use eyre::Result;
use facet::Facet;
use figue as args;
use teamy_llm_service::TeamyLlmService;
#[derive(Debug, Facet)]
pub struct ModelDefaultCli {
    #[facet(args::subcommand)]
    pub command: ModelDefaultCommand,
}
#[derive(Debug, Facet)]
#[repr(u8)]
pub enum ModelDefaultCommand {
    /// Persist a validated managed model as the default.
    Set(ModelDefaultSetCli),
    /// Show effective default and whether it is explicit, legacy or built in.
    Show(ModelDefaultShowCli),
}
impl ModelDefaultCli {
    pub fn invoke(self, service: &TeamyLlmService, output: Option<OutputFormat>) -> Result<()> {
        match self.command {
            ModelDefaultCommand::Set(args) => args.invoke(service, output),
            ModelDefaultCommand::Show(args) => args.invoke(service, output),
        }
    }
}
