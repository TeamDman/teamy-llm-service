use super::{
    default::ModelDefaultCli, list::ModelListCli, prepare::ModelPrepareCli, show::ModelShowCli,
};
use crate::cli::output::OutputFormat;
use eyre::Result;
use facet::Facet;
use figue as args;
use teamy_llm_service::TeamyLlmService;
/// Model inventory and persistent default selection.
#[derive(Debug, Facet)]
pub struct ModelCli {
    #[facet(args::subcommand)]
    pub command: ModelCommand,
}
#[derive(Debug, Facet)]
#[repr(u8)]
pub enum ModelCommand {
    /// List known models and mark the effective default.
    List(ModelListCli),
    /// Inspect model metadata and backend support.
    #[facet(args::alias = "inspect")]
    Show(ModelShowCli),
    /// Acquire or prepare a known model. Does not override an explicit default.
    Prepare(ModelPrepareCli),
    /// Set or show the persistent default model.
    Default(ModelDefaultCli),
}
impl ModelCli {
    pub async fn invoke(
        self,
        service: &TeamyLlmService,
        output: Option<OutputFormat>,
    ) -> Result<()> {
        match self.command {
            ModelCommand::List(args) => args.invoke(service, output),
            ModelCommand::Show(args) => args.invoke(service, output),
            ModelCommand::Prepare(args) => args.invoke(service, output).await,
            ModelCommand::Default(args) => args.invoke(service, output),
        }
    }
}
