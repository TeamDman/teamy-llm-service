use crate::cli::{
    model_selection::ModelSelectionArgs,
    output::{OutputFormat, emit_report},
};
use eyre::{Result, ensure};
use facet::Facet;
use figue as args;
use teamy_llm_service::TeamyLlmService;
/// Inspect metadata without loading model weights.
#[derive(Debug, Facet)]
pub struct ModelShowCli {
    #[facet(flatten, default)]
    pub selection: ModelSelectionArgs,
    /// Optional managed model positional alias for --model.
    #[facet(args::positional)]
    pub name: Option<String>,
}
impl ModelShowCli {
    pub fn invoke(mut self, service: &TeamyLlmService, output: Option<OutputFormat>) -> Result<()> {
        if let Some(name) = self.name {
            ensure!(
                self.selection.model.is_none() && self.selection.model_dir.is_none(),
                "positional model cannot be combined with --model or --model-dir"
            );
            self.selection.model = Some(name);
        }
        let report = service.inspect_model(self.selection.to_selector()?)?;
        emit_report(&report, output)
    }
}
