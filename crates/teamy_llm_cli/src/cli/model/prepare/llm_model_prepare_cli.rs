use crate::cli::output::{OutputFormat, emit_report};
use eyre::Result;
use facet::Facet;
use figue as args;
use teamy_llm_core::PrepareEvent;
use teamy_llm_service::{TeamyLlmService, default_prepare_request};
/// Acquire or prepare a managed model. Existing artifacts require --overwrite to replace.
#[derive(Debug, Facet)]
pub struct ModelPrepareCli {
    /// Managed model name. Omitted uses the existing preparation default.
    #[facet(args::positional)]
    pub model: Option<String>,
    /// Explicitly allow replacement of existing preparation artifacts.
    #[facet(args::named, default)]
    pub overwrite: bool,
    /// Include the upstream multimodal projector artifact.
    #[facet(args::named, default)]
    pub with_mmproj: bool,
    /// Export weights for the legacy Burn text backend.
    #[facet(args::named, default)]
    pub with_burn_text: bool,
    /// Export Burn text weights without downloading the main GGUF.
    #[facet(args::named, default)]
    pub burn_text_only: bool,
    /// Burn export datatype supported by the preparer.
    #[facet(args::named)]
    pub burn_text_dtype: Option<String>,
}
impl ModelPrepareCli {
    pub async fn invoke(
        self,
        service: &TeamyLlmService,
        output: Option<OutputFormat>,
    ) -> Result<()> {
        let mut request = default_prepare_request(self.model);
        request.overwrite = self.overwrite;
        request.include_mmproj = self.with_mmproj;
        request.export_burn_text = self.with_burn_text || self.burn_text_only;
        request.download_main_model_file = !self.burn_text_only;
        request.burn_text_dtype = self.burn_text_dtype;
        let result = service
            .prepare_model(request, |event| {
                match event {
                    PrepareEvent::Started { model } => tracing::info!(model, "preparing model"),
                    PrepareEvent::ModelPrepared { model_dir } => {
                        tracing::info!(model_dir, "model files ready")
                    }
                    PrepareEvent::BurnTextExportStarted => {
                        tracing::info!("exporting Burn text runtime")
                    }
                    PrepareEvent::BurnTextExportCompleted { message } => {
                        tracing::info!(message, "Burn export completed")
                    }
                    PrepareEvent::Completed { model_dir } => {
                        tracing::info!(model_dir, "preparation completed")
                    }
                    PrepareEvent::Failed { message } => {
                        tracing::error!(message, "preparation failed")
                    }
                }
                Ok(())
            })
            .await?;
        if let Some(warning) = &result.registration_warning {
            tracing::warn!(warning, "model registration warning");
        }
        emit_report(&result, output)
    }
}
