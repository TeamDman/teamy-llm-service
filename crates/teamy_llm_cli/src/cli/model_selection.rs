use eyre::{Result, ensure};
use facet::Facet;
use figue as args;
use std::path::PathBuf;
use teamy_llm_core::ModelSelector;

/// An explicit model selection; omitted fields use the configured default.
#[derive(Debug, Default, Clone, Facet)]
pub struct ModelSelectionArgs {
    /// Managed model name. Cannot be combined with --model-dir.
    #[facet(args::named)]
    pub model: Option<String>,
    /// Existing model artifact directory. Cannot be combined with --model.
    #[facet(args::named)]
    pub model_dir: Option<PathBuf>,
}

impl ModelSelectionArgs {
    pub fn to_selector(&self) -> Result<ModelSelector> {
        ensure!(
            self.model.is_none() || self.model_dir.is_none(),
            "--model and --model-dir cannot be combined"
        );
        Ok(ModelSelector {
            model: self.model.clone(),
            model_dir: self
                .model_dir
                .as_ref()
                .map(|path| path.display().to_string()),
        })
    }
}
