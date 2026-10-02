use eyre::Result;
use facet::Facet;
use figue as args;
use teamy_llm_service::TeamyLlmService;
/// Serve resident local models over Vox IPC.
#[derive(Debug, Facet)]
pub struct ServeCli {
    /// Local service address. This command keeps model runtimes resident.
    #[facet(args::named, default = String::from("local://teamy-llm-service"))]
    pub address: String,
}
impl ServeCli {
    pub async fn invoke(self, service: TeamyLlmService) -> Result<()> {
        tracing::info!(address = %self.address, "serving Teamy LLM over Vox local IPC");
        teamy_llm_service::vox_api::serve_local(&self.address, service).await?;
        Ok(())
    }
}
