use crate::cli::output::{OutputFormat, emit_report};
use eyre::Result;
use facet::Facet;
use std::io::{self, Write};
use teamy_llm_core::ModelStatus;
use teamy_llm_service::TeamyLlmService;
/// List model identity, preparation state, locations and effective default.
#[derive(Debug, Facet)]
pub struct ModelListCli;
#[derive(Debug, Facet)]
#[facet(rename_all = "kebab-case")]
struct InventoryModel {
    #[facet(flatten)]
    status: ModelStatus,
    is_default: bool,
}
impl ModelListCli {
    pub fn invoke(self, service: &TeamyLlmService, output: Option<OutputFormat>) -> Result<()> {
        let selected = service.default_model()?;
        let rows = service
            .list_models()?
            .into_iter()
            .map(|status| {
                let is_default = status
                    .locations
                    .iter()
                    .any(|location| location.path == selected.model_dir);
                InventoryModel { status, is_default }
            })
            .collect::<Vec<_>>();
        if OutputFormat::resolved(output) != OutputFormat::Text {
            return emit_report(&rows, output);
        }
        let mut stdout = io::stdout().lock();
        writeln!(stdout, "{:<44} {:<24} DEFAULT", "MODEL", "STATE")?;
        for row in rows {
            writeln!(
                stdout,
                "{:<44} {:<24} {}",
                row.status.model_name,
                format!("{:?}", row.status.state),
                if row.is_default { "(default)" } else { "" }
            )?;
            for location in row.status.locations {
                writeln!(
                    stdout,
                    "  {} {} exists={} compatible={}",
                    location.label, location.path, location.exists, location.compatible
                )?;
            }
        }
        stdout.flush()?;
        Ok(())
    }
}
