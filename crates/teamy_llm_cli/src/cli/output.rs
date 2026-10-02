use eyre::{Result, WrapErr};
use facet::Facet;
use facet_pretty::{ColorMode, PrettyPrinter};
use std::io::{self, IsTerminal, Write};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Facet)]
#[repr(u8)]
#[facet(rename_all = "kebab-case")]
pub enum OutputFormat {
    #[default]
    Text,
    Json,
    Csv,
}
impl OutputFormat {
    pub fn resolved(requested: Option<Self>) -> Self {
        requested.unwrap_or(if io::stdout().is_terminal() {
            Self::Text
        } else {
            Self::Json
        })
    }
}
/// Emit strongly typed reports once. Progress and diagnostics use stderr independently.
pub fn emit_report<T: Facet<'static>>(report: &T, requested: Option<OutputFormat>) -> Result<()> {
    let terminal = io::stdout().is_terminal();
    let rendered = match OutputFormat::resolved(requested) {
        OutputFormat::Text => PrettyPrinter::new()
            .with_colors(if terminal {
                ColorMode::Always
            } else {
                ColorMode::Never
            })
            .format(report),
        OutputFormat::Json => {
            facet_json::to_string_pretty(report).wrap_err("cannot serialize JSON result")?
        }
        OutputFormat::Csv => {
            facet_csv::to_string(report).wrap_err("cannot serialize CSV result")?
        }
    };
    let mut stdout = io::stdout().lock();
    stdout.write_all(rendered.as_bytes())?;
    if !rendered.ends_with('\n') {
        stdout.write_all(b"\n")?;
    }
    stdout.flush()?;
    Ok(())
}
