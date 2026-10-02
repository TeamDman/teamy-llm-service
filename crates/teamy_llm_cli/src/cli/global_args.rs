use super::output::OutputFormat;
use facet::Facet;
use figue as args;
use std::path::PathBuf;
use teamy_cancellation::StopAfterArgs;

/// Options accepted before or after any subcommand.
#[derive(Debug, Default, Facet)]
pub struct GlobalArgs {
    /// Enable project debug diagnostics; overrides inherited RUST_LOG.
    #[facet(args::named, default)]
    pub debug: bool,
    /// Tracing filter directive, also accepted as --log-level.
    #[facet(args::named, args::alias = "log-level")]
    pub log_filter: Option<String>,
    /// Append structured diagnostics to this NDJSON file.
    #[facet(args::named)]
    pub log_file: Option<PathBuf>,
    /// Cooperatively cancel after a span/message or elapsed duration.
    #[facet(flatten, default)]
    pub stop_after: StopAfterArgs,
    /// Select text, JSON or CSV results. Non-streaming reports default to JSON when redirected.
    #[facet(args::named)]
    pub output_format: Option<OutputFormat>,
}

impl GlobalArgs {
    pub fn validate(&self) -> eyre::Result<()> {
        eyre::ensure!(
            !(self.debug && self.log_filter.is_some()),
            "cannot specify a log filter with --debug"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::cli;

    #[test]
    fn explicit_filter_conflicts_with_debug_before_logging_is_installed() {
        for alias in ["--log-filter", "--log-level"] {
            let cli = cli::parse(["--debug", alias, "trace", "model", "default", "show"])
                .into_result()
                .unwrap();
            assert!(
                cli.value
                    .global_args
                    .validate()
                    .unwrap_err()
                    .to_string()
                    .contains("cannot specify a log filter with --debug")
            );
        }
        assert!(
            cli::parse(["--debug", "model", "default", "show"])
                .into_result()
                .unwrap()
                .value
                .global_args
                .validate()
                .is_ok()
        );
        assert!(
            cli::parse(["--log-level", "trace", "model", "default", "show"])
                .into_result()
                .unwrap()
                .value
                .global_args
                .validate()
                .is_ok()
        );
        // Builtins terminate successfully before normal command validation or file effects.
        assert!(
            cli::parse(["--debug", "--log-level", "trace", "--help"])
                .unwrap_err()
                .is_help()
        );
        assert!(matches!(
            cli::parse(["--debug", "--log-level", "trace", "--version"]).unwrap_err(),
            figue::DriverError::Version { .. }
        ));
    }
}
