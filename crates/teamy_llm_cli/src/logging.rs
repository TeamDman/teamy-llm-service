//! Template-derived diagnostics: stderr plus optional NDJSON, independent Tracy filtering.
use std::fs::OpenOptions;
use std::io::{self, IsTerminal};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

#[cfg(test)]
use eyre::ensure;
use eyre::{Result, bail};
use teamy_cancellation::{CancellationToken, StopAfterArgs};
use tracing_subscriber::filter::FilterExt;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::fmt::writer::BoxMakeWriter;
use tracing_subscriber::prelude::*;
use tracing_subscriber::{EnvFilter, Registry};

#[derive(Debug, Default)]
pub struct LoggingOptions {
    pub debug: bool,
    pub filter: Option<String>,
    pub file: Option<PathBuf>,
}

impl LoggingOptions {
    #[cfg(test)]
    pub fn extract(args: &mut Vec<String>) -> Result<Self> {
        let mut result = Self::default();
        let mut cursor = 0;
        while cursor < args.len() {
            if args[cursor] == "--" {
                break;
            }
            match args[cursor].as_str() {
                "--debug" => {
                    ensure!(!result.debug, "--debug was specified more than once");
                    result.debug = true;
                    args.remove(cursor);
                }
                "--log-filter" | "--log-level" | "--log-file" => {
                    let flag = args.remove(cursor);
                    let value = args
                        .get(cursor)
                        .filter(|v| !v.starts_with("--"))
                        .ok_or_else(|| eyre::eyre!("{flag} requires a value"))?
                        .clone();
                    args.remove(cursor);
                    if flag == "--log-file" {
                        ensure!(
                            result.file.is_none(),
                            "--log-file was specified more than once"
                        );
                        result.file = Some(PathBuf::from(value));
                    } else {
                        ensure!(
                            result.filter.is_none(),
                            "only one log filter may be specified"
                        );
                        result.filter = Some(value);
                    }
                }
                _ => cursor += 1,
            }
        }
        ensure!(
            !(result.debug && result.filter.is_some()),
            "cannot specify a log filter with --debug"
        );
        Ok(result)
    }

    fn filter(&self) -> Result<EnvFilter> {
        let inherited = std::env::var("RUST_LOG").ok();
        let value = self.filter_directive(inherited.as_deref());
        Ok(EnvFilter::try_new(value)?)
    }

    pub(crate) fn filter_directive(&self, inherited: Option<&str>) -> String {
        if let Some(explicit) = &self.filter {
            explicit.clone()
        } else if self.debug {
            "warn,teamy_llm=debug".into()
        } else {
            inherited.unwrap_or("warn,teamy_llm=info").into()
        }
    }
}

pub fn install(
    options: &LoggingOptions,
    stop_after: &StopAfterArgs,
    cancellation: CancellationToken,
) -> Result<()> {
    let keep_event = || {
        tracing_subscriber::filter::filter_fn(|meta| {
            meta.fields().field("tracy.frame_mark").is_none()
        })
    };
    let stderr = tracing_subscriber::fmt::layer()
        .with_writer(crate::interactive::terminal_output::human_writer())
        .log_internal_errors(false)
        .with_ansi(io::stderr().is_terminal())
        .with_span_events(FmtSpan::CLOSE)
        .with_filter(options.filter()?.and(keep_event()));
    let file_layer = if let Some(path) = &options.file {
        if path.is_dir() {
            bail!("--log-file requires an explicit file path in this CLI");
        }
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        // Retain an existing capture; an explicit log destination is never truncated.
        let file = Mutex::new(OpenOptions::new().create(true).append(true).open(path)?);
        let writer = BoxMakeWriter::new(file);
        Some(
            tracing_subscriber::fmt::layer()
                .json()
                .with_ansi(false)
                .with_span_events(FmtSpan::CLOSE)
                .with_writer(writer)
                .with_filter(options.filter()?.and(keep_event())),
        )
    } else {
        None
    };
    let subscriber = Registry::default()
        .with(stderr)
        .with(file_layer)
        .with(stop_after.stop_after_span_layer(cancellation));
    #[cfg(feature = "tracy")]
    let subscriber = subscriber.with(tracing_tracy::TracyLayer::default().with_filter(
        EnvFilter::try_new("trace,cranelift_codegen=warn,cubecl=info")?,
    ));
    tracing::subscriber::set_global_default(subscriber)?;
    // The child captures native stderr as well as tracing. Only the parent
    // opens an optional NDJSON destination; no environment mutation is needed.
    let _ = WORKER_FILTER.set(options.filter_directive(std::env::var("RUST_LOG").ok().as_deref()));
    let _ = WORKER_STOP_AFTER.set(stop_after.stop_after_span.clone());
    tracing::debug!(debug = options.debug, "diagnostic logging initialized");
    Ok(())
}

static WORKER_FILTER: OnceLock<String> = OnceLock::new();
pub(crate) fn worker_filter() -> Option<&'static str> {
    WORKER_FILTER.get().map(String::as_str)
}
static WORKER_STOP_AFTER: OnceLock<Option<String>> = OnceLock::new();
pub(crate) fn worker_stop_after() -> Option<&'static str> {
    WORKER_STOP_AFTER.get().and_then(Option::as_deref)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_cli_filters_override_inherited_rust_log() {
        let debug = LoggingOptions {
            debug: true,
            ..LoggingOptions::default()
        };
        assert_eq!(debug.filter_directive(Some("warn")), "warn,teamy_llm=debug");
        let explicit = LoggingOptions {
            filter: Some("trace".into()),
            ..LoggingOptions::default()
        };
        assert_eq!(explicit.filter_directive(Some("off")), "trace");
        assert_eq!(
            LoggingOptions::default().filter_directive(Some("warn")),
            "warn"
        );
    }
    #[test]
    fn aliases_are_explicit_and_do_not_consume_literal_prompt_arguments() {
        let mut args = vec!["prompt", "--log-level", "trace", "--", "--debug"]
            .into_iter()
            .map(String::from)
            .collect();
        let parsed = LoggingOptions::extract(&mut args).unwrap();
        assert_eq!(parsed.filter.as_deref(), Some("trace"));
        assert_eq!(args, ["prompt", "--", "--debug"]);
    }
    #[test]
    fn malformed_or_conflicting_filters_fail() {
        for case in [
            vec!["--log-filter"],
            vec!["--debug", "--log-level", "trace"],
            vec!["--log-level", "info", "--log-filter", "debug"],
        ] {
            let mut args = case.into_iter().map(String::from).collect();
            assert!(LoggingOptions::extract(&mut args).is_err());
        }
    }
}
