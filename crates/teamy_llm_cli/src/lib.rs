//! Opinionated Teamy CLI startup with typed Figue commands and resident inference.
pub mod cli;
mod interactive;
mod logging;

use cli::{Cli, Command};
use eyre::{Result, WrapErr, eyre};
use std::env;
use teamy_cancellation::CtrlCHandler;
use teamy_llm_service::TeamyLlmService;

pub(crate) fn version() -> String {
    format!(
        "{} (repo {}, branch {}, rev {}, worktree {}, built unix {})",
        env!("CARGO_PKG_VERSION"),
        env!("CARGO_PKG_REPOSITORY"),
        option_env!("GIT_BRANCH").unwrap_or("unknown"),
        option_env!("GIT_REVISION").unwrap_or("unknown"),
        option_env!("GIT_WORKTREE_STATUS").unwrap_or("unknown"),
        option_env!("BUILD_TIMESTAMP_UNIX").unwrap_or("unknown")
    )
}

/// Process entrypoint. Help and version use Figue's success exit before runtime setup.
pub fn main() -> Result<()> {
    #[cfg(windows)]
    if env::var_os("CUBLAS_WORKSPACE_CONFIG").is_none() {
        // SAFETY: Windows environment mutation is thread-safe and precedes our workers.
        unsafe { env::set_var("CUBLAS_WORKSPACE_CONFIG", ":4096:8") };
    }
    color_eyre::install()?;
    let arguments = env::args_os()
        .skip(1)
        .map(|argument| {
            argument
                .into_string()
                .map_err(|_| eyre!("command-line arguments must be valid UTF-8"))
        })
        .collect::<Result<Vec<_>>>()?;
    // Figue d0 has no hidden-command attribute. Only this exact private bootstrap
    // vector enters the worker; every other invocation is checked by the typed CLI.
    if is_worker_invocation(&arguments) {
        let cancellation = CtrlCHandler::default().install()?;
        let options = logging::LoggingOptions::default();
        // The parent forwards only its resolved span/message selector. Elapsed
        // duration stays parent-owned so it includes terminal and worker startup.
        let worker_stop_after = teamy_cancellation::StopAfterArgs {
            stop_after_span: env::var("TEAMY_LLM_WORKER_STOP_AFTER").ok(),
            ..Default::default()
        };
        logging::install(&options, &worker_stop_after, cancellation.clone())?;
        let service = TeamyLlmService::from_default_dirs()?;
        runtime()?.block_on(interactive::run_worker(service, cancellation))?;
        interactive::terminal_output::check_human_output()?;
        return Ok(());
    }
    let cli: Cli = cli::parse(arguments).unwrap();
    cli.global_args.validate()?;
    let cancellation = CtrlCHandler::default().install()?;
    let _stop_after_duration_thread = cli
        .global_args
        .stop_after
        .start_stop_after_duration_thread(cancellation.clone())?;
    let logging_options = logging::LoggingOptions {
        debug: cli.global_args.debug,
        filter: cli.global_args.log_filter.clone(),
        file: cli.global_args.log_file.clone(),
        ansi: matches!(&cli.command, Command::Interactive(_)).then_some(false),
    };
    logging::install(
        &logging_options,
        &cli.global_args.stop_after,
        cancellation.clone(),
    )?;
    let requested_output = cli.global_args.output_format;
    runtime()?.block_on(async move {
        cancellation.bail_if_cancelled()?;
        if let Command::Decide(args) = cli.command {
            cli::decide::run(args, requested_output, cancellation.clone())?;
        } else {
            let service = TeamyLlmService::from_default_dirs()?;
            match cli.command {
                Command::Health(args) => args.invoke(&service, requested_output)?,
                Command::Serve(args) => args.invoke(service).await?,
                Command::Model(args) => args.invoke(&service, requested_output).await?,
                Command::Prompt(args) => {
                    args.invoke(&service, requested_output, cancellation.clone())?
                }
                Command::Interactive(args) => {
                    interactive::run_interactive(args, service, cancellation.clone()).await?
                }
                Command::TopTokens(args) => args.invoke(&service, requested_output)?,
                Command::TopCompletions(args) => args.invoke(&service, requested_output)?,
                Command::Rank(args) => args.invoke(&service, requested_output)?,
                Command::Benchmark(args) => {
                    args.invoke(&service, requested_output, cancellation.clone())?
                }
                Command::Decide(_) => {
                    unreachable!("decide was dispatched before service construction")
                }
            }
        }
        cancellation.bail_if_cancelled()?;
        interactive::terminal_output::check_human_output()?;
        Ok(())
    })
}
fn runtime() -> Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .wrap_err("cannot create CLI runtime")
}
fn is_worker_invocation(arguments: &[String]) -> bool {
    arguments == ["--internal-interactive-worker"]
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn private_worker_route_requires_the_exact_argument_vector() {
        assert!(is_worker_invocation(&[
            "--internal-interactive-worker".into()
        ]));
        assert!(!is_worker_invocation(&[
            "--internal-interactive-worker".into(),
            "extra".into()
        ]));
        assert!(cli::parse(["--internal-interactive-worker", "extra"]).is_err());
        assert!(cli::parse(["prompt", "--", "--internal-interactive-worker"]).is_ok());
    }
}
