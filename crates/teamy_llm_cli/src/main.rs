use std::env;
use std::io::{self, Write};

use color_eyre::eyre::{Result, bail, eyre};
use teamy_cancellation::{CancellationToken, CtrlCHandler, StopAfterArgs};
use teamy_llm_core::{GenerateEvent, GenerateRequest, ModelSelector, PrepareEvent};
use teamy_llm_service::{TeamyLlmService, default_prepare_request};
use tracing_subscriber::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    let stop_after = stop_after_args_from(&mut args)?;
    let cancellation_token = CtrlCHandler::default().install()?;
    let _stop_after_duration_thread =
        stop_after.start_stop_after_duration_thread(cancellation_token.clone())?;
    tracing::subscriber::set_global_default(
        tracing_subscriber::registry()
            .with(
                tracing_subscriber::fmt::layer()
                    .with_filter(tracing_subscriber::EnvFilter::from_default_env()),
            )
            .with(stop_after.stop_after_span_layer(cancellation_token.clone())),
    )?;

    if args.is_empty() {
        print_help();
        return Ok(());
    }

    let command = args.remove(0);
    let service = TeamyLlmService::from_default_dirs().map_err(|error| eyre!(error.to_string()))?;
    match command.as_str() {
        "health" => {
            let health = service.health();
            println!(
                "{} {} backend={} ready={}",
                health.service_name, health.version, health.backend, health.ready
            );
        }
        "serve" => {
            let address =
                take_option_value(&mut args, "--address").unwrap_or_else(default_local_address);
            serve(service, &address).await?;
        }
        "model" => run_model(service, args).await?,
        "prompt" => run_prompt(service, args, cancellation_token.clone())?,
        "help" | "--help" | "-h" => print_help(),
        other => bail!("unknown command `{other}`"),
    }
    cancellation_token.bail_if_cancelled()?;
    Ok(())
}

async fn run_model(service: TeamyLlmService, mut args: Vec<String>) -> Result<()> {
    if args.is_empty() {
        bail!("model command requires `list`, `show`, or `prepare`");
    }
    let command = args.remove(0);
    match command.as_str() {
        "list" => {
            for status in service
                .list_models()
                .map_err(|error| eyre!(error.to_string()))?
            {
                println!("{} {:?}", status.model_name, status.state);
                for location in status.locations {
                    println!(
                        "  {} {} exists={} compatible={}",
                        location.label, location.path, location.exists, location.compatible
                    );
                }
            }
        }
        "show" | "inspect" => {
            let selector = selector_from_args(&mut args);
            let report = service
                .inspect_model(selector)
                .map_err(|error| eyre!(error.to_string()))?;
            println!("{}", report.details);
        }
        "prepare" => {
            let mut request = default_prepare_request(
                args.first()
                    .filter(|value| !value.starts_with('-'))
                    .cloned(),
            );
            if args.first().is_some_and(|value| !value.starts_with('-')) {
                args.remove(0);
            }
            request.overwrite = take_flag(&mut args, "--overwrite");
            request.include_mmproj = take_flag(&mut args, "--with-mmproj");
            request.export_burn_text = take_flag(&mut args, "--with-burn-text");
            if take_flag(&mut args, "--burn-text-only") {
                request.download_main_model_file = false;
                request.export_burn_text = true;
            }
            request.burn_text_dtype = take_option_value(&mut args, "--burn-text-dtype");
            if !args.is_empty() {
                bail!("unknown model prepare arguments: {}", args.join(" "));
            }
            let result = service
                .prepare_model(request, |event| {
                    print_prepare_event(event);
                    Ok(())
                })
                .await
                .map_err(|error| eyre!(error.to_string()))?;
            println!("prepared: {}", result.model_dir);
            if let Some(warning) = result.registration_warning {
                eprintln!("warning: {warning}");
            }
        }
        other => bail!("unknown model command `{other}`"),
    }
    Ok(())
}

fn run_prompt(
    service: TeamyLlmService,
    mut args: Vec<String>,
    cancellation_token: CancellationToken,
) -> Result<()> {
    let model = take_option_value(&mut args, "--model");
    let model_dir = take_option_value(&mut args, "--model-dir");
    let system_prompt = take_option_value(&mut args, "--system-prompt");
    let max_new_tokens = take_option_value(&mut args, "--max-new-tokens")
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(256);
    let timeout_ms = take_option_value(&mut args, "--timeout-ms")
        .map(|value| value.parse::<u64>())
        .transpose()?;
    if args.is_empty() {
        bail!("prompt command requires prompt text");
    }
    let user_prompt = args.join(" ");
    let request = GenerateRequest {
        selector: ModelSelector { model, model_dir },
        system_prompt,
        user_prompt,
        max_new_tokens,
        timeout_ms,
    };
    let mut stdout = io::stdout();
    let _result = service
        .generate_with_cancellation(request, Some(cancellation_token), &mut |event| {
            match event {
                GenerateEvent::Token { text, .. } => {
                    print!("{text}");
                    stdout.flush().map_err(|error| {
                        teamy_llm_core::ServiceError::new(
                            teamy_llm_core::ServiceErrorKind::Internal,
                            error.to_string(),
                        )
                    })?;
                }
                GenerateEvent::Failed { message } => eprintln!("failed: {message}"),
                _ => {}
            }
            Ok(())
        })
        .map_err(|error| eyre!(error.to_string()))?;
    println!();
    Ok(())
}

fn stop_after_args_from(args: &mut Vec<String>) -> Result<StopAfterArgs> {
    Ok(StopAfterArgs {
        stop_after_span: take_required_option_value(args, "--stop-after")?,
        stop_after_duration: take_required_option_value(args, "--stop-after-duration")?,
    })
}

fn take_required_option_value(args: &mut Vec<String>, flag: &str) -> Result<Option<String>> {
    let Some(index) = args.iter().position(|arg| arg == flag) else {
        return Ok(None);
    };
    args.remove(index);
    args.get(index)
        .filter(|value| !value.starts_with('-'))
        .cloned()
        .inspect(|_| {
            args.remove(index);
        })
        .map(Some)
        .ok_or_else(|| eyre!("{flag} requires a value"))
}

fn selector_from_args(args: &mut Vec<String>) -> ModelSelector {
    let model_dir = take_option_value(args, "--model-dir");
    let model = take_option_value(args, "--model").or_else(|| {
        if args.first().is_some_and(|value| !value.starts_with('-')) {
            Some(args.remove(0))
        } else {
            None
        }
    });
    ModelSelector { model, model_dir }
}

async fn serve(service: TeamyLlmService, address: &str) -> Result<()> {
    println!("serving Teamy LLM over Vox local IPC at {address}");
    teamy_llm_service::vox_api::serve_local(address, service)
        .await
        .map_err(|error| eyre!(error.to_string()))
}

fn print_prepare_event(event: PrepareEvent) {
    match event {
        PrepareEvent::Started { model } => println!("preparing: {model}"),
        PrepareEvent::ModelPrepared { model_dir } => println!("model files ready: {model_dir}"),
        PrepareEvent::BurnTextExportStarted => println!("exporting Burn text runtime"),
        PrepareEvent::BurnTextExportCompleted { message } => println!("{message}"),
        PrepareEvent::Completed { model_dir } => println!("completed: {model_dir}"),
        PrepareEvent::Failed { message } => eprintln!("failed: {message}"),
    }
}

fn take_flag(args: &mut Vec<String>, flag: &str) -> bool {
    if let Some(index) = args.iter().position(|arg| arg == flag) {
        args.remove(index);
        true
    } else {
        false
    }
}

fn take_option_value(args: &mut Vec<String>, flag: &str) -> Option<String> {
    let index = args.iter().position(|arg| arg == flag)?;
    args.remove(index);
    if index < args.len() {
        Some(args.remove(index))
    } else {
        None
    }
}

fn default_local_address() -> String {
    "local://teamy-llm-service".to_owned()
}

fn print_help() {
    println!("global options: [--stop-after <span-or-message>] [--stop-after-duration <duration>]");
    println!("teamy-llm health");
    println!("teamy-llm serve [--address local://teamy-llm-service]");
    println!("teamy-llm model list");
    println!("teamy-llm model show [--model <name>|--model-dir <path>]");
    println!(
        "teamy-llm model prepare [model] [--overwrite] [--with-mmproj] [--with-burn-text] [--burn-text-only] [--burn-text-dtype <dtype>]"
    );
    println!(
        "teamy-llm prompt [--stop-after <span-or-message>] [--stop-after-duration <duration>] [--model <name>|--model-dir <path>] [--system-prompt <text>] [--max-new-tokens <n>] [--timeout-ms <ms>] <prompt>"
    );
}

#[cfg(test)]
mod tests {
    use super::stop_after_args_from;
    use std::time::Duration;

    #[test]
    fn parses_stop_after_options_without_consuming_the_prompt_command() {
        let mut args = vec![
            "prompt".to_owned(),
            "--stop-after".to_owned(),
            "LLM model weights loaded".to_owned(),
            "--stop-after-duration".to_owned(),
            "2s".to_owned(),
            "Why is the sky blue?".to_owned(),
        ];

        let stop_after = stop_after_args_from(&mut args).expect("stop-after args should parse");

        assert_eq!(
            stop_after.stop_after_span.as_deref(),
            Some("LLM model weights loaded")
        );
        assert_eq!(
            stop_after
                .stop_after_duration()
                .expect("duration should parse"),
            Some(Duration::from_secs(2))
        );
        assert_eq!(args, ["prompt", "Why is the sky blue?"]);
    }
}
