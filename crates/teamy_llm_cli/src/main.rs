use std::env;
use std::io::{self, Write};
use std::time::{Duration, Instant};

use color_eyre::eyre::{Result, bail, eyre};
use teamy_cancellation::{CancellationToken, CtrlCHandler, StopAfterArgs};
use teamy_llm_core::{GenerateEvent, GenerateRequest, ModelSelector, PrepareEvent, ThinkingMode};
use teamy_llm_service::{TeamyLlmService, TopTokensRequest, default_prepare_request};
use tracing_subscriber::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    let stop_after = stop_after_args_from(&mut args)?;
    let cancellation_token = CtrlCHandler::default().install()?;
    let _stop_after_duration_thread =
        stop_after.start_stop_after_duration_thread(cancellation_token.clone())?;
    tracing::subscriber::set_global_default({
        let subscriber = tracing_subscriber::registry()
            .with(
                tracing_subscriber::fmt::layer()
                    .with_filter(tracing_subscriber::EnvFilter::from_default_env()),
            )
            .with(stop_after.stop_after_span_layer(cancellation_token.clone()));
        #[cfg(feature = "tracy")]
        let subscriber = subscriber.with(
            tracing_tracy::TracyLayer::default()
                .with_filter(tracing_subscriber::filter::LevelFilter::TRACE),
        );
        subscriber
    })?;

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
        "top-tokens" => run_top_tokens(service, args)?,
        "top-completions" => run_top_completions(service, args)?,
        "benchmark" => run_benchmark(service, args, cancellation_token.clone())?,
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
    let thinking_mode = thinking_mode_from_args(&mut args)?;
    let max_new_tokens = take_option_value(&mut args, "--max-new-tokens")
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(match thinking_mode {
            ThinkingMode::Direct => 256,
            ThinkingMode::Thinking => 1024,
        });
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
        thinking_mode,
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

fn run_top_tokens(service: TeamyLlmService, mut args: Vec<String>) -> Result<()> {
    let request = top_tokens_request_from_args(&mut args)?;
    let report = service
        .inspect_top_tokens(request)
        .map_err(|error| eyre!(error.to_string()))?;
    println!(
        "top_tokens input_token_count={} candidate_count={}",
        report.input_token_count,
        report.top_tokens.len()
    );
    for (rank, token) in report.top_tokens.iter().enumerate() {
        println!(
            "rank={} token_id={} logit={:.6} text={:?}",
            rank + 1,
            token.token_id,
            token.logit,
            token.text
        );
    }
    Ok(())
}

fn run_top_completions(service: TeamyLlmService, mut args: Vec<String>) -> Result<()> {
    let max_continuation_tokens = take_option_value(&mut args, "--max-continuation-tokens")
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(12);
    if max_continuation_tokens == 0 {
        bail!("--max-continuation-tokens must be at least 1");
    }
    let request = top_tokens_request_from_args(&mut args)?;
    if !request.raw_prompt {
        bail!(
            "top-completions requires --raw-prompt so continuations extend the actual model prefix"
        );
    }
    let original_prompt = request.user_prompt.clone();
    let first_report = service
        .inspect_top_tokens(request.clone())
        .map_err(|error| eyre!(error.to_string()))?;
    println!(
        "top_completions input_token_count={} initial_candidate_count={} max_continuation_tokens={max_continuation_tokens}",
        first_report.input_token_count,
        first_report.top_tokens.len(),
    );
    let initial_prompt_token_ids = first_report.input_token_ids;
    for (rank, first_token) in first_report.top_tokens.into_iter().enumerate() {
        let mut model_token_ids = initial_prompt_token_ids.clone();
        let mut continuation_token_ids = vec![first_token.token_id];
        model_token_ids.push(first_token.token_id);
        let mut text = first_token.text;
        while !continuation_reaches_whitespace(&text)
            && continuation_token_ids.len() < max_continuation_tokens
        {
            let mut continuation_request = request.clone();
            continuation_request.user_prompt = original_prompt.clone();
            continuation_request.prompt_token_ids = Some(model_token_ids.clone());
            continuation_request.top_k = 1;
            let next_token = service
                .inspect_top_tokens(continuation_request)
                .map_err(|error| eyre!(error.to_string()))?
                .top_tokens
                .into_iter()
                .next()
                .ok_or_else(|| eyre!("top-token inspection returned no continuation candidate"))?;
            model_token_ids.push(next_token.token_id);
            continuation_token_ids.push(next_token.token_id);
            text.push_str(&next_token.text);
        }
        println!(
            "rank={} initial_logit={:.6} token_ids={continuation_token_ids:?} text={text:?} stopped_at_whitespace={}",
            rank + 1,
            first_token.logit,
            continuation_reaches_whitespace(&text),
        );
    }
    Ok(())
}

fn top_tokens_request_from_args(args: &mut Vec<String>) -> Result<TopTokensRequest> {
    let model = take_option_value(args, "--model");
    let model_dir = take_option_value(args, "--model-dir");
    let system_prompt = take_option_value(args, "--system-prompt");
    let thinking_mode = thinking_mode_from_args(args)?;
    let raw_prompt = take_flag(args, "--raw-prompt");
    let top_k = take_option_value(args, "--top-k")
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(5);
    if args.is_empty() {
        bail!("top-tokens command requires prompt text");
    }
    Ok(TopTokensRequest {
        selector: ModelSelector { model, model_dir },
        system_prompt,
        user_prompt: args.join(" "),
        thinking_mode,
        raw_prompt,
        prompt_token_ids: None,
        top_k,
    })
}

fn continuation_reaches_whitespace(text: &str) -> bool {
    text.chars().any(char::is_whitespace)
}

/// Measures a complete request from submission to final token.  `time_to_first_token_ms` therefore
/// includes CUDA validation, model initialization, prompt processing, and the first decode step.
/// `tokens_per_second` measures decode throughput after the first token arrives.
fn run_benchmark(
    service: TeamyLlmService,
    mut args: Vec<String>,
    cancellation_token: CancellationToken,
) -> Result<()> {
    let model = take_option_value(&mut args, "--model");
    let model_dir = take_option_value(&mut args, "--model-dir");
    let system_prompt = take_option_value(&mut args, "--system-prompt");
    let thinking_modes = benchmark_thinking_modes_from_args(&mut args)?;
    let max_new_tokens = take_option_value(&mut args, "--max-new-tokens")
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(200);
    let timeout_ms = take_option_value(&mut args, "--timeout-ms")
        .map(|value| value.parse::<u64>())
        .transpose()?;
    let user_prompt = if args.is_empty() {
        "Why is the sky blue?".to_owned()
    } else {
        args.join(" ")
    };

    for thinking_mode in thinking_modes {
        run_benchmark_case(
            &service,
            GenerateRequest {
                selector: ModelSelector {
                    model: model.clone(),
                    model_dir: model_dir.clone(),
                },
                system_prompt: system_prompt.clone(),
                user_prompt: user_prompt.clone(),
                thinking_mode,
                max_new_tokens,
                timeout_ms,
            },
            cancellation_token.clone(),
        )?;
    }
    Ok(())
}

fn run_benchmark_case(
    service: &TeamyLlmService,
    request: GenerateRequest,
    cancellation_token: CancellationToken,
) -> Result<()> {
    let started_at = Instant::now();
    let mut first_token_at = None;
    let mode = thinking_mode_name(request.thinking_mode);
    let result =
        service.generate_with_cancellation(request, Some(cancellation_token), &mut |event| {
            if matches!(event, GenerateEvent::Token { .. }) && first_token_at.is_none() {
                first_token_at = Some(Instant::now());
            }
            Ok(())
        });
    let completed_at = Instant::now();
    let time_to_first_token = first_token_at.map(|first_token_at| first_token_at - started_at);
    let elapsed = completed_at - started_at;

    match result {
        Ok(result) => {
            let token_count = result.generated_token_ids.len();
            let tokens_per_second = time_to_first_token
                .map(|first_token| tokens_per_second(token_count, elapsed - first_token));
            print_benchmark_result(
                mode,
                token_count,
                time_to_first_token,
                elapsed,
                tokens_per_second,
                "completed",
            );
            Ok(())
        }
        Err(error) => {
            print_benchmark_result(mode, 0, time_to_first_token, elapsed, None, "failed");
            bail!(error.to_string())
        }
    }
}

fn benchmark_thinking_modes_from_args(args: &mut Vec<String>) -> Result<Vec<ThinkingMode>> {
    let thinking = take_flag(args, "--thinking");
    let direct = take_flag(args, "--no-thinking");
    let both = take_flag(args, "--both-thinking-modes");
    if both && (thinking || direct) {
        bail!("`--both-thinking-modes` cannot be combined with `--thinking` or `--no-thinking`");
    }
    if thinking && direct {
        bail!("`--thinking` and `--no-thinking` cannot be used together");
    }
    Ok(if thinking {
        vec![ThinkingMode::Thinking]
    } else if direct {
        vec![ThinkingMode::Direct]
    } else {
        vec![ThinkingMode::Direct, ThinkingMode::Thinking]
    })
}

fn thinking_mode_name(mode: ThinkingMode) -> &'static str {
    match mode {
        ThinkingMode::Direct => "direct",
        ThinkingMode::Thinking => "thinking",
    }
}

fn tokens_per_second(token_count: usize, duration: Duration) -> f64 {
    token_count as f64 / duration.as_secs_f64()
}

fn print_benchmark_result(
    mode: &str,
    token_count: usize,
    time_to_first_token: Option<Duration>,
    time_to_completion: Duration,
    tokens_per_second: Option<f64>,
    status: &str,
) {
    let time_to_first_token_ms = time_to_first_token
        .map(|duration| duration.as_secs_f64() * 1_000.0)
        .unwrap_or(f64::NAN);
    let tokens_per_second = tokens_per_second.unwrap_or(f64::NAN);
    println!(
        "benchmark status={status} mode={mode} token_count={token_count} time_to_first_token_ms={time_to_first_token_ms:.3} time_to_completion_ms={:.3} tokens_per_second={tokens_per_second:.3}",
        time_to_completion.as_secs_f64() * 1_000.0,
    );
}

fn thinking_mode_from_args(args: &mut Vec<String>) -> Result<ThinkingMode> {
    let thinking = take_flag(args, "--thinking");
    let direct = take_flag(args, "--no-thinking");
    if thinking && direct {
        bail!("`--thinking` and `--no-thinking` cannot be used together");
    }
    Ok(if thinking {
        ThinkingMode::Thinking
    } else {
        ThinkingMode::Direct
    })
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
        "teamy-llm prompt [--stop-after <span-or-message>] [--stop-after-duration <duration>] [--model <name>|--model-dir <path>] [--system-prompt <text>] [--thinking|--no-thinking] [--max-new-tokens <n>] [--timeout-ms <ms>] <prompt>"
    );
    println!(
        "teamy-llm top-tokens [--model <name>|--model-dir <path>] [--system-prompt <text>] [--raw-prompt] [--thinking|--no-thinking] [--top-k <n>] <prompt>"
    );
    println!(
        "teamy-llm top-completions --raw-prompt [--model <name>|--model-dir <path>] [--thinking|--no-thinking] [--top-k <n>] [--max-continuation-tokens <n>] <prompt>"
    );
    println!(
        "teamy-llm benchmark [--model <name>|--model-dir <path>] [--system-prompt <text>] [--thinking|--no-thinking|--both-thinking-modes] [--max-new-tokens <n>] [--timeout-ms <ms>] [prompt]"
    );
}

#[cfg(test)]
mod tests {
    use super::{
        benchmark_thinking_modes_from_args, continuation_reaches_whitespace, stop_after_args_from,
        thinking_mode_from_args,
    };
    use std::time::Duration;
    use teamy_llm_core::ThinkingMode;

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

    #[test]
    fn defaults_to_direct_mode_and_accepts_explicit_thinking_mode() {
        let mut default_args = vec!["prompt".to_owned()];
        assert_eq!(
            thinking_mode_from_args(&mut default_args).expect("default thinking mode"),
            ThinkingMode::Direct
        );

        let mut thinking_args = vec!["--thinking".to_owned(), "prompt".to_owned()];
        assert_eq!(
            thinking_mode_from_args(&mut thinking_args).expect("thinking mode"),
            ThinkingMode::Thinking
        );
        assert_eq!(thinking_args, vec!["prompt"]);
    }

    #[test]
    fn benchmark_defaults_to_both_modes_and_can_select_one() {
        let mut default_args = Vec::new();
        assert_eq!(
            benchmark_thinking_modes_from_args(&mut default_args).expect("default modes"),
            vec![ThinkingMode::Direct, ThinkingMode::Thinking]
        );

        let mut thinking_args = vec!["--thinking".to_owned()];
        assert_eq!(
            benchmark_thinking_modes_from_args(&mut thinking_args).expect("thinking mode"),
            vec![ThinkingMode::Thinking]
        );
        assert!(thinking_args.is_empty());
    }

    #[test]
    fn top_completion_stops_after_a_decoded_whitespace_character() {
        assert!(!continuation_reaches_whitespace("Python"));
        assert!(continuation_reaches_whitespace("Python "));
        assert!(continuation_reaches_whitespace("Python\n"));
    }
}
