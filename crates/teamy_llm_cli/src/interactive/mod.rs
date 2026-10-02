//! Persistent, sequential inference with a separate terminal owner.
mod engine;
mod protocol;
mod script;
pub(crate) mod terminal_output;
mod tui;
mod worker;

use crate::cli::{decide::DecideCli, interactive::InteractiveCli};
use eyre::{Result, ensure};
use teamy_cancellation::CancellationToken;
use teamy_llm_core::GenerateRequest;
use teamy_llm_service::TeamyLlmService;
pub(crate) use worker::run_worker;

pub(crate) async fn run_interactive(
    mut args: InteractiveCli,
    service: TeamyLlmService,
    cancellation: CancellationToken,
) -> Result<()> {
    args.selection.to_selector()?;
    ensure!(
        args.max_new_tokens != Some(0),
        "--max-new-tokens must be positive"
    );
    ensure!(args.timeout_ms != Some(0), "--timeout-ms must be positive");
    if args.decision_provider.is_some()
        || args.decision_model_dir.is_some()
        || args.runtime_library.is_some()
        || args.allow_cpu_fallback
    {
        decision_config(&args)?;
    }
    // Snapshot implicit selection once. The displayed model and the requests
    // stay aligned even if another process changes the preference meanwhile;
    // repeated thinking prompts do not reparse a tokenizer for default lookup.
    let model_label = if args.selection.model.is_none() && args.selection.model_dir.is_none() {
        let selected = service.default_model()?;
        if args.thinking && args.max_new_tokens.is_none() {
            args.max_new_tokens = Some(
                if selected.model_name == teamy_llm_service::ORCA_MODEL_NAME {
                    512
                } else {
                    1024
                },
            );
        }
        args.selection.model_dir = Some(selected.model_dir.into());
        Some(format!("{} (default)", selected.model_name))
    } else {
        None
    };
    if args.script_file.is_some() {
        script::run_script(args, service, cancellation).await
    } else {
        tui::run_tui(args, service, cancellation, model_label).await
    }
}

fn prompt_request(
    args: &InteractiveCli,
    service: &TeamyLlmService,
    text: String,
) -> Result<GenerateRequest> {
    crate::cli::prompt::generation_request(
        service,
        &args.selection,
        args.system_prompt.clone(),
        args.thinking,
        args.max_new_tokens,
        args.timeout_ms,
        text,
    )
}

fn decision_config(args: &InteractiveCli) -> Result<DecideCli> {
    DecideCli::for_interactive(args)
}

async fn wait_cancelled(cancellation: &CancellationToken) {
    while !cancellation.is_cancelled() {
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
}
