//! Repeatable data-driven sessions using the same resident worker as the TUI.
use super::{
    decision_config,
    engine::InteractiveEngine,
    prompt_request,
    protocol::{MAX_FRAME_BYTES, WorkerAction, WorkerEvent},
};
use crate::cli::interactive::InteractiveCli;
use eyre::{Result, WrapErr, ensure};
use facet::Facet;
use std::{
    io::{Read, Write},
    time::{Duration, Instant},
};
use teamy_cancellation::CancellationToken;
use teamy_llm_service::TeamyLlmService;

#[derive(Debug, Facet)]
#[facet(rename_all = "kebab-case", deny_unknown_fields)]
struct SessionScript {
    version: u32,
    actions: Vec<ScriptAction>,
}
#[derive(Debug, Facet)]
#[repr(u8)]
#[facet(rename_all = "kebab-case")]
enum ScriptAction {
    Prompt(PromptAction),
    Decide(DecideAction),
}
// Enum-level rename_all names variants, not their fields. Separate payload
// structs give each externally tagged action its own strict field boundary.
#[derive(Debug, Facet)]
#[facet(deny_unknown_fields)]
struct PromptAction {
    text: String,
    #[facet(rename = "cancel-after-ms", default)]
    cancel_after_ms: Option<u64>,
}
#[derive(Debug, Facet)]
#[facet(deny_unknown_fields)]
struct DecideAction {
    #[facet(rename = "request-json")]
    request_json: String,
    #[facet(rename = "cancel-after-ms", default)]
    cancel_after_ms: Option<u64>,
}
#[derive(Facet)]
#[facet(rename_all = "kebab-case")]
struct SessionOutput {
    version: u32,
    id: u64,
    mode: String,
    elapsed_ms: f64,
    worker_elapsed_ms: Option<f64>,
    first_token_ms: Option<f64>,
    result: Option<SessionResult>,
    error: Option<String>,
    cancelled: bool,
}

#[derive(Facet)]
#[repr(u8)]
#[facet(rename_all = "kebab-case")]
#[expect(
    dead_code,
    reason = "Variant fields are read through Facet reflection by the output serializer"
)]
enum SessionResult {
    Prompt {
        result: teamy_llm_core::GenerateResult,
    },
    Decide {
        result: Box<crate::cli::decide::DecideOutput>,
    },
}

fn decode_script(bytes: &[u8]) -> Result<SessionScript> {
    ensure!(
        bytes.len() <= MAX_FRAME_BYTES,
        "session script exceeds {MAX_FRAME_BYTES} bytes"
    );
    let script: SessionScript = facet_json::from_str(std::str::from_utf8(bytes)?)?;
    ensure!(
        script.version == 1,
        "unsupported session script version {}",
        script.version
    );
    ensure!(
        (1..=1000).contains(&script.actions.len()),
        "session must contain 1–1000 actions"
    );
    for action in &script.actions {
        let cancel_after_ms = match action {
            ScriptAction::Prompt(action) => action.cancel_after_ms,
            ScriptAction::Decide(action) => action.cancel_after_ms,
        };
        ensure!(
            cancel_after_ms.is_none_or(|delay| delay <= 86_400_000),
            "cancel-after-ms must not exceed one day"
        );
    }
    Ok(script)
}

pub(super) async fn run_script(
    args: InteractiveCli,
    service: TeamyLlmService,
    cancellation: CancellationToken,
) -> Result<()> {
    let path = args
        .script_file
        .as_ref()
        .expect("script dispatch requires a path");
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .wrap_err("cannot open interactive session script")?
        .take(MAX_FRAME_BYTES as u64 + 1)
        .read_to_end(&mut bytes)?;
    let script = decode_script(&bytes)?;
    // Validate all configuration and prompt limits before starting a child.
    for action in &script.actions {
        match action {
            ScriptAction::Prompt(action) => {
                prompt_request(&args, &service, action.text.clone())?;
            }
            ScriptAction::Decide(_) => {
                decision_config(&args)?;
            }
        }
    }
    let mut engine = InteractiveEngine::spawn(cancellation.clone()).await?;
    let execution = async {
        for action in script.actions {
            cancellation.bail_if_cancelled()?;
            let (command, mode, cancel_after_ms) = match action {
                ScriptAction::Prompt(action) => (WorkerAction::Prompt { request: prompt_request(&args, &service, action.text)? }, "prompt", action.cancel_after_ms),
                ScriptAction::Decide(action) => (WorkerAction::Decide { config: decision_config(&args)?, request_json: action.request_json }, "decide", action.cancel_after_ms),
            };
            let started = Instant::now();
            let id = engine.submit(command).await?;
            let mut first_token_ms = None;
            let mut cancel_sent = false;
            let deadline = tokio::time::sleep(Duration::from_millis(cancel_after_ms.unwrap_or(86_400_000)));
            tokio::pin!(deadline);
            let output = loop {
                tokio::select! {
                    _ = super::wait_cancelled(&cancellation) => {
                        engine.cancel(id).await?;
                        cancellation.bail_if_cancelled()?;
                    }
                    _ = &mut deadline, if cancel_after_ms.is_some() && !cancel_sent => {
                        engine.cancel(id).await?;
                        cancel_sent = true;
                    }
                    reply = engine.next_event() => {
                        match reply?.event {
                            WorkerEvent::Token { .. } => { first_token_ms.get_or_insert_with(|| started.elapsed().as_secs_f64() * 1000.0); }
                            WorkerEvent::Completed { result_json, elapsed_ms } => {
                                let result = if mode == "prompt" { SessionResult::Prompt { result: facet_json::from_str(&result_json)? } } else { SessionResult::Decide { result: Box::new(facet_json::from_str(&result_json)?) } };
                                break SessionOutput { version: 1, id, mode: mode.into(), elapsed_ms: started.elapsed().as_secs_f64() * 1000.0, worker_elapsed_ms: Some(elapsed_ms), first_token_ms, result: Some(result), error: None, cancelled: false };
                            }
                            WorkerEvent::Failed { message, cancelled } => break SessionOutput { version: 1, id, mode: mode.into(), elapsed_ms: started.elapsed().as_secs_f64() * 1000.0, worker_elapsed_ms: None, first_token_ms, result: None, error: Some(message), cancelled },
                            _ => {}
                        }
                    }
                }
            };
            writeln!(std::io::stdout(), "{}", facet_json::to_string(&output)?)?;
            std::io::stdout().flush()?;
        }
        Ok::<(), eyre::Report>(())
    }.await;
    let shutdown = engine.shutdown().await;
    match (execution, shutdown) {
        (Ok(()), result) | (result, Ok(())) => result,
        (Err(error), Err(cleanup)) => {
            Err(error.wrap_err(format!("resident worker shutdown also failed: {cleanup:#}")))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn script_is_versioned_strict_and_bounded() {
        let script = SessionScript {
            version: 1,
            actions: vec![ScriptAction::Prompt(PromptAction {
                text: "café".into(),
                cancel_after_ms: None,
            })],
        };
        let bytes = facet_json::to_string(&script).unwrap();
        assert_eq!(decode_script(bytes.as_bytes()).unwrap().actions.len(), 1);
        assert!(decode_script(br#"{"version":2,"actions":[]}"#).is_err());
        assert!(decode_script(br#"{"version":1,"actions":[],"execute":"anything"}"#).is_err());
        assert!(decode_script(&vec![b' '; MAX_FRAME_BYTES + 1]).is_err());
    }

    #[test]
    fn literal_public_action_fields_preserve_cancellation_and_decision_json() {
        let script = decode_script(
            br#"{
                "version": 1,
                "actions": [
                    {"prompt": {"text": "fixture", "cancel-after-ms": 5}},
                    {"decide": {"request-json": "{\"version\":1}", "cancel-after-ms": 0}}
                ]
            }"#,
        )
        .unwrap();
        let ScriptAction::Prompt(prompt) = &script.actions[0] else {
            panic!("expected prompt action")
        };
        assert_eq!(prompt.text, "fixture");
        assert_eq!(prompt.cancel_after_ms, Some(5));
        let ScriptAction::Decide(decide) = &script.actions[1] else {
            panic!("expected decide action")
        };
        assert_eq!(decide.request_json, r#"{"version":1}"#);
        assert_eq!(decide.cancel_after_ms, Some(0));
        let encoded = facet_json::to_string(&script).unwrap();
        assert!(encoded.contains(r#""cancel-after-ms":5"#));
        assert!(encoded.contains(r#""request-json":"#));
        assert!(!encoded.contains("cancel_after_ms"));
        assert!(!encoded.contains("request_json"));
    }

    #[test]
    fn unknown_or_misspelled_action_payload_fields_are_rejected() {
        // Optional cancellation must never turn a typo into a silently uncancelled job.
        for action in [
            r#"{"prompt":{"text":"fixture","cancel_after_ms":5}}"#,
            r#"{"prompt":{"text":"fixture","cancel-after-m":5}}"#,
            r#"{"prompt":{"text":"fixture","extra":true}}"#,
            r#"{"decide":{"request-json":"{}","cancel_after_ms":5}}"#,
            r#"{"decide":{"request-json":"{}","request_json":"{}"}}"#,
            r#"{"decide":{"request-json":"{}","extra":true}}"#,
            r#"{"prompt":{"text":"fixture"},"extra":true}"#,
            r#"{"unknown":{"text":"fixture"}}"#,
        ] {
            let script = format!(r#"{{"version":1,"actions":[{action}]}}"#);
            assert!(
                decode_script(script.as_bytes()).is_err(),
                "accepted {action}"
            );
        }
    }

    #[test]
    fn cancellation_limit_is_applied_to_the_public_kebab_case_field() {
        for action in [
            r#"{"prompt":{"text":"fixture","cancel-after-ms":86400001}}"#,
            r#"{"decide":{"request-json":"{}","cancel-after-ms":86400001}}"#,
            r#"{"prompt":{"text":"fixture","cancel-after-ms":-1}}"#,
        ] {
            let script = format!(r#"{{"version":1,"actions":[{action}]}}"#);
            assert!(
                decode_script(script.as_bytes()).is_err(),
                "accepted {action}"
            );
        }
    }

    #[test]
    fn five_action_resident_fixture_decodes_the_fourth_action_deadline() {
        // A literal copy of the public grammar reproduces the missed cancellation
        // observed after three successful resident prompts, without requiring models.
        let script = decode_script(
            r#"{
                "version": 1,
                "actions": [
                    {"prompt": {"text": "Why is the sky blue?"}},
                    {"prompt": {"text": "Why is the sky blue?"}},
                    {"prompt": {"text": "Why is the sky blue?"}},
                    {"prompt": {"text": "Tell me a long story about the ocean.", "cancel-after-ms": 5}},
                    {"prompt": {"text": "Explain café in one sentence."}}
                ]
            }"#
            .as_bytes(),
        )
        .unwrap();
        let deadlines = script
            .actions
            .into_iter()
            .map(|action| match action {
                ScriptAction::Prompt(action) => action.cancel_after_ms,
                ScriptAction::Decide(_) => panic!("fixture contains only prompts"),
            })
            .collect::<Vec<_>>();
        assert_eq!(deadlines, [None, None, None, Some(5), None]);
    }
}
