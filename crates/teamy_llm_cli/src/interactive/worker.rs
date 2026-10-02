//! Resident inference runs in a captured child; this process never owns the terminal.
//! One blocking job owns every runtime, and cancellation never starts overlapping
//! GPU work. The parent restarts this child on a mode change: dropping runtime
//! references alone cannot guarantee release of process-global CUDA memory pools.
use std::io::{self, BufReader, Read, Write};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use eyre::{Result, WrapErr, eyre};
use teamy_cancellation::CancellationToken;
use teamy_llm_core::{GenerateEvent, ServiceError, ServiceErrorKind};
use teamy_llm_service::TeamyLlmService;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use super::protocol::{
    PROTOCOL_VERSION, WorkerAction, WorkerCommand, WorkerEvent, WorkerReply, encode_reply,
    parse_command, read_frame,
};
use crate::cli::decide::{DecideCli, DecisionEngine};

pub async fn run_worker(service: TeamyLlmService, cancellation: CancellationToken) -> Result<()> {
    run_worker_with_io(service, cancellation, io::stdin(), io::stdout()).await
}

struct ResidentDecision {
    config: DecideCli,
    engine: DecisionEngine,
    // Native/ONNX providers retain this token when first loaded. It must remain
    // the same across successful requests; cancellation discards the provider.
    cancellation: CancellationToken,
}

struct RuntimeState {
    prompt: Option<TeamyLlmService>,
    decision: Option<ResidentDecision>,
}
impl RuntimeState {
    fn new(service: TeamyLlmService) -> Self {
        Self {
            prompt: Some(service),
            decision: None,
        }
    }

    fn request_token(&self, action: &WorkerAction) -> CancellationToken {
        if let WorkerAction::Decide { config, .. } = action
            && let Some(resident) = &self.decision
            && resident.config == *config
            && !resident.cancellation.is_cancelled()
        {
            return resident.cancellation.clone();
        }
        CancellationToken::new()
    }
}

struct JobFinished {
    state: RuntimeState,
    result: Result<String>,
    cancelled: bool,
    elapsed_ms: f64,
}

struct ActiveJob {
    id: u64,
    cancellation: CancellationToken,
    task: JoinHandle<JobFinished>,
}
impl ActiveJob {
    /// Stale cancellation cannot affect a different request, including the next job.
    fn cancel_matching(&self, request_id: u64) -> bool {
        if self.id != request_id {
            return false;
        }
        self.cancellation
            .request_cancel("interactive request cancelled");
        true
    }
}

enum Input {
    Command(Box<WorkerCommand>),
    Failed(eyre::Report),
}

struct ReplyWriter<W>(Arc<Mutex<W>>);
impl<W> Clone for ReplyWriter<W> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<W: Write> ReplyWriter<W> {
    fn send(&self, id: u64, event: WorkerEvent) -> Result<()> {
        // Prepare the entire bounded frame before taking the lock: token and
        // control writers can never interleave JSON bytes on the shared pipe.
        let bytes = encode_reply(&WorkerReply {
            version: PROTOCOL_VERSION,
            id,
            event,
        })?;
        let mut writer = self
            .0
            .lock()
            .map_err(|_| eyre!("worker stdout lock poisoned"))?;
        writer
            .write_all(&bytes)
            .wrap_err("cannot write worker reply")?;
        writer.flush().wrap_err("cannot flush worker reply")
    }
}

fn spawn_reader<R: Read + Send + 'static>(reader: R) -> Result<mpsc::Receiver<Input>> {
    // Backpressure bounds pending input to one frame plus the reader's one frame.
    // A blocking pipe read never blocks the controller's cancellation handling.
    let (sender, receiver) = mpsc::channel(1);
    std::thread::Builder::new()
        .name("llm-worker-input".into())
        .spawn(move || {
            let mut reader = BufReader::new(reader);
            loop {
                let input = match read_frame(&mut reader) {
                    Ok(Some(bytes)) => match parse_command(&bytes) {
                        Ok(command) => Input::Command(Box::new(command)),
                        Err(error) => Input::Failed(error),
                    },
                    Ok(None) => break,
                    Err(error) => Input::Failed(error),
                };
                let failed = matches!(input, Input::Failed(_));
                if sender.blocking_send(input).is_err() || failed {
                    break;
                }
            }
        })
        .wrap_err("cannot start worker input reader")?;
    Ok(receiver)
}

async fn run_worker_with_io<R, W>(
    service: TeamyLlmService,
    cancellation: CancellationToken,
    reader: R,
    writer: W,
) -> Result<()>
where
    R: Read + Send + 'static,
    W: Write + Send + 'static,
{
    let mut input = spawn_reader(reader)?;
    let replies = ReplyWriter(Arc::new(Mutex::new(writer)));
    replies.send(0, WorkerEvent::Ready)?;
    let mut state = Some(RuntimeState::new(service));
    let mut active: Option<ActiveJob> = None;
    let mut stopping = false;
    let mut terminal_error: Option<eyre::Report> = None;
    let mut poll = tokio::time::interval(Duration::from_millis(20));
    poll.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        if stopping && active.is_none() {
            break;
        }
        tokio::select! {
            finished = async { (&mut active.as_mut().expect("active job").task).await }, if active.is_some() => {
                let job = active.take().expect("finished job");
                match finished {
                    Ok(finished) => {
                        state = Some(finished.state);
                        let event = match finished.result {
                            Ok(result_json) => WorkerEvent::Completed { result_json, elapsed_ms: finished.elapsed_ms },
                            Err(error) => WorkerEvent::Failed { message: bounded_message(&format!("{error:#}")), cancelled: finished.cancelled },
                        };
                        if let Err(error) = replies.send(job.id, event) {
                            // A very large successful result still gets a terminal,
                            // bounded failure. Broken pipes stop this worker.
                            let fallback = replies.send(job.id, WorkerEvent::Failed {
                                message: bounded_message(&format!("cannot deliver inference result: {error:#}")),
                                cancelled: finished.cancelled,
                            });
                            if let Err(failure) = fallback { stop(&mut stopping, &mut terminal_error, None, Some(failure)); }
                        }
                    }
                    Err(error) => {
                        stop(&mut stopping, &mut terminal_error, None, Some(eyre!(error).wrap_err("resident inference task failed")));
                    }
                }
            }
            incoming = input.recv(), if !stopping => {
                match incoming {
                    Some(Input::Command(command)) => match command.command {
                        WorkerAction::Cancel { request_id } => {
                            if active.as_ref().is_none_or(|job| !job.cancel_matching(request_id)) {
                                tracing::trace!(request_id, "ignored stale worker cancellation");
                            }
                        }
                        WorkerAction::Shutdown => stop(&mut stopping, &mut terminal_error, active.as_ref(), None),
                        action @ (WorkerAction::Prompt { .. } | WorkerAction::Decide { .. }) => {
                            if let Some(job) = &active {
                                if command.id == job.id {
                                    stop(&mut stopping, &mut terminal_error, active.as_ref(), Some(eyre!("duplicate active worker request ID {}", command.id)));
                                } else if let Err(error) = replies.send(command.id, WorkerEvent::Failed { message: "an inference request is already active".into(), cancelled: false }) {
                                    stop(&mut stopping, &mut terminal_error, active.as_ref(), Some(error));
                                }
                                continue;
                            }
                            let runtime = state.take().expect("idle worker owns runtime state");
                            let token = runtime.request_token(&action);
                            let mode = if matches!(action, WorkerAction::Prompt { .. }) { "prompt" } else { "decide" };
                            if let Err(error) = replies.send(command.id, WorkerEvent::Started { mode: mode.into() }) {
                                state = Some(runtime);
                                stop(&mut stopping, &mut terminal_error, None, Some(error));
                                continue;
                            }
                            let writer = replies.clone();
                            let job_token = token.clone();
                            let id = command.id;
                            let task = tokio::task::spawn_blocking(move || run_job(runtime, id, action, job_token, writer));
                            active = Some(ActiveJob { id, cancellation: token, task });
                        }
                    },
                    Some(Input::Failed(error)) => stop(&mut stopping, &mut terminal_error, active.as_ref(), Some(error)),
                    None => stop(&mut stopping, &mut terminal_error, active.as_ref(), None),
                }
            }
            _ = poll.tick() => {
                if cancellation.is_cancelled() {
                    let reason = cancellation.cancellation_reason().unwrap_or_else(|| "worker cancelled".into());
                    stop(&mut stopping, &mut terminal_error, active.as_ref(), Some(eyre!("{reason}")));
                }
            }
        }
    }
    // Drop idle runtime references before returning. Process exit is the release
    // boundary for retained CUDA pools. The stdin thread belongs to this child;
    // it can remain blocked until process exit closes it.
    drop(state);
    if let Some(error) = terminal_error {
        return Err(error);
    }
    Ok(())
}

fn stop(
    stopping: &mut bool,
    terminal_error: &mut Option<eyre::Report>,
    active: Option<&ActiveJob>,
    error: Option<eyre::Report>,
) {
    *stopping = true;
    if terminal_error.is_none() {
        *terminal_error = error;
    }
    if let Some(job) = active {
        job.cancellation.request_cancel("resident worker stopping");
    }
}

#[tracing::instrument(level = "info", skip_all, fields(request_id = id))]
fn run_job<W: Write>(
    mut state: RuntimeState,
    id: u64,
    action: WorkerAction,
    cancellation: CancellationToken,
    replies: ReplyWriter<W>,
) -> JobFinished {
    let started = Instant::now();
    let result: Result<String> = (|| match action {
        WorkerAction::Prompt { request } => {
            // No service clone escapes this job. Drop opposite runtime references
            // defensively; the parent restarts the process for actual mode changes.
            state.decision = None;
            cancellation.bail_if_cancelled()?;
            if state.prompt.is_none() {
                state.prompt = Some(TeamyLlmService::from_default_dirs()?);
            }
            let service = state.prompt.as_ref().expect("prompt service");
            let mut sink = |event: GenerateEvent| -> std::result::Result<(), ServiceError> {
                if let GenerateEvent::Token { text, .. } = event {
                    replies
                        .send(id, WorkerEvent::Token { text })
                        .map_err(|error| {
                            ServiceError::new(
                                ServiceErrorKind::Internal,
                                format!("worker token output failed: {error:#}"),
                            )
                        })?;
                }
                Ok(())
            };
            let result = service.generate_with_cancellation(
                request,
                Some(cancellation.clone()),
                &mut sink,
            )?;
            Ok(facet_json::to_string(&result)?)
        }
        WorkerAction::Decide {
            config,
            request_json,
        } => {
            // Drop Qwen runtime references before constructing a Julia engine.
            // The parent restarts on mode changes to release global CUDA pools.
            state.prompt = None;
            cancellation.bail_if_cancelled()?;
            if state.decision.as_ref().is_none_or(|resident| {
                resident.config != config || resident.cancellation.is_cancelled()
            }) {
                state.decision = None;
                let engine = DecisionEngine::new(config.clone())?;
                state.decision = Some(ResidentDecision {
                    config,
                    engine,
                    cancellation: cancellation.clone(),
                });
            }
            let result = state
                .decision
                .as_mut()
                .expect("decision engine")
                .engine
                .evaluate_json(request_json.as_bytes(), cancellation.clone())?;
            Ok(facet_json::to_string(&result)?)
        }
        WorkerAction::Cancel { .. } | WorkerAction::Shutdown => {
            Err(eyre!("control frame cannot execute as an inference job"))
        }
    })();
    let cancelled = cancellation.is_cancelled()
        || result
            .as_ref()
            .err()
            .and_then(|error| error.downcast_ref::<ServiceError>())
            .is_some_and(|error| error.kind == ServiceErrorKind::Cancelled);
    if cancelled {
        state.decision = None;
    }
    let result = if cancellation.is_cancelled() {
        Err(eyre!(
            "{}",
            cancellation
                .cancellation_reason()
                .unwrap_or_else(|| "interactive request cancelled".into())
        ))
    } else {
        result
    };
    JobFinished {
        state,
        result,
        cancelled,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
    }
}

fn bounded_message(message: &str) -> String {
    const LIMIT: usize = 8 * 1024;
    if message.len() <= LIMIT {
        return message.into();
    }
    let mut end = LIMIT;
    while !message.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}… (truncated)", &message[..end])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use teamy_llm_core::{GenerateRequest, ModelSelector, ThinkingMode};

    #[derive(Clone, Default)]
    struct Output(Arc<Mutex<Vec<u8>>>);
    impl Write for Output {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    fn service() -> TeamyLlmService {
        // Resolving these directory values does not create directories, select
        // a default, inspect models or load artifacts. Invalid input below is
        // rejected before the service could use any of those locations.
        TeamyLlmService::from_default_dirs().unwrap()
    }
    fn invalid_prompt() -> WorkerAction {
        WorkerAction::Prompt {
            request: GenerateRequest {
                selector: ModelSelector::managed("nonexistent-fixture-model"),
                system_prompt: None,
                user_prompt: "fixture".into(),
                thinking_mode: ThinkingMode::Direct,
                max_new_tokens: 0,
                timeout_ms: None,
            },
        }
    }

    #[test]
    fn invalid_job_is_a_recoverable_failure_without_artifact_reads() {
        let state = RuntimeState::new(service());
        let writer = ReplyWriter(Arc::new(Mutex::new(Output::default())));
        let finished = run_job(
            state,
            1,
            invalid_prompt(),
            CancellationToken::new(),
            writer.clone(),
        );
        assert!(
            finished
                .result
                .unwrap_err()
                .to_string()
                .contains("greater than zero")
        );
        assert!(!finished.cancelled);
        assert!(finished.state.prompt.is_some());
        assert!(finished.state.decision.is_none());
        let finished = run_job(
            finished.state,
            2,
            invalid_prompt(),
            CancellationToken::new(),
            writer,
        );
        assert!(finished.result.is_err());
        assert!(!finished.cancelled);
    }

    #[tokio::test]
    async fn cancellation_matches_only_the_current_job_and_shutdown_joins() {
        let token = CancellationToken::new();
        let state = RuntimeState::new(service());
        let task_token = token.clone();
        let task = tokio::task::spawn_blocking(move || {
            while !task_token.is_cancelled() {
                std::thread::sleep(Duration::from_millis(1));
            }
            JobFinished {
                state,
                result: Err(eyre!("cancelled fixture")),
                cancelled: true,
                elapsed_ms: 0.0,
            }
        });
        let job = ActiveJob {
            id: 9,
            cancellation: token.clone(),
            task,
        };
        assert!(!job.cancel_matching(8));
        assert!(!token.is_cancelled());
        assert!(job.cancel_matching(9));
        assert!(token.is_cancelled());
        assert!(
            tokio::time::timeout(Duration::from_secs(1), job.task)
                .await
                .unwrap()
                .unwrap()
                .cancelled
        );
    }

    #[tokio::test]
    async fn shutdown_and_bad_protocol_are_model_free() {
        let command = WorkerCommand {
            version: 1,
            id: 0,
            command: WorkerAction::Shutdown,
        };
        let bytes = super::super::protocol::encode_command(&command).unwrap();
        let output = Output::default();
        run_worker_with_io(
            service(),
            CancellationToken::new(),
            Cursor::new(bytes),
            output.clone(),
        )
        .await
        .unwrap();
        let bytes = output.0.lock().unwrap().clone();
        let ready = super::super::protocol::parse_reply(&bytes).unwrap();
        assert_eq!(ready.id, 0);
        assert!(matches!(ready.event, WorkerEvent::Ready));
        let result = run_worker_with_io(
            service(),
            CancellationToken::new(),
            Cursor::new(b"{\"version\":2}\n"),
            Output::default(),
        )
        .await;
        assert!(result.is_err());
    }

    #[test]
    fn failure_messages_are_bounded_at_a_unicode_boundary() {
        let message = bounded_message(&"💠".repeat(16 * 1024));
        assert!(message.len() <= 8 * 1024 + 32);
        assert!(message.ends_with("(truncated)"));
    }

    #[test]
    fn decision_cache_reuses_its_token_then_discards_it_after_cancellation() {
        let cli = crate::cli::parse([
            "decide",
            "--provider",
            "native",
            "--model-dir",
            "deliberately-missing-artifacts",
            "--model",
            "julia-1",
        ])
        .into_result()
        .unwrap();
        let crate::cli::Command::Decide(config) = cli.value.command else {
            panic!("decide command")
        };
        let action = WorkerAction::Decide {
            config,
            request_json: "{\"version\":2}".into(),
        };
        let token = CancellationToken::new();
        let replies = ReplyWriter(Arc::new(Mutex::new(Output::default())));
        // The invalid request stops before tokenization or numerical artifact reads.
        let finished = run_job(
            RuntimeState::new(service()),
            1,
            action.clone(),
            token.clone(),
            replies.clone(),
        );
        assert!(finished.result.is_err());
        assert!(finished.state.prompt.is_none());
        assert!(finished.state.decision.is_some());
        let reused = finished.state.request_token(&action);
        token.request_cancel("fixture cancellation");
        assert!(reused.is_cancelled());
        let cancelled = run_job(finished.state, 2, action.clone(), reused, replies);
        assert!(cancelled.cancelled);
        assert!(cancelled.state.decision.is_none());
        assert!(!cancelled.state.request_token(&action).is_cancelled());
    }

    struct PipeInput {
        receiver: std::sync::mpsc::Receiver<Vec<u8>>,
        pending: Cursor<Vec<u8>>,
    }
    impl Read for PipeInput {
        fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
            if bytes.is_empty() {
                return Ok(0);
            }
            loop {
                let count = self.pending.read(bytes)?;
                if count != 0 {
                    return Ok(count);
                }
                match self.receiver.recv() {
                    Ok(next) => self.pending = Cursor::new(next),
                    Err(_) => return Ok(0),
                }
            }
        }
    }

    #[tokio::test]
    async fn worker_recovers_after_invalid_requests_and_ignores_stale_cancel() {
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        let reader = PipeInput {
            receiver,
            pending: Cursor::new(Vec::new()),
        };
        let output = Output::default();
        let worker_output = output.clone();
        let worker = tokio::spawn(async move {
            run_worker_with_io(service(), CancellationToken::new(), reader, worker_output).await
        });
        async fn wait_failed(output: &Output, id: u64) -> WorkerReply {
            tokio::time::timeout(Duration::from_secs(2), async {
                loop {
                    let bytes = output.0.lock().unwrap().clone();
                    for frame in bytes
                        .split(|byte| *byte == b'\n')
                        .filter(|frame| !frame.is_empty())
                    {
                        let reply = super::super::protocol::parse_reply(frame).unwrap();
                        if reply.id == id && matches!(reply.event, WorkerEvent::Failed { .. }) {
                            return reply;
                        }
                    }
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
            })
            .await
            .unwrap()
        }
        fn send(sender: &std::sync::mpsc::SyncSender<Vec<u8>>, id: u64, command: WorkerAction) {
            sender
                .send(
                    super::super::protocol::encode_command(&WorkerCommand {
                        version: 1,
                        id,
                        command,
                    })
                    .unwrap(),
                )
                .unwrap();
        }
        send(&sender, 1, invalid_prompt());
        let first = wait_failed(&output, 1).await;
        assert!(matches!(
            first.event,
            WorkerEvent::Failed {
                cancelled: false,
                ..
            }
        ));
        send(&sender, 0, WorkerAction::Cancel { request_id: 1 });
        send(&sender, 2, invalid_prompt());
        let second = wait_failed(&output, 2).await;
        assert!(matches!(
            second.event,
            WorkerEvent::Failed {
                cancelled: false,
                ..
            }
        ));
        send(&sender, 0, WorkerAction::Shutdown);
        assert!(
            tokio::time::timeout(Duration::from_secs(2), worker)
                .await
                .unwrap()
                .unwrap()
                .is_ok()
        );
        let bytes = output.0.lock().unwrap().clone();
        let events: Vec<_> = bytes
            .split(|byte| *byte == b'\n')
            .filter(|frame| !frame.is_empty())
            .map(|frame| super::super::protocol::parse_reply(frame).unwrap())
            .collect();
        assert!(matches!(events[0].event, WorkerEvent::Ready));
        assert_eq!(events.len(), 5); // Ready, then Started + Failed for each request.
    }
}
