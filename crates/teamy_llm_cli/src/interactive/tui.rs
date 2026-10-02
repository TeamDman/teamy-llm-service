//! A single-owner terminal view over the resident worker engine.
//! The parent draws on stderr; model work and native diagnostics stay in the
//! owned child. Prompt turns are independent rather than implicit chat history.

use std::collections::VecDeque;
use std::io::{self, BufWriter, IsTerminal, Stderr, Write};
use std::panic::{AssertUnwindSafe, PanicHookInfo};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crossterm::cursor::{Hide, Show};
use crossterm::event::{
    DisableBracketedPaste, EnableBracketedPaste, Event, EventStream, KeyCode, KeyEvent,
    KeyEventKind, KeyModifiers,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use eyre::{Result, bail, ensure};
use futures_util::{FutureExt, StreamExt};
use ratatui::backend::{Backend, CrosstermBackend};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::Line;
use ratatui::widgets::{Block, Paragraph};
use ratatui::{Frame, Terminal};
use teamy_cancellation::CancellationToken;
use teamy_llm_service::TeamyLlmService;
use tracing_subscriber::fmt::MakeWriter;

use super::engine::InteractiveEngine;
use super::protocol::{
    PROTOCOL_VERSION, WorkerAction, WorkerCommand, WorkerEvent, WorkerReply, encode_command,
};
use super::terminal_output::{
    TerminalLease, human_writer, sanitize_for_terminal_display, snapshot_logs,
};
use super::{decision_config, prompt_request};
use crate::cli::decide::DecideOutput;
use crate::cli::interactive::InteractiveCli;

const MAX_INPUT_BYTES: usize = 2 * 1024 * 1024;
const MAX_HISTORY_BYTES: usize = 1024 * 1024;
const MAX_HISTORY_ENTRIES: usize = 256;
const MAX_ENTRY_BYTES: usize = 64 * 1024;
const DISPLAY_BYTES: usize = 16 * 1024;
const REFRESH_INTERVAL: Duration = Duration::from_millis(50);
const DECISION_EXAMPLE: &str = r#"{
  "version": 1,
  "input": {"text": "Health is low and a potion is available.", "visuals": []},
  "options": [
    {"id": "heal", "content": {"text": "Use the potion.", "visuals": []}},
    {"id": "wait", "content": {"text": "Keep waiting.", "visuals": []}}
  ]
}"#;

type UiTerminal = Terminal<CrosstermBackend<BufWriter<Stderr>>>;

pub(super) async fn run_tui(
    args: InteractiveCli,
    service: TeamyLlmService,
    cancellation: CancellationToken,
    model_label: Option<String>,
) -> Result<()> {
    ensure!(
        io::stdin().is_terminal() && io::stderr().is_terminal(),
        "interactive requires terminal stdin and stderr; use --script for a non-terminal session"
    );
    cancellation.bail_if_cancelled()?;
    let model = if let Some(label) = model_label {
        label
    } else if let Some(model) = &args.selection.model {
        model.clone()
    } else if let Some(path) = &args.selection.model_dir {
        format!("directory: {}", path.display())
    } else {
        format!("{} (default)", service.default_model()?.model_name)
    };

    // No loaded inference runtime is created through this metadata-only service.
    // Declare the hook first so dropped futures destroy the worker and terminal
    // before restoring the previous hook's direct-stderr behavior.
    let mut panic_hook = ScopedPanicHook::install();
    let mut terminal = None;
    let mut engine = None;
    // Protect setup as well as the view, so its panic hook is restored outside
    // unwinding. Never allow the previous direct-stderr hook to draw over a TUI.
    let outcome = AssertUnwindSafe(async {
        terminal = Some(TerminalOwner::enter()?);
        engine = Some(InteractiveEngine::spawn(cancellation.clone()).await?);
        run_loop(
            &args,
            &service,
            &cancellation,
            engine
                .as_mut()
                .ok_or_else(|| eyre::eyre!("worker was not initialized"))?,
            terminal
                .as_mut()
                .ok_or_else(|| eyre::eyre!("terminal was not initialized"))?,
            model,
        )
        .await
    })
    .catch_unwind()
    .await;

    // Cleanup stages have their own catch boundaries: a worker cleanup panic
    // must not skip raw-mode restoration or leave the scoped hook installed.
    let shutdown = AssertUnwindSafe(async {
        match engine.as_mut() {
            Some(engine) => engine.shutdown().await,
            None => Ok(()),
        }
    })
    .catch_unwind()
    .await;
    let worker_drop = std::panic::catch_unwind(AssertUnwindSafe(|| drop(engine.take())));
    let restoration = std::panic::catch_unwind(AssertUnwindSafe(|| match terminal.as_mut() {
        Some(terminal) => terminal.restore(),
        None => Ok(()),
    }));
    // The fallback destructor can retry unsuccessful mechanical restoration.
    // Keep its diagnostics under this hook too, and destroy every terminal
    // field before the original panic reporter regains access to stderr.
    let terminal_drop = std::panic::catch_unwind(AssertUnwindSafe(|| drop(terminal.take())));
    panic_hook.restore();

    let mut panic_payload = None;
    let result = match outcome {
        Ok(result) => result,
        Err(payload) => {
            panic_payload = Some(payload);
            Err(eyre::eyre!("interactive view panicked"))
        }
    };
    let shutdown = match shutdown {
        Ok(result) => result,
        Err(payload) => {
            panic_payload.get_or_insert(payload);
            Err(eyre::eyre!("resident worker shutdown panicked"))
        }
    };
    if let Err(payload) = worker_drop {
        panic_payload.get_or_insert(payload);
    }
    let mut restoration = match restoration {
        Ok(result) => result,
        Err(payload) => {
            panic_payload.get_or_insert(payload);
            Err(eyre::eyre!("terminal restoration panicked"))
        }
    };
    if let Err(payload) = terminal_drop {
        panic_payload.get_or_insert(payload);
        restoration = Err(eyre::eyre!("terminal cleanup destructor panicked"));
    }
    if let Some(payload) = panic_payload {
        let notice = panic_hook.notice();
        if restoration.is_err() {
            // Keep a clear nonzero error without invoking an uncontrolled panic
            // reporter against a terminal whose restoration failed.
            return combine_cleanup(
                Err(eyre::eyre!("interactive panic: {notice}")),
                shutdown,
                restoration,
            );
        }
        let mut writer = human_writer().make_writer();
        writeln!(
            writer,
            "Interactive panic after terminal restoration:\n{notice}"
        )?;
        if let Err(error) = shutdown {
            writeln!(
                writer,
                "Worker shutdown also failed: {}",
                sanitize_for_terminal_display(&format!("{error:#}"))
            )?;
        }
        drop(writer);
        std::panic::resume_unwind(payload);
    }
    combine_cleanup(result, shutdown, restoration)
}

type PanicHook = dyn Fn(&PanicHookInfo<'_>) + Send + Sync + 'static;

struct ScopedPanicHook {
    previous: Option<Arc<PanicHook>>,
    active: Arc<AtomicBool>,
    first_notice: Arc<Mutex<Option<String>>>,
}

impl ScopedPanicHook {
    fn install() -> Self {
        let previous: Arc<PanicHook> = Arc::from(std::panic::take_hook());
        let active = Arc::new(AtomicBool::new(true));
        let first_notice = Arc::new(Mutex::new(None));
        let prior = previous.clone();
        let is_active = active.clone();
        let notice_slot = first_notice.clone();
        std::panic::set_hook(Box::new(move |info| {
            if !is_active.load(Ordering::Acquire) {
                prior(info);
                return;
            }
            // Formatting streams into a fixed-size writer; info.to_string()
            // could allocate an arbitrarily large user-controlled panic payload.
            let mut capture = BoundedPanicMessage::default();
            let _ = write!(capture, "{info}");
            let notice = sanitize_for_terminal_display(&capture.into_string());
            let mut first = notice_slot
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if first.is_none() {
                *first = Some(notice.clone());
            }
            drop(first);
            let mut writer = human_writer().make_writer();
            let _ = writeln!(
                writer,
                "Interactive panic buffered until terminal restoration:\n{notice}"
            );
        }));
        Self {
            previous: Some(previous),
            active,
            first_notice,
        }
    }

    fn notice(&self) -> String {
        self.first_notice
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
            .unwrap_or_else(|| "panic description unavailable".into())
    }

    fn restore(&mut self) {
        self.active.store(false, Ordering::Release);
        if !std::thread::panicking()
            && let Some(previous) = self.previous.take()
        {
            std::panic::set_hook(Box::new(move |info| previous(info)));
        }
        // During exceptional Drop while unwinding, set_hook is illegal. The
        // inactive installed wrapper delegates directly to the previous hook,
        // restoring its behavior without silencing later non-TUI panics.
    }
}

impl Drop for ScopedPanicHook {
    fn drop(&mut self) {
        self.restore();
    }
}

#[derive(Default)]
struct BoundedPanicMessage {
    bytes: Vec<u8>,
    truncated: bool,
}

impl Write for BoundedPanicMessage {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        const LIMIT: usize = 4_096;
        let accepted = bytes.len().min(LIMIT.saturating_sub(self.bytes.len()));
        self.bytes.extend_from_slice(&bytes[..accepted]);
        self.truncated |= accepted != bytes.len();
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl BoundedPanicMessage {
    fn into_string(self) -> String {
        let mut message = String::from_utf8_lossy(&self.bytes).into_owned();
        if self.truncated {
            message.push_str("\n[panic description truncated]");
        }
        message
    }
}

fn combine_cleanup(
    result: Result<()>,
    shutdown: Result<()>,
    restoration: Result<()>,
) -> Result<()> {
    let mut failures = Vec::new();
    for (stage, result) in [
        ("interactive session", result),
        ("resident worker shutdown", shutdown),
        ("terminal restoration", restoration),
    ] {
        if let Err(error) = result {
            failures.push(format!("{stage}: {error:#}"));
        }
    }
    if !failures.is_empty() {
        bail!("{}", failures.join("; "));
    }
    Ok(())
}

struct TerminalOwner {
    terminal: Option<UiTerminal>,
    lease: Option<TerminalLease>,
    raw_attempted: bool,
    alternate_attempted: bool,
    cursor_attempted: bool,
    paste_attempted: bool,
}

impl TerminalOwner {
    fn enter() -> Result<Self> {
        let mut owner = Self {
            terminal: None,
            lease: Some(TerminalLease::acquire()?),
            raw_attempted: false,
            alternate_attempted: false,
            cursor_attempted: false,
            paste_attempted: false,
        };
        let setup = (|| -> Result<()> {
            // Mark each stage before attempting it: an OS error can follow a
            // partial mode change, so restoration must still attempt that stage.
            owner.raw_attempted = true;
            enable_raw_mode()?;
            owner.alternate_attempted = true;
            execute!(io::stderr(), EnterAlternateScreen)?;
            owner.cursor_attempted = true;
            execute!(io::stderr(), Hide)?;
            owner.paste_attempted = true;
            execute!(io::stderr(), EnableBracketedPaste)?;
            let backend = CrosstermBackend::new(BufWriter::new(io::stderr()));
            let mut terminal = Terminal::new(backend)?;
            terminal.clear()?;
            owner.terminal = Some(terminal);
            Ok(())
        })();
        if let Err(error) = setup {
            return match owner.restore() {
                Ok(()) => Err(error.wrap_err("cannot initialize interactive terminal")),
                Err(cleanup) => Err(eyre::eyre!(
                    "cannot initialize interactive terminal: {error}; restoration failed: {cleanup}"
                )),
            };
        }
        Ok(owner)
    }

    fn draw(&mut self, state: &UiState) -> Result<()> {
        self.terminal
            .as_mut()
            .ok_or_else(|| eyre::eyre!("terminal was already restored"))?
            .draw(|frame| draw(frame, state))?;
        Ok(())
    }

    fn restore(&mut self) -> Result<()> {
        let mut failures = Vec::new();
        if let Some(terminal) = &mut self.terminal
            && let Err(error) = Backend::flush(terminal.backend_mut())
        {
            failures.push(format!("flush screen: {error}"));
        }
        self.terminal.take();
        // Attempt every independent stage even if an earlier one fails.
        if self.paste_attempted {
            match execute!(io::stderr(), DisableBracketedPaste) {
                Ok(()) => self.paste_attempted = false,
                Err(error) => failures.push(format!("disable bracketed paste: {error}")),
            }
        }
        if self.cursor_attempted {
            match execute!(io::stderr(), Show) {
                Ok(()) => self.cursor_attempted = false,
                Err(error) => failures.push(format!("show cursor: {error}")),
            }
        }
        if self.alternate_attempted {
            match execute!(io::stderr(), LeaveAlternateScreen) {
                Ok(()) => self.alternate_attempted = false,
                Err(error) => failures.push(format!("leave alternate screen: {error}")),
            }
        }
        if self.raw_attempted {
            match disable_raw_mode() {
                Ok(()) => self.raw_attempted = false,
                Err(error) => failures.push(format!("disable raw mode: {error}")),
            }
        }
        if !failures.is_empty() {
            // Unreleased lease Drop permanently poisons routing. A later retry
            // may repair modes, but cannot claim the failed handoff was safe.
            self.lease.take();
            bail!("{}", failures.join("; "));
        }
        if let Some(lease) = self.lease.take() {
            lease.release()?;
        }
        Ok(())
    }
}

impl Drop for TerminalOwner {
    fn drop(&mut self) {
        // The normal/error/unwind path calls restore explicitly and reports its
        // errors. This fallback covers construction failure or dropped futures.
        let _ = self.restore();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Prompt,
    Decide,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::Prompt => "Prompt",
            Self::Decide => "Decide",
        }
    }
}

struct ActiveRequest {
    id: u64,
    mode: Mode,
    started: Instant,
}

struct UiState {
    mode: Mode,
    model: String,
    decision_model: String,
    editor: Editor,
    history: History,
    active: Option<ActiveRequest>,
    status: String,
    dirty: bool,
}

impl UiState {
    fn new(model: String, args: &InteractiveCli) -> Self {
        let decision_model = args.decision_provider.map_or_else(
            || "Julia needs explicit --decision-provider and --decision-model-dir".into(),
            |provider| format!("julia-1 / {} / {}", provider.as_str(), args.device.as_str()),
        );
        Self {
            mode: Mode::Prompt,
            model,
            decision_model,
            editor: Editor::new(MAX_INPUT_BYTES),
            history: History::default(),
            active: None,
            status: "Ready. Each prompt is an independent turn; the worker remains resident."
                .into(),
            dirty: true,
        }
    }

    fn receive(&mut self, reply: WorkerReply) {
        match reply.event {
            WorkerEvent::Ready => self.status = "Resident worker ready".into(),
            WorkerEvent::Started { mode } => {
                self.status = format!("{mode}: preparing / running");
            }
            WorkerEvent::Token { text } => self.history.append_response(&text),
            WorkerEvent::Completed {
                result_json,
                elapsed_ms,
            } => {
                if self
                    .active
                    .as_ref()
                    .is_some_and(|active| active.mode == Mode::Decide)
                {
                    self.history
                        .append_response(&decision_summary(&result_json));
                } else if self.history.response_is_empty() {
                    self.history.append_response(&result_json);
                }
                self.active = None;
                self.status = format!(
                    "Completed in {:.3}s; resident worker retained",
                    elapsed_ms / 1_000.0
                );
            }
            WorkerEvent::Failed { message, cancelled } => {
                self.history.append_response(&format!(
                    "\n{}: {message}",
                    if cancelled { "Cancelled" } else { "Failed" }
                ));
                self.active = None;
                self.status = if cancelled {
                    "Request cancelled; ready for another request".into()
                } else {
                    format!("Request failed: {}", bounded_prefix(&message, 1_024))
                };
            }
        }
        self.dirty = true;
    }
}

async fn run_loop(
    args: &InteractiveCli,
    service: &TeamyLlmService,
    cancellation: &CancellationToken,
    engine: &mut InteractiveEngine,
    terminal: &mut TerminalOwner,
    model: String,
) -> Result<()> {
    let mut state = UiState::new(model, args);
    let mut events = EventStream::new();
    let mut ticks = tokio::time::interval(REFRESH_INTERVAL);
    terminal.draw(&state)?;
    state.dirty = false;
    loop {
        // Check every iteration, not only a tick: busy key/output streams must
        // never starve the global stop-after or Ctrl+C cancellation token.
        if cancellation.is_cancelled() {
            if let Some(active) = &state.active {
                engine.cancel(active.id).await?;
            }
            cancellation.bail_if_cancelled()?;
        }
        tokio::select! {
            event = events.next() => {
                let Some(event) = event else {
                    bail!("interactive terminal event stream ended");
                };
                match event? {
                    Event::Key(key) if matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) => {
                        match key_action(key) {
                            KeyAction::CancelOrQuit => {
                                if let Some(active) = &state.active {
                                    engine.cancel(active.id).await?;
                                    state.status = "Cancellation requested; waiting for the worker to stop".into();
                                } else {
                                    return Ok(());
                                }
                            }
                            KeyAction::Quit => return Ok(()),
                            KeyAction::SwitchMode => {
                                if state.active.is_some() {
                                    state.status = "Cancel or finish the current request before switching modes".into();
                                } else if state.mode == Mode::Prompt {
                                    match decision_config(args) {
                                        Ok(_) => {
                                            state.mode = Mode::Decide;
                                            state.status = "Decide: supply version 1 request JSON; choices are not executed".into();
                                            if state.editor.text.is_empty() {
                                                state.editor.insert(DECISION_EXAMPLE);
                                            }
                                        }
                                        Err(error) => state.status = format!("Decide is not configured: {error}"),
                                    }
                                } else {
                                    state.mode = Mode::Prompt;
                                    state.status = "Prompt: independent text generation turn".into();
                                }
                            }
                            KeyAction::Submit => {
                                if state.active.is_some() {
                                    state.status = "A request is already running; Ctrl+C cancels it".into();
                                } else if state.editor.text.trim().is_empty() {
                                    state.status = "Enter a prompt or request before submitting".into();
                                } else {
                                    let text = state.editor.text.clone();
                                    let action = match state.mode {
                                        Mode::Prompt => prompt_request(args, service, text.clone()).map(|request| WorkerAction::Prompt { request }),
                                        Mode::Decide => decision_config(args).map(|config| WorkerAction::Decide { config, request_json: text.clone() }),
                                    };
                                    // The editor limit counts input bytes; the
                                    // transport limit counts escaped JSON and its
                                    // envelope too. Reject oversized data without
                                    // treating it as a broken worker connection.
                                    let action = action.and_then(|action| {
                                        encode_command(&WorkerCommand {
                                            version: PROTOCOL_VERSION,
                                            id: 1,
                                            command: action.clone(),
                                        })?;
                                        Ok(action)
                                    });
                                    match action {
                                        Ok(action) => {
                                            let id = engine.submit(action).await?;
                                            state.history.begin_turn(state.mode, &text);
                                            state.active = Some(ActiveRequest { id, mode: state.mode, started: Instant::now() });
                                            state.editor.clear();
                                            state.status = "Submitted; loading occurs in the resident worker".into();
                                        }
                                        Err(error) => state.status = format!("Request rejected: {error}"),
                                    }
                                }
                            }
                            action => {
                                if !state.editor.apply(action) {
                                    state.status = format!("Input limit reached: {MAX_INPUT_BYTES} UTF-8 bytes");
                                }
                            }
                        }
                        state.dirty = true;
                        terminal.draw(&state)?;
                        state.dirty = false;
                    }
                    Event::Paste(text) => {
                        let text = text.replace("\r\n", "\n");
                        if !state.editor.insert(&text) {
                            state.status = format!("Paste rejected: input limit is {MAX_INPUT_BYTES} UTF-8 bytes");
                        }
                        state.dirty = true;
                    }
                    Event::Resize(_, _) => state.dirty = true,
                    _ => {}
                }
            }
            reply = engine.next_event() => state.receive(reply?),
            _ = ticks.tick() => {
                // Refresh bounded log history even if inference has not produced
                // an event yet. Draw cadence stays independent from GPU kernels.
                state.dirty = true;
            }
        }
        if state.dirty {
            terminal.draw(&state)?;
            state.dirty = false;
        }
    }
}

fn decision_summary(json: &str) -> String {
    // Worker frames and choices are bounded by the same contract as `decide`.
    // Render the typed result rather than dumping encoding/backend metadata.
    let parsed = (json.len() <= super::protocol::MAX_FRAME_BYTES)
        .then(|| facet_json::from_str::<DecideOutput>(json));
    let Some(Ok(output)) = parsed else {
        return format!(
            "Decision result could not be summarized; use --script for structured JSON.\n{}",
            bounded_prefix(json, DISPLAY_BYTES)
        );
    };
    if output.version != 1
        || !(2..=20).contains(&output.result.scores.len())
        || output.result.scores.len() != output.option_count
        || output.result.scores.iter().any(|score| {
            !score.logit.is_finite()
                || !score.probability.is_finite()
                || !(0.0..=1.0).contains(&score.probability)
        })
        || output
            .result
            .best_option()
            .is_none_or(|best| best.option_id != output.best_option_id)
    {
        return "Decision result failed display validation; use --script to inspect structured JSON."
            .into();
    }
    let mut summary = format!(
        "Chosen option: {}\nModel: {} / {} / {}\nOptions (input order; softmax is not calibrated confidence):\n",
        bounded_prefix(&output.best_option_id, 512),
        bounded_prefix(&output.model, 512),
        bounded_prefix(&output.provider, 64),
        bounded_prefix(&output.device, 64)
    );
    for score in &output.result.scores {
        // At most 20 bounded IDs. Formatting remains bounded independently of
        // any optional encoding/audit fields in the structured response.
        summary.push_str(&format!(
            "{} {}: {:.2}%\n",
            if score.option_id == output.best_option_id {
                "*"
            } else {
                " "
            },
            bounded_prefix(&score.option_id, 512),
            score.probability * 100.0
        ));
    }
    summary.push_str("This choice is data; no option has been executed.");
    summary
}

fn draw(frame: &mut Frame<'_>, state: &UiState) {
    let compact = frame.area().width < 90;
    let short = frame.area().height < 24;
    let areas = Layout::vertical([
        Constraint::Length(4),
        Constraint::Min(3),
        Constraint::Length(if short { 5 } else { 7 }),
        Constraint::Length(if short { 3 } else { 6 }),
        Constraint::Length(if compact { 3 } else { 2 }),
    ])
    .split(frame.area());
    let model = match state.mode {
        Mode::Prompt => &state.model,
        Mode::Decide => &state.decision_model,
    };
    let status = state.active.as_ref().map_or_else(
        || state.status.clone(),
        |active| {
            format!(
                "{} ({:.1}s)",
                state.status,
                active.started.elapsed().as_secs_f64()
            )
        },
    );
    let header = format!(
        "Mode: {} | Model: {}\n{}",
        state.mode.name(),
        bounded_prefix(model, 2_048),
        bounded_prefix(&status, 2_048)
    );
    frame.render_widget(
        Paragraph::new(sanitize_for_terminal_display(&header))
            .style(Style::default().fg(Color::Cyan))
            .block(Block::bordered().title("teamy-llm interactive")),
        areas[0],
    );

    let history_title = format!("Results · {} history omissions", state.history.omissions);
    render_tail(frame, areas[1], &state.history.display(), &history_title);

    let editor_block = Block::bordered().title(format!(
        "{} input · {} / {} bytes",
        state.mode.name(),
        state.editor.text.len(),
        MAX_INPUT_BYTES
    ));
    let editor_inner = editor_block.inner(areas[2]);
    let (before, after) = state.editor.display_window();
    let before = sanitize_for_terminal_display(&before).replace('\t', "    ");
    let editor_text = format!(
        "{before}▏{}",
        sanitize_for_terminal_display(&after).replace('\t', "    ")
    );
    let lines = wrap_lines(&editor_text, editor_inner.width);
    let caret_row = wrap_lines(&before, editor_inner.width)
        .len()
        .saturating_sub(1);
    let first_row = caret_row.saturating_sub(usize::from(editor_inner.height) / 2);
    frame.render_widget(editor_block, areas[2]);
    frame.render_widget(
        Paragraph::new(lines.into_iter().skip(first_row).collect::<Vec<_>>()),
        editor_inner,
    );

    let logs = snapshot_logs();
    let log_text = logs
        .records
        .iter()
        .rev()
        .take(8)
        .rev()
        .map(String::as_str)
        .collect::<String>();
    render_tail(
        frame,
        areas[3],
        &log_text,
        &format!("Logs · {} dropped", logs.dropped),
    );
    let help = if compact {
        "Enter submit · Tab mode\nCtrl+C/Esc cancel busy, otherwise quit\nShift+Enter newline · Ctrl+Q stop + quit"
    } else {
        "Enter submit · Shift+Enter newline · Tab mode · Ctrl+C/Esc cancel busy, otherwise quit\nArrows/Home/End edit · Ctrl+Q quit and stop worker · Prompt turns do not retain chat history"
    };
    frame.render_widget(
        Paragraph::new(help).style(Style::default().fg(Color::Gray)),
        areas[4],
    );
}

fn render_tail(frame: &mut Frame<'_>, area: Rect, text: &str, title: &str) {
    let block = Block::bordered().title(title.to_owned());
    let inner = block.inner(area);
    let safe =
        sanitize_for_terminal_display(bounded_tail(text, DISPLAY_BYTES)).replace('\t', "    ");
    let lines = wrap_lines(&safe, inner.width);
    let skip = lines.len().saturating_sub(usize::from(inner.height));
    frame.render_widget(block, area);
    frame.render_widget(
        Paragraph::new(lines.into_iter().skip(skip).collect::<Vec<_>>()),
        inner,
    );
}

// Explicitly wrap bounded display text without Ratatui's unstable line-count
// API. Measure the combined line so wide characters and combining marks use
// Ratatui's own Unicode width behavior. Stored model/input text is unchanged.
fn wrap_lines(text: &str, width: u16) -> Vec<Line<'static>> {
    if width == 0 {
        return Vec::new();
    }
    let mut lines = Vec::new();
    let mut current = String::new();
    for character in text.chars() {
        if character == '\n' {
            lines.push(Line::from(std::mem::take(&mut current)));
            continue;
        }
        current.push(character);
        if Line::from(current.as_str()).width() > usize::from(width) {
            current.pop();
            if !current.is_empty() {
                lines.push(Line::from(std::mem::take(&mut current)));
            }
            current.push(character);
        }
    }
    lines.push(Line::from(current));
    lines
}

fn bounded_prefix(text: &str, limit: usize) -> &str {
    let mut end = text.len().min(limit);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

fn bounded_tail(text: &str, limit: usize) -> &str {
    let mut start = text.len().saturating_sub(limit);
    while !text.is_char_boundary(start) {
        start += 1;
    }
    &text[start..]
}

#[derive(Debug, PartialEq, Eq)]
enum KeyAction {
    Submit,
    SwitchMode,
    CancelOrQuit,
    Quit,
    Insert(char),
    Newline,
    Left,
    Right,
    Up,
    Down,
    Home,
    End,
    Start,
    Finish,
    Backspace,
    Delete,
    Ignore,
}

fn key_action(key: KeyEvent) -> KeyAction {
    if key.modifiers.contains(KeyModifiers::CONTROL) {
        return match key.code {
            KeyCode::Char('c' | 'C') => KeyAction::CancelOrQuit,
            KeyCode::Char('q' | 'Q') => KeyAction::Quit,
            KeyCode::Home => KeyAction::Start,
            KeyCode::End => KeyAction::Finish,
            _ => KeyAction::Ignore,
        };
    }
    match key.code {
        KeyCode::Esc => KeyAction::CancelOrQuit,
        KeyCode::Tab | KeyCode::BackTab => KeyAction::SwitchMode,
        KeyCode::Enter if key.modifiers.contains(KeyModifiers::SHIFT) => KeyAction::Newline,
        KeyCode::Enter => KeyAction::Submit,
        KeyCode::Char(character)
            if !character.is_control() && !key.modifiers.contains(KeyModifiers::ALT) =>
        {
            KeyAction::Insert(character)
        }
        KeyCode::Left => KeyAction::Left,
        KeyCode::Right => KeyAction::Right,
        KeyCode::Up => KeyAction::Up,
        KeyCode::Down => KeyAction::Down,
        KeyCode::Home => KeyAction::Home,
        KeyCode::End => KeyAction::End,
        KeyCode::Backspace => KeyAction::Backspace,
        KeyCode::Delete => KeyAction::Delete,
        _ => KeyAction::Ignore,
    }
}

struct Editor {
    text: String,
    cursor: usize,
    limit: usize,
}

impl Editor {
    fn new(limit: usize) -> Self {
        Self {
            text: String::new(),
            cursor: 0,
            limit,
        }
    }

    fn insert(&mut self, text: &str) -> bool {
        if self.text.len().saturating_add(text.len()) > self.limit {
            return false;
        }
        self.text.insert_str(self.cursor, text);
        self.cursor += text.len();
        true
    }

    fn clear(&mut self) {
        self.text.clear();
        self.cursor = 0;
    }

    fn apply(&mut self, action: KeyAction) -> bool {
        match action {
            KeyAction::Insert(character) => return self.insert(character.encode_utf8(&mut [0; 4])),
            KeyAction::Newline => return self.insert("\n"),
            KeyAction::Left => self.cursor = self.previous(),
            KeyAction::Right => self.cursor = self.next(),
            KeyAction::Backspace => {
                let previous = self.previous();
                self.text.replace_range(previous..self.cursor, "");
                self.cursor = previous;
            }
            KeyAction::Delete => self.text.replace_range(self.cursor..self.next(), ""),
            KeyAction::Home => self.cursor = self.line_start(),
            KeyAction::End => self.cursor = self.line_end(),
            KeyAction::Start => self.cursor = 0,
            KeyAction::Finish => self.cursor = self.text.len(),
            KeyAction::Up => self.move_vertical(false),
            KeyAction::Down => self.move_vertical(true),
            _ => {}
        }
        true
    }

    fn previous(&self) -> usize {
        self.text[..self.cursor]
            .char_indices()
            .next_back()
            .map_or(0, |(offset, _)| offset)
    }

    fn next(&self) -> usize {
        self.cursor
            + self.text[self.cursor..]
                .chars()
                .next()
                .map_or(0, char::len_utf8)
    }

    fn line_start(&self) -> usize {
        self.text[..self.cursor]
            .rfind('\n')
            .map_or(0, |offset| offset + 1)
    }

    fn line_end(&self) -> usize {
        self.cursor
            + self.text[self.cursor..]
                .find('\n')
                .unwrap_or(self.text.len() - self.cursor)
    }

    fn move_vertical(&mut self, down: bool) {
        let start = self.line_start();
        let column = self.text[start..self.cursor].chars().count();
        let (next_start, next_end) = if down {
            let end = self.line_end();
            if end == self.text.len() {
                return;
            }
            let next_start = end + 1;
            let next_end = next_start
                + self.text[next_start..]
                    .find('\n')
                    .unwrap_or(self.text.len() - next_start);
            (next_start, next_end)
        } else {
            if start == 0 {
                return;
            }
            let next_end = start - 1;
            let next_start = self.text[..next_end]
                .rfind('\n')
                .map_or(0, |offset| offset + 1);
            (next_start, next_end)
        };
        self.cursor = self.text[next_start..next_end]
            .char_indices()
            .nth(column)
            .map_or(next_end, |(offset, _)| next_start + offset);
    }

    fn display_window(&self) -> (String, String) {
        (
            bounded_tail(&self.text[..self.cursor], DISPLAY_BYTES / 2).to_owned(),
            bounded_prefix(&self.text[self.cursor..], DISPLAY_BYTES / 2).to_owned(),
        )
    }
}

#[derive(Default)]
struct History {
    entries: VecDeque<String>,
    bytes: usize,
    omissions: u64,
    response_bytes: usize,
    response_truncated: bool,
}

impl History {
    fn begin_turn(&mut self, mode: Mode, input: &str) {
        let prefix = bounded_prefix(input, MAX_ENTRY_BYTES - 64);
        if prefix.len() != input.len() {
            self.omissions = self.omissions.saturating_add(1);
        }
        self.push(format!("You ({}):\n{prefix}", mode.name()));
        self.push(format!("{} result:\n", mode.name()));
        self.response_bytes = 0;
        self.response_truncated = false;
    }

    fn push(&mut self, text: String) {
        self.bytes += text.len();
        self.entries.push_back(text);
        self.trim();
    }

    fn response_is_empty(&self) -> bool {
        self.response_bytes == 0
    }

    fn append_response(&mut self, text: &str) {
        if self.response_truncated {
            return;
        }
        let Some(last) = self.entries.back_mut() else {
            return;
        };
        let remaining = MAX_ENTRY_BYTES.saturating_sub(last.len() + 64);
        let accepted = bounded_prefix(text, remaining);
        last.push_str(accepted);
        self.bytes += accepted.len();
        self.response_bytes += accepted.len();
        if accepted.len() != text.len() {
            const NOTICE: &str = "\n[Result display truncated]\n";
            last.push_str(NOTICE);
            self.bytes += NOTICE.len();
            self.response_truncated = true;
            self.omissions = self.omissions.saturating_add(1);
        }
        self.trim();
    }

    fn trim(&mut self) {
        while self.entries.len() > MAX_HISTORY_ENTRIES || self.bytes > MAX_HISTORY_BYTES {
            let Some(evicted) = self.entries.pop_front() else {
                break;
            };
            self.bytes -= evicted.len();
            self.omissions = self.omissions.saturating_add(1);
        }
    }

    fn display(&self) -> String {
        let mut selected = Vec::new();
        let mut remaining = DISPLAY_BYTES;
        for entry in self.entries.iter().rev() {
            if remaining == 0 {
                break;
            }
            let excerpt = bounded_tail(entry, remaining);
            remaining = remaining.saturating_sub(excerpt.len() + 2);
            selected.push(excerpt);
        }
        selected.reverse();
        selected.join("\n\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panic_description_formatting_has_a_fixed_retention_limit() {
        let mut capture = BoundedPanicMessage::default();
        write!(capture, "panic: {}", "é🙂".repeat(10_000)).unwrap();
        assert!(capture.bytes.len() <= 4_096);
        assert!(capture.truncated);
        let description = capture.into_string();
        assert!(description.starts_with("panic: "));
        assert!(description.ends_with("[panic description truncated]"));
        assert!(description.len() < 4_160);
    }

    #[test]
    fn header_status_and_quit_help_remain_visible_at_common_terminal_sizes() {
        use ratatui::backend::TestBackend;

        let state = UiState {
            mode: Mode::Prompt,
            model: "qwen-test".into(),
            decision_model: "julia-test".into(),
            editor: Editor::new(MAX_INPUT_BYTES),
            history: History::default(),
            active: Some(ActiveRequest {
                id: 1,
                mode: Mode::Prompt,
                started: Instant::now(),
            }),
            status: "Running request".into(),
            dirty: true,
        };
        for (width, height) in [(80, 24), (40, 18)] {
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal.draw(|frame| draw(frame, &state)).unwrap();
            let screen = terminal
                .backend()
                .buffer()
                .content()
                .chunks(usize::from(width))
                .map(|row| row.iter().map(|cell| cell.symbol()).collect::<String>())
                .collect::<Vec<_>>()
                .join("\n");
            assert!(
                screen.contains("Mode: Prompt"),
                "{width}×{height}: {screen}"
            );
            assert!(
                screen.contains("Model: qwen-test"),
                "{width}×{height}: {screen}"
            );
            assert!(
                screen.contains("Running request ("),
                "{width}×{height}: {screen}"
            );
            assert!(screen.contains("Ctrl+C/Esc"), "{width}×{height}: {screen}");
            assert!(screen.contains("Ctrl+Q"), "{width}×{height}: {screen}");
        }
    }

    #[test]
    fn finite_choice_summary_preserves_input_order_and_identifies_best_option() {
        use teamy_llm_core::finite_choice::{FiniteChoiceResult, FiniteChoiceScore};
        use teamy_llm_julia_common::EncodingPolicy;

        let output = DecideOutput {
            version: 1,
            task: "finite-choice".into(),
            model: "julia-1".into(),
            provider: "native".into(),
            device: "cuda".into(),
            question: "Which option should be chosen?".into(),
            encoding_policy: EncodingPolicy::default(),
            token_count: 10,
            option_count: 2,
            load_ms: 1.0,
            inference_ms: vec![2.0],
            best_option_id: "heal".into(),
            result: FiniteChoiceResult {
                scores: vec![
                    FiniteChoiceScore {
                        option_id: "wait".into(),
                        logit: 0.0,
                        probability: 0.25,
                    },
                    FiniteChoiceScore {
                        option_id: "heal".into(),
                        logit: 3.0_f64.ln(),
                        probability: 0.75,
                    },
                ],
            },
            encoding: None,
            onnx_audit: None,
            native_metadata: None,
            profiling_file: None,
        };
        let summary = decision_summary(&facet_json::to_string(&output).unwrap());
        assert!(summary.contains("Chosen option: heal"));
        assert!(summary.contains("* heal: 75.00%"));
        assert!(summary.find("wait: 25.00%").unwrap() < summary.find("heal: 75.00%").unwrap());
        assert!(summary.contains("not calibrated confidence"));
        assert!(!summary.contains("encoding-policy"));
    }

    #[test]
    fn unicode_insert_navigation_and_deletion_preserve_byte_boundaries() {
        let mut editor = Editor::new(32);
        assert!(editor.insert("aé🙂z"));
        editor.apply(KeyAction::Left);
        editor.apply(KeyAction::Backspace);
        assert_eq!(editor.text, "aéz");
        assert_eq!(editor.cursor, "aé".len());
        editor.apply(KeyAction::Left);
        editor.apply(KeyAction::Delete);
        assert_eq!(editor.text, "az");
        assert!(editor.insert("界"));
        assert_eq!(editor.text, "a界z");
        assert!(editor.text.is_char_boundary(editor.cursor));
    }

    #[test]
    fn oversized_unicode_paste_is_rejected_without_partial_editing() {
        let mut editor = Editor::new(5);
        assert!(editor.insert("é"));
        assert!(!editor.insert("🙂"));
        assert_eq!(editor.text, "é");
        assert_eq!(editor.cursor, 2);
        assert!(editor.insert("界"));
        assert_eq!(editor.text.len(), 5);
    }

    #[test]
    fn multiline_navigation_uses_character_columns_and_empty_line_edges() {
        let mut editor = Editor::new(64);
        editor.insert("éx\n🙂yz\n\nend");
        editor.apply(KeyAction::Start);
        editor.apply(KeyAction::Right);
        editor.apply(KeyAction::Down);
        assert_eq!(&editor.text[..editor.cursor], "éx\n🙂");
        editor.apply(KeyAction::End);
        assert_eq!(&editor.text[..editor.cursor], "éx\n🙂yz");
        editor.apply(KeyAction::Down);
        assert_eq!(&editor.text[..editor.cursor], "éx\n🙂yz\n");
        editor.apply(KeyAction::Up);
        assert_eq!(&editor.text[..editor.cursor], "éx\n");
    }

    #[test]
    fn controls_route_to_actions_instead_of_prompt_text() {
        assert_eq!(
            key_action(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
            KeyAction::CancelOrQuit
        );
        assert_eq!(
            key_action(KeyEvent::new(KeyCode::Enter, KeyModifiers::SHIFT)),
            KeyAction::Newline
        );
        assert_eq!(
            key_action(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE)),
            KeyAction::SwitchMode
        );
        assert_eq!(
            key_action(KeyEvent::new(KeyCode::Char('\u{1b}'), KeyModifiers::NONE)),
            KeyAction::Ignore
        );
    }

    #[test]
    fn escaped_controls_and_unicode_are_safe_display_data() {
        let safe = sanitize_for_terminal_display("界\u{1b}[2J\r\n🙂");
        assert!(!safe.contains('\u{1b}'));
        assert!(!safe.contains('\r'));
        let lines = wrap_lines(&safe, 8);
        assert!(lines.iter().all(|line| line.width() <= 8));
        assert!(lines.iter().any(|line| line.to_string().contains("🙂")));
    }

    #[test]
    fn bounded_history_does_not_keep_unbounded_inputs_or_token_streams() {
        let mut history = History::default();
        history.begin_turn(Mode::Prompt, &"é".repeat(MAX_ENTRY_BYTES));
        history.append_response(&"🙂".repeat(MAX_ENTRY_BYTES));
        assert!(
            history
                .entries
                .iter()
                .all(|entry| entry.len() <= MAX_ENTRY_BYTES)
        );
        assert_eq!(history.omissions, 2);
        for _ in 0..300 {
            history.begin_turn(Mode::Prompt, "next");
        }
        assert!(history.entries.len() <= MAX_HISTORY_ENTRIES);
        assert!(history.bytes <= MAX_HISTORY_BYTES);
        assert_eq!(
            history.bytes,
            history.entries.iter().map(String::len).sum::<usize>()
        );
    }

    #[test]
    fn cleanup_reports_worker_and_restoration_failures_together() {
        let error = combine_cleanup(
            Ok(()),
            Err(eyre::eyre!("worker failed")),
            Err(eyre::eyre!("raw restoration failed")),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("worker failed"));
        assert!(error.contains("raw restoration failed"));
    }
}
