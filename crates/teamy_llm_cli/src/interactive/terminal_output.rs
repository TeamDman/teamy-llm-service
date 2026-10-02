//! One-owner terminal output routing for the current interactive engine.
//!
//! This is deliberately smaller than Cloud-Terrastodon's nested coordinator:
//! overlapping owners are rejected rather than suspended and resumed. Only the
//! human tracing layer uses this writer; NDJSON remains an independent layer.
//! A lease must be released *after* successful terminal restoration. Dropping
//! an unreleased lease poisons the router and keeps subsequent logs in memory.

use std::collections::VecDeque;
use std::fmt::Write as _;
use std::io::{self, Write};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use eyre::{Result, bail};
use tracing_subscriber::fmt::MakeWriter;

const MAX_RECORDS: usize = 1_024;
const MAX_RETAINED_BYTES: usize = 1_048_576;
const MAX_WRITER_BYTES: usize = 65_536;
const TRUNCATION_NOTICE: &str = "\n[human log record truncated]\n";

static HUMAN_OUTPUT: OnceLock<Arc<OutputRouter>> = OnceLock::new();

/// A cloneable formatted-record writer for the human tracing layer.
#[derive(Clone)]
pub struct HumanWriter {
    router: Arc<OutputRouter>,
}

/// A bounded snapshot of records captured while a terminal lease was active.
///
/// `dropped` counts evicted records and records lost wholly or partly because
/// of a size limit. The count saturates instead of overflowing. Snapshotting
/// does not consume history or write to the terminal.
#[derive(Clone, Debug, Default)]
pub struct LogSnapshot {
    pub records: Vec<String>,
    pub dropped: u64,
}

/// Return the process-wide writer without changing terminal ownership.
pub fn human_writer() -> HumanWriter {
    HumanWriter {
        router: global_router().clone(),
    }
}

/// Inspect bounded human-log history. Render its strings as data using
/// [`sanitize_for_terminal_display`], rather than replaying terminal escapes.
pub fn snapshot_logs() -> LogSnapshot {
    global_router().snapshot()
}

/// Report delivery or ownership failures before a command returns success.
///
/// `MakeWriter` commits a complete formatted record on writer drop, where Rust
/// cannot return a write error. Errors are therefore retained here and also
/// checked when acquiring or releasing a lease. The CLI must call this after
/// command dispatch, including commands that never acquire a terminal lease.
/// Disable the human fmt layer's `log_internal_errors` fallback: that fallback
/// writes straight to stderr and bypasses terminal ownership.
pub fn check_human_output() -> Result<()> {
    global_router().check_health()
}

fn global_router() -> &'static Arc<OutputRouter> {
    HUMAN_OUTPUT.get_or_init(|| {
        Arc::new(OutputRouter::new(
            Box::new(io::stderr()),
            Limits {
                records: MAX_RECORDS,
                retained_bytes: MAX_RETAINED_BYTES,
                writer_bytes: MAX_WRITER_BYTES,
            },
        ))
    })
}

/// Exclusive ownership of the current terminal screen.
#[must_use = "retain the lease until the terminal has been restored"]
pub struct TerminalLease {
    router: Arc<OutputRouter>,
    id: u64,
    released: bool,
}

impl TerminalLease {
    /// Stop human stderr delivery before changing any terminal modes.
    pub fn acquire() -> Result<Self> {
        Self::acquire_from(global_router().clone())
    }

    fn acquire_from(router: Arc<OutputRouter>) -> Result<Self> {
        let id = {
            let mut state = router.lock_state();
            state.check_health()?;
            match state.mode {
                Mode::Idle => {}
                Mode::Active(_) => bail!("an interactive terminal owner is already active"),
                Mode::Poisoned => bail!("human terminal output is poisoned"),
            }
            let id = state.next_lease_id;
            let Some(next) = id.checked_add(1) else {
                state.fail("terminal lease identifier space exhausted".into());
                bail!("terminal lease identifier space exhausted");
            };
            state.next_lease_id = next;
            state.mode = Mode::Active(id);
            id
        };
        Ok(Self {
            router,
            id,
            released: false,
        })
    }

    /// Resume normal human output after raw mode, screen and cursor restoration.
    ///
    /// This does not replay buffered records into the terminal. A failed release
    /// leaves the router poisoned and its history retained.
    pub fn release(mut self) -> Result<()> {
        {
            let mut state = self.router.lock_state();
            state.check_health()?;
            match state.mode {
                Mode::Active(id) if id == self.id => state.mode = Mode::Idle,
                _ => {
                    state.fail("terminal lease release did not match its active owner".into());
                    bail!("terminal lease release did not match its active owner");
                }
            }
        }
        self.released = true;
        Ok(())
    }
}

impl Drop for TerminalLease {
    fn drop(&mut self) {
        if !self.released {
            self.router
                .lock_state()
                .fail("terminal lease was dropped before successful terminal restoration".into());
        }
    }
}

/// One bounded formatted event. The router commits it when formatting finishes.
pub struct RecordWriter {
    router: Arc<OutputRouter>,
    bytes: Vec<u8>,
    truncated: bool,
}

impl<'a> MakeWriter<'a> for HumanWriter {
    type Writer = RecordWriter;

    fn make_writer(&'a self) -> Self::Writer {
        RecordWriter {
            router: self.router.clone(),
            bytes: Vec::new(),
            truncated: false,
        }
    }
}

impl Write for RecordWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let remaining = self
            .router
            .limits
            .writer_bytes
            .saturating_sub(self.bytes.len());
        let accepted = remaining.min(bytes.len());
        self.bytes.extend_from_slice(&bytes[..accepted]);
        self.truncated |= accepted != bytes.len();
        // Oversized human events are deliberately lossy, with an explicit
        // truncation marker and dropped count. They cannot allocate without bound.
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        // Formatting may write a record in several chunks. Commit once on Drop.
        Ok(())
    }
}

impl Drop for RecordWriter {
    fn drop(&mut self) {
        if self.bytes.is_empty() && !self.truncated {
            return;
        }
        let mut text = String::from_utf8_lossy(&self.bytes).into_owned();
        if self.truncated {
            text.push_str(TRUNCATION_NOTICE);
        }
        self.router.commit(text, self.truncated);
    }
}

#[derive(Clone, Copy)]
struct Limits {
    records: usize,
    retained_bytes: usize,
    writer_bytes: usize,
}

struct OutputRouter {
    state: Mutex<State>,
    limits: Limits,
}

#[derive(Clone, Copy)]
enum Mode {
    Idle,
    Active(u64),
    Poisoned,
}

struct State {
    sink: Box<dyn Write + Send>,
    mode: Mode,
    next_lease_id: u64,
    records: VecDeque<String>,
    retained_bytes: usize,
    dropped: u64,
    failure: Option<String>,
}

impl State {
    fn fail(&mut self, message: String) {
        self.mode = Mode::Poisoned;
        if self.failure.is_none() {
            self.failure = Some(message);
        }
    }

    fn check_health(&self) -> Result<()> {
        if let Some(error) = &self.failure {
            bail!("human terminal output failed: {error}");
        }
        Ok(())
    }

    fn retain(&mut self, text: String, limits: Limits) {
        if limits.records == 0 || text.len() > limits.retained_bytes {
            self.dropped = self.dropped.saturating_add(1);
            return;
        }
        while self.records.len() >= limits.records
            || self.retained_bytes.saturating_add(text.len()) > limits.retained_bytes
        {
            let Some(evicted) = self.records.pop_front() else {
                break;
            };
            self.retained_bytes -= evicted.len();
            self.dropped = self.dropped.saturating_add(1);
        }
        self.retained_bytes += text.len();
        self.records.push_back(text);
    }
}

impl OutputRouter {
    fn new(sink: Box<dyn Write + Send>, limits: Limits) -> Self {
        Self {
            state: Mutex::new(State {
                sink,
                mode: Mode::Idle,
                next_lease_id: 1,
                records: VecDeque::new(),
                retained_bytes: 0,
                dropped: 0,
                failure: None,
            }),
            limits,
        }
    }

    fn lock_state(&self) -> MutexGuard<'_, State> {
        match self.state.lock() {
            Ok(state) => state,
            Err(poisoned) => {
                let mut state = poisoned.into_inner();
                state.fail("human-output router mutex was poisoned".into());
                state
            }
        }
    }

    fn check_health(&self) -> Result<()> {
        self.lock_state().check_health()
    }

    fn snapshot(&self) -> LogSnapshot {
        let state = self.lock_state();
        LogSnapshot {
            records: state.records.iter().cloned().collect(),
            dropped: state.dropped,
        }
    }

    fn commit(&self, text: String, truncated: bool) {
        // Keep routing and writes under the same lock as acquisition/release.
        // A writer created before acquisition but dropped afterwards is buffered.
        let mut state = self.lock_state();
        if truncated {
            state.dropped = state.dropped.saturating_add(1);
        }
        match state.mode {
            Mode::Idle => {
                let result = state
                    .sink
                    .write_all(text.as_bytes())
                    .and_then(|()| state.sink.flush());
                if let Err(error) = result {
                    state.fail(format!("stderr delivery failed: {error}"));
                    state.retain(text, self.limits);
                }
            }
            Mode::Active(_) | Mode::Poisoned => state.retain(text, self.limits),
        }
    }
}

/// Escape control characters in model text or formatted log records before
/// displaying them as terminal data. Newlines and tabs retain their normal
/// layout meaning; ESC, CR, C0/C1 controls and bidi controls become visible text.
pub fn sanitize_for_terminal_display(text: &str) -> String {
    let mut safe = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '\n' | '\t' => safe.push(character),
            '\r' => safe.push_str("\\r"),
            '\u{1b}' => safe.push_str("\\x1b"),
            character if character.is_control() || is_bidi_control(character) => {
                let code = u32::from(character);
                if code <= 0xff {
                    let _ = write!(safe, "\\x{code:02x}");
                } else {
                    let _ = write!(safe, "\\u{{{code:04x}}}");
                }
            }
            _ => safe.push(character),
        }
    }
    safe
}

fn is_bidi_control(character: char) -> bool {
    matches!(
        character,
        '\u{061c}' | '\u{200e}' | '\u{200f}' | '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}'
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct CapturedSink(Arc<Mutex<Vec<u8>>>);

    impl Write for CapturedSink {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn isolated(limits: Limits) -> (HumanWriter, Arc<Mutex<Vec<u8>>>) {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let router = Arc::new(OutputRouter::new(
            Box::new(CapturedSink(captured.clone())),
            limits,
        ));
        (HumanWriter { router }, captured)
    }

    fn generous_limits() -> Limits {
        Limits {
            records: 1_024,
            retained_bytes: 1_048_576,
            writer_bytes: 65_536,
        }
    }

    fn record(writer: &HumanWriter, text: &str) {
        let mut event = writer.make_writer();
        event.write_all(text.as_bytes()).unwrap();
    }

    #[test]
    fn normal_records_go_to_sink_after_the_complete_writer_is_dropped() {
        let (writer, captured) = isolated(generous_limits());
        let mut event = writer.make_writer();
        event.write_all(b"one ").unwrap();
        event.write_all(b"record\n").unwrap();
        event.flush().unwrap();
        assert!(captured.lock().unwrap().is_empty());
        drop(event);
        assert_eq!(&*captured.lock().unwrap(), b"one record\n");
        assert!(writer.router.snapshot().records.is_empty());
        writer.router.check_health().unwrap();
    }

    #[test]
    fn an_active_lease_buffers_background_records_and_release_resumes_output() {
        let (writer, captured) = isolated(generous_limits());
        let lease = TerminalLease::acquire_from(writer.router.clone()).unwrap();
        let background = writer.clone();
        std::thread::spawn(move || record(&background, "background\n"))
            .join()
            .unwrap();
        assert!(captured.lock().unwrap().is_empty());
        assert_eq!(writer.router.snapshot().records, ["background\n"]);
        lease.release().unwrap();
        record(&writer, "restored\n");
        assert_eq!(&*captured.lock().unwrap(), b"restored\n");
        assert_eq!(writer.router.snapshot().records, ["background\n"]);
    }

    #[test]
    fn acquisition_routes_an_already_created_writer_without_a_screen_race() {
        let (writer, captured) = isolated(generous_limits());
        let mut event = writer.make_writer();
        event.write_all(b"created before acquisition\n").unwrap();
        let lease = TerminalLease::acquire_from(writer.router.clone()).unwrap();
        drop(event);
        assert!(captured.lock().unwrap().is_empty());
        assert_eq!(writer.router.snapshot().records.len(), 1);
        lease.release().unwrap();
    }

    #[test]
    fn retention_evicts_oldest_records_by_count_and_bytes() {
        let (writer, _) = isolated(Limits {
            records: 2,
            retained_bytes: 8,
            writer_bytes: 64,
        });
        let lease = TerminalLease::acquire_from(writer.router.clone()).unwrap();
        record(&writer, "1111");
        record(&writer, "2222");
        record(&writer, "3333");
        assert_eq!(writer.router.snapshot().records, ["2222", "3333"]);
        assert_eq!(writer.router.snapshot().dropped, 1);
        record(&writer, "44444");
        assert_eq!(writer.router.snapshot().records, ["44444"]);
        assert_eq!(writer.router.snapshot().dropped, 3);
        record(&writer, "oversized record");
        assert_eq!(writer.router.snapshot().records, ["44444"]);
        assert_eq!(writer.router.snapshot().dropped, 4);
        lease.release().unwrap();
    }

    #[test]
    fn a_single_formatted_record_cannot_grow_without_bound() {
        let (writer, _) = isolated(Limits {
            writer_bytes: 4,
            ..generous_limits()
        });
        let lease = TerminalLease::acquire_from(writer.router.clone()).unwrap();
        let mut event = writer.make_writer();
        event.write_all(b"123456").unwrap();
        event.write_all(b"789").unwrap();
        assert_eq!(event.bytes, b"1234");
        drop(event);
        let snapshot = writer.router.snapshot();
        assert_eq!(snapshot.dropped, 1);
        assert_eq!(snapshot.records, [format!("1234{TRUNCATION_NOTICE}")]);
        lease.release().unwrap();
    }

    #[test]
    fn overlapping_owners_are_rejected_without_poisoning_the_first() {
        let (writer, _) = isolated(generous_limits());
        let lease = TerminalLease::acquire_from(writer.router.clone()).unwrap();
        assert!(TerminalLease::acquire_from(writer.router.clone()).is_err());
        writer.router.check_health().unwrap();
        lease.release().unwrap();
        TerminalLease::acquire_from(writer.router.clone())
            .unwrap()
            .release()
            .unwrap();
    }

    #[test]
    fn failed_restoration_poisoning_retains_logs_and_rejects_future_ownership() {
        let (writer, captured) = isolated(generous_limits());
        let lease = TerminalLease::acquire_from(writer.router.clone()).unwrap();
        record(&writer, "before restoration failure\n");
        drop(lease);
        record(&writer, "after restoration failure\n");
        assert!(writer.router.check_health().is_err());
        assert!(TerminalLease::acquire_from(writer.router.clone()).is_err());
        assert!(captured.lock().unwrap().is_empty());
        assert_eq!(writer.router.snapshot().records.len(), 2);
    }

    #[test]
    fn writer_drop_failure_is_reported_by_health_and_blocks_ownership() {
        struct FailingSink;
        impl Write for FailingSink {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "test sink failed",
                ))
            }

            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let writer = HumanWriter {
            router: Arc::new(OutputRouter::new(Box::new(FailingSink), generous_limits())),
        };
        record(&writer, "undelivered\n");
        let error = writer.router.check_health().unwrap_err().to_string();
        assert!(error.contains("test sink failed"));
        assert_eq!(writer.router.snapshot().records, ["undelivered\n"]);
        assert!(TerminalLease::acquire_from(writer.router.clone()).is_err());
    }

    #[test]
    fn a_poisoned_mutex_does_not_panic_or_resume_human_output() {
        let (writer, captured) = isolated(generous_limits());
        let router = writer.router.clone();
        let failure = std::thread::spawn(move || {
            let _held = router.state.lock().unwrap();
            panic!("injected lock-holder failure");
        });
        assert!(failure.join().is_err());
        record(&writer, "retained after mutex poison\n");
        assert!(writer.router.check_health().is_err());
        assert_eq!(writer.router.snapshot().records.len(), 1);
        assert!(captured.lock().unwrap().is_empty());
    }

    #[test]
    fn terminal_display_preserves_unicode_layout_and_escapes_controls() {
        assert_eq!(
            sanitize_for_terminal_display("café\n\t\u{1b}[2J\r\0\u{7}\u{7f}\u{85}\u{202e}data"),
            "café\n\t\\x1b[2J\\r\\x00\\x07\\x7f\\x85\\u{202e}data"
        );
        assert_eq!(sanitize_for_terminal_display("👋🏽 hello"), "👋🏽 hello");
    }
}
