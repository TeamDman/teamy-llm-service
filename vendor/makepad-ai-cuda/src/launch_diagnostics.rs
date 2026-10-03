//! Host launch diagnostics routed through the application's tracing subscriber.
//!
//! The C formatter owns its bounded stack buffer throughout the callback. Its bytes are
//! borrowed only for this call, decoded without loss and never retained. The callback is
//! a process-lifetime function with no caller state; registration is safe to repeat and
//! the native function pointer is stored atomically. Subscriber panics are caught before
//! returning through the C ABI. Kernel-less builds reference no native symbol.

#[cfg(any(makepad_ai_cuda_kernels, test))]
use std::ffi::c_char;

#[cfg(any(makepad_ai_cuda_kernels, test))]
const MAX_DIAGNOSTIC_BYTES: usize = 1023;
#[cfg(any(makepad_ai_cuda_kernels, test))]
const LOG_TARGET: &str = "teamy_llm_makepad_gguf::cuda::launch";

#[cfg(makepad_ai_cuda_kernels)]
unsafe extern "C" {
    fn mkllm_set_launch_diagnostic_callback(
        callback: Option<unsafe extern "C" fn(*const c_char, usize)>,
    );
}

/// Install process-lifetime host launch diagnostics at DEBUG level.
///
/// This initializes no CUDA context or model. Events respect the current tracing subscriber
/// and its filters/log destinations; without a subscriber they are silent. A build without
/// compiled CUDA kernels does nothing and requires no native libraries for this hook.
pub fn install_tracing_callback() {
    #[cfg(makepad_ai_cuda_kernels)]
    // SAFETY: this stateless function exists for the process lifetime. Native code stores
    // only its function pointer and supplies bounded, live formatter bytes during each call.
    unsafe {
        mkllm_set_launch_diagnostic_callback(Some(receive_launch_diagnostic));
    }
}

/// # Safety
/// For a non-null pointer and a length in 1..=1023, `bytes` must identify readable bytes
/// that stay unchanged until this callback returns. Native formatting satisfies this contract.
#[cfg(any(makepad_ai_cuda_kernels, test))]
unsafe extern "C" fn receive_launch_diagnostic(bytes: *const c_char, len: usize) {
    if bytes.is_null() || len == 0 || len > MAX_DIAGNOSTIC_BYTES {
        return;
    }
    // A subscriber is application code and can panic. No unwind may escape to C.
    let outcome = std::panic::catch_unwind(|| {
        if !tracing::enabled!(target: LOG_TARGET, tracing::Level::DEBUG) {
            return;
        }
        // SAFETY: the native caller promises live, immutable formatter bytes for this call;
        // the null and size checks above also reject malformed bounds before constructing a view.
        let borrowed = unsafe { std::slice::from_raw_parts(bytes.cast::<u8>(), len) };
        let Ok(message) = std::str::from_utf8(borrowed) else {
            return;
        };
        let message = message.trim_end_matches(['\r', '\n']);
        if !message.is_empty() {
            tracing::debug!(target: LOG_TARGET, diagnostic = %message, "CUDA kernel launch");
        }
    });
    if let Err(payload) = outcome {
        // Even a caller-defined panic payload can panic in Drop. Retain this exceptional
        // payload rather than run an untrusted destructor outside the unwind guard in C.
        std::mem::forget(payload);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use tracing::field::{Field, Visit};
    use tracing::span::{Attributes, Id, Record};
    use tracing::{Event, Metadata, Subscriber};

    #[derive(Clone)]
    struct Recorder {
        debug_enabled: bool,
        panic_on_event: bool,
        events: Arc<Mutex<Vec<String>>>,
    }

    struct DiagnosticVisitor(Option<String>);

    struct HostilePanicPayload;

    impl Drop for HostilePanicPayload {
        fn drop(&mut self) {
            panic!("fixture panic payload destructor must not cross C");
        }
    }

    impl Visit for DiagnosticVisitor {
        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            if field.name() == "diagnostic" {
                self.0 = Some(format!("{value:?}"));
            }
        }
    }

    impl Subscriber for Recorder {
        fn enabled(&self, metadata: &Metadata<'_>) -> bool {
            metadata.target() == LOG_TARGET
                && (*metadata.level() != tracing::Level::DEBUG || self.debug_enabled)
        }

        fn new_span(&self, _: &Attributes<'_>) -> Id {
            Id::from_u64(1)
        }

        fn record(&self, _: &Id, _: &Record<'_>) {}

        fn record_follows_from(&self, _: &Id, _: &Id) {}

        fn event(&self, event: &Event<'_>) {
            assert_eq!(*event.metadata().level(), tracing::Level::DEBUG);
            assert_eq!(event.metadata().target(), LOG_TARGET);
            if self.panic_on_event {
                std::panic::panic_any(HostilePanicPayload);
            }
            let mut fields = DiagnosticVisitor(None);
            event.record(&mut fields);
            self.events.lock().unwrap().push(fields.0.unwrap());
        }

        fn enter(&self, _: &Id) {}

        fn exit(&self, _: &Id) {}
    }

    fn recorder(debug_enabled: bool, panic_on_event: bool) -> Recorder {
        Recorder {
            debug_enabled,
            panic_on_event,
            events: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn receive(bytes: &[u8]) {
        // SAFETY: the borrowed fixture remains immutable and live for the entire call.
        unsafe { receive_launch_diagnostic(bytes.as_ptr().cast(), bytes.len()) };
    }

    #[test]
    fn launch_diagnostics_use_debug_target_and_respect_subscriber_filter() {
        let capture = recorder(true, false);
        tracing::subscriber::with_default(capture.clone(), || {
            receive(b"fattn.launch: kind=mma32x2 n_q=18\n");
        });
        assert_eq!(
            capture.events.lock().unwrap().as_slice(),
            &["fattn.launch: kind=mma32x2 n_q=18"]
        );
        let disabled = recorder(false, false);
        tracing::subscriber::with_default(disabled.clone(), || receive(b"mmq.launch: m=18"));
        assert!(disabled.events.lock().unwrap().is_empty());
    }

    #[test]
    fn callback_rejects_invalid_bounds_or_utf8_and_preserves_full_bound() {
        let capture = recorder(true, false);
        tracing::subscriber::with_default(capture.clone(), || {
            // SAFETY: null/zero/oversize are rejected before any pointer dereference.
            unsafe {
                receive_launch_diagnostic(std::ptr::null(), 1);
                receive_launch_diagnostic(std::ptr::null(), 0);
                receive_launch_diagnostic(std::ptr::dangling::<c_char>(), MAX_DIAGNOSTIC_BYTES + 1);
            }
            receive(&[0xff]);
            receive(b"\r\n");
            receive(&[b'x'; MAX_DIAGNOSTIC_BYTES]);
        });
        let events = capture.events.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].len(), MAX_DIAGNOSTIC_BYTES);
        assert!(events[0].bytes().all(|byte| byte == b'x'));
    }

    #[test]
    fn subscriber_panic_is_contained_inside_the_c_callback() {
        tracing::subscriber::with_default(recorder(true, true), || {
            receive(b"mmvq.launch: ncols_dst=1");
        });
    }
}
