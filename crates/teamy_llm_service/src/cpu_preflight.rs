//! Join CPU verification and tokenizer validation before permitting GPU construction.

use teamy_cancellation::CancellationToken;
use teamy_llm_core::{ServiceError, ServiceErrorKind};

pub(crate) fn validate_with_preflight<T: Send>(
    cancellation: &CancellationToken,
    preflight: impl FnOnce() -> Result<T, ServiceError> + Send,
    validate: impl FnOnce() -> Result<(), ServiceError>,
) -> Result<T, ServiceError> {
    let span = tracing::Span::current();
    std::thread::scope(|scope| {
        let worker = std::thread::Builder::new()
            .name("llm-cpu-preflight".into())
            .spawn_scoped(scope, move || span.in_scope(preflight))
            .map_err(|error| {
                ServiceError::new(
                    ServiceErrorKind::Internal,
                    format!("Starting CPU model preflight failed: {error}"),
                )
            })?;
        let validation = validate();
        if validation.is_err() {
            cancellation.request_cancel("Tokenizer validation failed during CPU model preflight");
        }
        // Even invalid tokenizers must not leave a reader using the mapped model behind.
        let prepared = worker.join().unwrap_or_else(|_| {
            Err(ServiceError::new(
                ServiceErrorKind::Internal,
                "CPU model preflight panicked",
            ))
        });
        validation?;
        prepared
    })
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Barrier};

    use super::*;

    #[test]
    fn cpu_jobs_overlap_and_both_complete_before_return() {
        let rendezvous = Arc::new(Barrier::new(2));
        let worker_barrier = rendezvous.clone();
        let result = validate_with_preflight(
            &CancellationToken::new(),
            move || {
                worker_barrier.wait();
                Ok(42)
            },
            || {
                rendezvous.wait();
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(result, 42);
    }

    #[test]
    fn invalid_tokenizer_cancels_and_joins_preflight_and_preserves_its_error() {
        let parent = CancellationToken::new();
        let cancellation = parent.child_token();
        let worker_token = cancellation.clone();
        let finished = Arc::new(AtomicBool::new(false));
        let worker_finished = finished.clone();
        let error = validate_with_preflight::<()>(
            &cancellation,
            move || {
                while !worker_token.is_cancelled() {
                    std::thread::yield_now();
                }
                worker_finished.store(true, Ordering::Release);
                Err(ServiceError::new(
                    ServiceErrorKind::Cancelled,
                    "CPU work stopped",
                ))
            },
            || {
                Err(ServiceError::new(
                    ServiceErrorKind::ModelNotPrepared,
                    "invalid tokenizer",
                ))
            },
        )
        .unwrap_err();
        assert_eq!(error.kind, ServiceErrorKind::ModelNotPrepared);
        assert!(finished.load(Ordering::Acquire));
        assert!(
            !parent.is_cancelled(),
            "invalid tokenizer must not cancel the service"
        );
    }

    #[test]
    fn preflight_failure_and_panic_are_typed_errors_before_gpu_construction() {
        let error = validate_with_preflight::<()>(
            &CancellationToken::new(),
            || {
                Err(ServiceError::new(
                    ServiceErrorKind::Internal,
                    "bad weight commitment",
                ))
            },
            || Ok(()),
        )
        .unwrap_err();
        assert!(error.message.contains("bad weight commitment"));
        let panic = validate_with_preflight::<()>(
            &CancellationToken::new(),
            || panic!("fixture CPU reader panic"),
            || Ok(()),
        )
        .unwrap_err();
        assert_eq!(panic.kind, ServiceErrorKind::Internal);
        assert!(panic.message.contains("panicked"));
    }

    #[test]
    fn parent_cancellation_reaches_cpu_child_and_reader_is_joined() {
        let parent = CancellationToken::new();
        let child = parent.child_token();
        let worker_token = child.clone();
        let stopped = AtomicBool::new(false);
        let error = validate_with_preflight::<()>(
            &child,
            || {
                while !worker_token.is_cancelled() {
                    std::thread::yield_now();
                }
                stopped.store(true, Ordering::Release);
                Err(ServiceError::new(
                    ServiceErrorKind::Cancelled,
                    "parent cancelled",
                ))
            },
            || {
                parent.request_cancel("fixture operator cancellation");
                Ok(())
            },
        )
        .unwrap_err();
        assert_eq!(error.kind, ServiceErrorKind::Cancelled);
        assert!(stopped.load(Ordering::Acquire));
    }
}
