//! A successful RPC reply does not imply that its queued token events were read.
use std::{future::Future, time::Duration};

use eyre::{Result, WrapErr, bail};
use teamy_cancellation::CancellationToken;
use teamy_llm_core::{GenerateEvent, GenerateRequest, GenerateResult};
use teamy_llm_service::vox_api::TeamyLlmClient;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const CANCEL_SEND_TIMEOUT: Duration = Duration::from_secs(2);
const TERMINAL_DRAIN_TIMEOUT: Duration = Duration::from_secs(5);
const EVENT_CREDIT: u32 = 16;

pub(crate) async fn wait_cancelled(cancellation: &CancellationToken) {
    while !cancellation.is_cancelled() {
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

trait EventSource {
    async fn next_event(&mut self) -> Result<Option<GenerateEvent>>;
}

struct VoxEvents(vox::Rx<GenerateEvent>);
impl EventSource for VoxEvents {
    async fn next_event(&mut self) -> Result<Option<GenerateEvent>> {
        self.0
            .recv()
            .await
            .map(|event| event.map(take_owned_event))
            .wrap_err("resident generation event stream failed")
    }
}

fn take_owned_event(event: vox::SelfRef<GenerateEvent>) -> GenerateEvent {
    // GenerateEvent contains owned Strings/integers only. SelfRef::get is for
    // Reborrow-enabled wire types; projecting this owned value with map moves
    // it out without adding a wire dependency or an unsafe impl to core types.
    let mut owned = None;
    drop(event.map(|event| owned = Some(event)));
    owned.expect("SelfRef::map always invokes its projection")
}

/// Require both terminal RPC status and event closure, in either arrival order.
/// Polling events while the RPC is pending returns bounded channel credits.
async fn drain_generation<R, F, C>(
    events: &mut R,
    rpc: F,
    cancel_rpc: C,
    cancellation: &CancellationToken,
    on_event: &mut impl FnMut(GenerateEvent) -> Result<()>,
) -> Result<GenerateResult>
where
    R: EventSource,
    F: Future<Output = Result<GenerateResult>>,
    C: Future<Output = Result<()>>,
{
    tokio::pin!(rpc, cancel_rpc);
    let mut result = None;
    let mut events_closed = false;
    let mut cancel_sent = false;
    let mut deadline = None;
    loop {
        if events_closed && let Some(result) = result.take() {
            cancellation.bail_if_cancelled()?;
            return result;
        }
        // The deadline only starts after one terminal signal, or an explicit
        // cancellation. Normal model loading/generation is not capped here.
        let until = deadline
            .unwrap_or_else(|| tokio::time::Instant::now() + Duration::from_secs(24 * 60 * 60));
        tokio::select! {
            reply = &mut rpc, if result.is_none() => {
                result = Some(reply);
                deadline.get_or_insert_with(|| tokio::time::Instant::now() + TERMINAL_DRAIN_TIMEOUT);
            }
            event = events.next_event(), if !events_closed => {
                match event? {
                    Some(event) => on_event(event)?,
                    None => {
                        events_closed = true;
                        deadline.get_or_insert_with(|| tokio::time::Instant::now() + TERMINAL_DRAIN_TIMEOUT);
                    }
                }
            }
            _ = wait_cancelled(cancellation), if !cancel_sent => {
                cancel_sent = true;
                tokio::time::timeout(CANCEL_SEND_TIMEOUT, &mut cancel_rpc)
                    .await
                    .wrap_err("resident cancellation channel did not become ready")??;
                deadline.get_or_insert_with(|| tokio::time::Instant::now() + TERMINAL_DRAIN_TIMEOUT);
            }
            _ = tokio::time::sleep_until(until), if deadline.is_some() => {
                cancellation.bail_if_cancelled()?;
                if let Some(Err(error)) = result {
                    return Err(error.wrap_err("resident event stream did not close after the failed RPC"));
                }
                bail!("resident generation did not deliver both its terminal result and event closure");
            }
        }
    }
}

pub(super) async fn generate_resident(
    address: &str,
    request: GenerateRequest,
    cancellation: &CancellationToken,
    on_event: &mut impl FnMut(GenerateEvent) -> Result<()>,
) -> Result<GenerateResult> {
    crate::cli::serve::validate_local_address(address)?;
    cancellation.bail_if_cancelled()?;
    let connection = tokio::select! {
        connection = vox::connect(address).channel_capacity(EVENT_CREDIT).connect_timeout(CONNECT_TIMEOUT) => {
            connection.wrap_err("cannot connect to resident local service; no cold fallback was attempted")?
        }
        _ = wait_cancelled(cancellation) => {
            cancellation.bail_if_cancelled()?;
            unreachable!("cancelled token")
        }
    };
    let result = async {
        let client: TeamyLlmClient = tokio::time::timeout(CONNECT_TIMEOUT, connection.open_lane())
            .await
            .wrap_err("resident service lane negotiation timed out")?
            .wrap_err("cannot open resident service lane")?;
        cancellation.bail_if_cancelled()?;
        let (event_tx, event_rx) = vox::channel::<GenerateEvent>();
        let (cancel_tx, cancel_rx) = vox::channel::<()>();
        let rpc = async {
            client
                .generate_cancellable(request, event_tx, cancel_rx)
                .await
                .map_err(|error| eyre::eyre!("resident generation failed: {error}"))
        };
        let cancel_rpc = async {
            cancel_tx
                .send(())
                .await
                .wrap_err("cannot request resident generation cancellation")
        };
        // Retain the paired cancellation sender until the complete event drain.
        drain_generation(
            &mut VoxEvents(event_rx),
            rpc,
            cancel_rpc,
            cancellation,
            on_event,
        )
        .await
    }
    .await;
    let shutdown = connection.shutdown();
    match result {
        Ok(result) => {
            shutdown.wrap_err("cannot close resident service connection")?;
            Ok(result)
        }
        Err(error) => {
            // Connection teardown also cancels a job if event delivery failed.
            if let Err(shutdown_error) = shutdown {
                tracing::debug!(%shutdown_error, "resident connection was already unavailable during cleanup");
            }
            Err(error)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestEvents(tokio::sync::mpsc::Receiver<GenerateEvent>);
    impl EventSource for TestEvents {
        async fn next_event(&mut self) -> Result<Option<GenerateEvent>> {
            Ok(self.0.recv().await)
        }
    }

    fn result() -> GenerateResult {
        GenerateResult {
            rendered_prompt: "prompt".into(),
            generated_token_ids: vec![1, 2],
            generated_text: "café".into(),
        }
    }

    #[test]
    fn owned_event_survives_selfref_backing_drop_without_reborrow() {
        let event = GenerateEvent::Token {
            token_id: 7,
            text: "café".into(),
        };
        let wrapped = vox::SelfRef::owning(
            vox::Backing::Boxed(vec![1, 2, 3].into_boxed_slice()),
            event.clone(),
        );
        assert_eq!(take_owned_event(wrapped), event);
    }

    #[tokio::test]
    async fn an_early_rpc_reply_does_not_truncate_pending_tokens() {
        let (sender, receiver) = tokio::sync::mpsc::channel(1);
        let producer = tokio::spawn(async move {
            for text in ["ca", "fé"] {
                sender
                    .send(GenerateEvent::Token {
                        token_id: 1,
                        text: text.into(),
                    })
                    .await
                    .unwrap();
            }
        });
        let mut text = String::new();
        let generated = drain_generation(
            &mut TestEvents(receiver),
            async { Ok(result()) },
            async { Ok(()) },
            &CancellationToken::new(),
            &mut |event| {
                if let GenerateEvent::Token { text: token, .. } = event {
                    text.push_str(&token);
                }
                Ok(())
            },
        )
        .await
        .unwrap();
        producer.await.unwrap();
        assert_eq!(text, generated.generated_text);
    }

    #[tokio::test]
    async fn event_closure_before_rpc_reply_still_waits_for_result() {
        let (sender, receiver) = tokio::sync::mpsc::channel(1);
        drop(sender);
        let generated = drain_generation(
            &mut TestEvents(receiver),
            async {
                tokio::task::yield_now().await;
                Ok(result())
            },
            async { Ok(()) },
            &CancellationToken::new(),
            &mut |_| Ok(()),
        )
        .await
        .unwrap();
        assert_eq!(generated.generated_text, "café");
    }

    #[tokio::test]
    async fn output_failure_aborts_instead_of_reporting_success() {
        let (sender, receiver) = tokio::sync::mpsc::channel(1);
        sender
            .send(GenerateEvent::Token {
                token_id: 1,
                text: "x".into(),
            })
            .await
            .unwrap();
        drop(sender);
        let failed = drain_generation(
            &mut TestEvents(receiver),
            async { Ok(result()) },
            async { Ok(()) },
            &CancellationToken::new(),
            &mut |_| Err(eyre::eyre!("sink failed")),
        )
        .await
        .unwrap_err();
        assert!(failed.to_string().contains("sink failed"));
    }

    #[tokio::test]
    async fn cancellation_is_sent_and_a_successful_reply_does_not_mask_it() {
        let (sender, receiver) = tokio::sync::mpsc::channel(1);
        drop(sender);
        let (cancel_sent, cancel_received) = tokio::sync::oneshot::channel();
        let cancellation = CancellationToken::new();
        cancellation.request_cancel("test cancellation");
        let error = drain_generation(
            &mut TestEvents(receiver),
            async {
                cancel_received.await.unwrap();
                Ok(result())
            },
            async {
                cancel_sent.send(()).unwrap();
                Ok(())
            },
            &cancellation,
            &mut |_| Ok(()),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().to_lowercase().contains("cancel"));
    }
}
