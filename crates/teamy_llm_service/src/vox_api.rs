//! Resident RPC generation with bounded streaming and per-call cancellation.
//!
//! Native inference is synchronous. It must run off the Vox driver task so
//! that channel credits, client cancellation and unrelated calls can progress.
use std::future::Future;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Duration;

use teamy_cancellation::CancellationToken;
use tokio::sync::{Notify, mpsc};

use super::{
    GenerateEvent, GenerateRequest, GenerateResult, ModelReport, ModelSelector, ModelStatus,
    PrepareEvent, PrepareModelRequest, PrepareModelResult, ServiceError, ServiceErrorKind,
    ServiceHealth, TeamyLlmService,
};

const EVENT_BRIDGE_CAPACITY: usize = 16;

#[vox::service]
pub trait TeamyLlm {
    async fn health(&self) -> ServiceHealth;
    async fn list_models(&self) -> Result<Vec<ModelStatus>, ServiceError>;
    async fn inspect_model(&self, selector: ModelSelector) -> Result<ModelReport, ServiceError>;
    async fn prepare_model(
        &self,
        request: PrepareModelRequest,
        events: vox::Tx<PrepareEvent>,
    ) -> Result<PrepareModelResult, ServiceError>;
    /// Legacy generation signature. Event delivery failure and RPC teardown
    /// still cancel its inference; explicit control uses generate_cancellable.
    async fn generate(
        &self,
        request: GenerateRequest,
        events: vox::Tx<GenerateEvent>,
    ) -> Result<GenerateResult, ServiceError>;
    /// Keep the paired cancellation Tx alive for the entire call. Sending ()
    /// or closing that channel requests cancellation of this call only.
    async fn generate_cancellable(
        &self,
        request: GenerateRequest,
        events: vox::Tx<GenerateEvent>,
        cancellation: vox::Rx<()>,
    ) -> Result<GenerateResult, ServiceError>;
}

impl TeamyLlm for TeamyLlmService {
    async fn health(&self) -> ServiceHealth {
        TeamyLlmService::health(self)
    }

    async fn list_models(&self) -> Result<Vec<ModelStatus>, ServiceError> {
        TeamyLlmService::list_models(self)
    }

    async fn inspect_model(&self, selector: ModelSelector) -> Result<ModelReport, ServiceError> {
        TeamyLlmService::inspect_model(self, selector)
    }

    async fn prepare_model(
        &self,
        request: PrepareModelRequest,
        events: vox::Tx<PrepareEvent>,
    ) -> Result<PrepareModelResult, ServiceError> {
        TeamyLlmService::prepare_model(self, request, |event| {
            events.try_send(event).map_err(event_delivery_error)
        })
        .await
    }

    async fn generate(
        &self,
        request: GenerateRequest,
        events: vox::Tx<GenerateEvent>,
    ) -> Result<GenerateResult, ServiceError> {
        generate_over_vox(self.clone(), request, events, None, None, None).await
    }

    async fn generate_cancellable(
        &self,
        request: GenerateRequest,
        events: vox::Tx<GenerateEvent>,
        cancellation: vox::Rx<()>,
    ) -> Result<GenerateResult, ServiceError> {
        generate_over_vox(
            self.clone(),
            request,
            events,
            Some(cancellation),
            None,
            None,
        )
        .await
    }
}

pub async fn serve_local(address: &str, service: TeamyLlmService) -> Result<(), vox::ServeError> {
    serve_local_with_cancellation(address, service, CancellationToken::new()).await
}

/// Stop accepting connections and cancel/drain active generation when requested.
/// Vox's high-level listener detaches accepted connections; dropping only that
/// listener does not stop their handlers. This dispatcher carries the server's
/// lifetime token across those connections and rejects generation after shutdown.
pub async fn serve_local_with_cancellation(
    address: &str,
    service: TeamyLlmService,
    shutdown: CancellationToken,
) -> Result<(), vox::ServeError> {
    let active = Arc::new(ActiveGenerations::default());
    let _shutdown_on_drop = ServerShutdownOnDrop(shutdown.clone());
    if shutdown.is_cancelled() {
        return Ok(());
    }
    let dispatcher = TeamyLlmDispatcher::new(ServingTeamyLlm {
        service,
        shutdown: shutdown.clone(),
        active: active.clone(),
    });
    let result = tokio::select! {
        result = vox::serve(address, dispatcher).run() => result,
        _ = wait_shutdown(&shutdown) => Ok(()),
    };
    shutdown.request_cancel("local inference server stopped");
    active.drain().await;
    result
}

#[derive(Clone)]
struct ServingTeamyLlm {
    service: TeamyLlmService,
    shutdown: CancellationToken,
    active: Arc<ActiveGenerations>,
}

impl TeamyLlm for ServingTeamyLlm {
    async fn health(&self) -> ServiceHealth {
        let mut health = self.service.health();
        health.ready &= !self.shutdown.is_cancelled();
        health
    }

    async fn list_models(&self) -> Result<Vec<ModelStatus>, ServiceError> {
        self.service.list_models()
    }

    async fn inspect_model(&self, selector: ModelSelector) -> Result<ModelReport, ServiceError> {
        self.service.inspect_model(selector)
    }

    async fn prepare_model(
        &self,
        request: PrepareModelRequest,
        events: vox::Tx<PrepareEvent>,
    ) -> Result<PrepareModelResult, ServiceError> {
        server_accepts_work(&self.shutdown)?;
        TeamyLlm::prepare_model(&self.service, request, events).await
    }

    async fn generate(
        &self,
        request: GenerateRequest,
        events: vox::Tx<GenerateEvent>,
    ) -> Result<GenerateResult, ServiceError> {
        let active = self.active.begin(&self.shutdown)?;
        generate_over_vox(
            self.service.clone(),
            request,
            events,
            None,
            Some(self.shutdown.clone()),
            Some(active),
        )
        .await
    }

    async fn generate_cancellable(
        &self,
        request: GenerateRequest,
        events: vox::Tx<GenerateEvent>,
        cancellation: vox::Rx<()>,
    ) -> Result<GenerateResult, ServiceError> {
        let active = self.active.begin(&self.shutdown)?;
        generate_over_vox(
            self.service.clone(),
            request,
            events,
            Some(cancellation),
            Some(self.shutdown.clone()),
            Some(active),
        )
        .await
    }
}

#[derive(Default)]
struct ActiveGenerations {
    count: AtomicUsize,
    changed: Notify,
}

impl ActiveGenerations {
    fn begin(
        self: &Arc<Self>,
        shutdown: &CancellationToken,
    ) -> Result<ActiveGeneration, ServiceError> {
        server_accepts_work(shutdown)?;
        self.count.fetch_add(1, Ordering::SeqCst);
        let active = ActiveGeneration(self.clone());
        // If shutdown raced registration, release the count and refuse the job.
        server_accepts_work(shutdown)?;
        Ok(active)
    }

    async fn drain(&self) {
        loop {
            let notified = self.changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.count.load(Ordering::SeqCst) == 0 {
                return;
            }
            notified.await;
        }
    }
}

struct ActiveGeneration(Arc<ActiveGenerations>);

impl Drop for ActiveGeneration {
    fn drop(&mut self) {
        self.0.count.fetch_sub(1, Ordering::SeqCst);
        self.0.changed.notify_waiters();
    }
}

struct ServerShutdownOnDrop(CancellationToken);

impl Drop for ServerShutdownOnDrop {
    fn drop(&mut self) {
        self.0
            .request_cancel("local inference server future dropped");
    }
}

fn server_accepts_work(shutdown: &CancellationToken) -> Result<(), ServiceError> {
    if shutdown.is_cancelled() {
        return Err(ServiceError::new(
            ServiceErrorKind::Cancelled,
            "local inference server is shutting down",
        ));
    }
    Ok(())
}

async fn wait_shutdown(shutdown: &CancellationToken) {
    while !shutdown.is_cancelled() {
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

async fn generate_over_vox(
    service: TeamyLlmService,
    request: GenerateRequest,
    events: vox::Tx<GenerateEvent>,
    cancellation: Option<vox::Rx<()>>,
    shutdown: Option<CancellationToken>,
    active: Option<ActiveGeneration>,
) -> Result<GenerateResult, ServiceError> {
    let result = run_generation_job(
        move |token, sender| {
            // The count follows the blocking job, including when Vox aborts the
            // async handler and the blocking job has to finish cooperatively.
            let _active = active;
            service.generate_with_cancellation(request, Some(token), &mut |event| {
                sender.blocking_send(event).map_err(event_delivery_error)
            })
        },
        |event| events.send(event),
        cancelled_for_call(cancellation, shutdown),
    )
    .await;
    // All event sends finish before this close, which precedes the RPC result.
    // A client must still drain its receiver: delivery into Vox is not receipt
    // by the client, and channel/result messages may arrive in either poll order.
    if let Err(error) = events.close(Default::default()).await
        && result.is_ok()
    {
        return Err(event_delivery_error(error));
    }
    result
}

async fn cancelled_for_call(
    mut cancellation: Option<vox::Rx<()>>,
    shutdown: Option<CancellationToken>,
) -> String {
    let client_cancelled = async {
        let Some(receiver) = cancellation.as_mut() else {
            return std::future::pending::<String>().await;
        };
        match receiver.recv().await {
            Ok(Some(_)) => "client requested generation cancellation".into(),
            Ok(None) => "client closed the generation cancellation channel".into(),
            Err(error) => format!("generation cancellation channel disconnected: {error}"),
        }
    };
    let server_shutdown = async {
        let Some(shutdown) = shutdown.as_ref() else {
            return std::future::pending::<String>().await;
        };
        wait_shutdown(shutdown).await;
        "local inference server is shutting down".into()
    };
    tokio::select! {
        reason = client_cancelled => reason,
        reason = server_shutdown => reason,
    }
}

/// Bridge a finite blocking generation job to an async, credit-aware consumer.
/// The injectable boundaries keep cancellation/backpressure tests model-free.
async fn run_generation_job<Generate, SendEvent, SendFuture, Cancelled>(
    generate: Generate,
    mut send_event: SendEvent,
    cancelled: Cancelled,
) -> Result<GenerateResult, ServiceError>
where
    Generate: FnOnce(
            CancellationToken,
            mpsc::Sender<GenerateEvent>,
        ) -> Result<GenerateResult, ServiceError>
        + Send
        + 'static,
    SendEvent: FnMut(GenerateEvent) -> SendFuture,
    SendFuture: Future<Output = Result<(), vox::TxError>>,
    Cancelled: Future<Output = String>,
{
    let token = CancellationToken::new();
    let mut guard = CancelOnDrop {
        token: token.clone(),
        armed: true,
    };
    let (sender, mut receiver) = mpsc::channel(EVENT_BRIDGE_CAPACITY);
    let job_token = token.clone();
    let job = tokio::task::spawn_blocking(move || generate(job_token, sender));
    let delivery = async {
        while let Some(event) = receiver.recv().await {
            send_event(event).await.map_err(event_delivery_error)?;
        }
        Ok::<(), ServiceError>(())
    };
    let delivery_result = tokio::select! {
        reason = cancelled => {
            token.request_cancel(&reason);
            Err(ServiceError::new(ServiceErrorKind::Cancelled, reason))
        }
        result = delivery => result,
    };
    if delivery_result.is_err() {
        token.request_cancel("generation event consumer stopped");
        // Release a producer waiting in blocking_send; cancellation must not
        // depend on a full event queue or additional channel credit.
        receiver.close();
    }
    // A running native kernel is cooperative, not preemptible. Await this job
    // before acknowledging an explicitly cancelled call. If Vox aborts the
    // whole handler, the guard and receiver drop still signal/unstick the job.
    let result = job.await.map_err(|error| {
        ServiceError::new(
            ServiceErrorKind::Internal,
            format!("resident generation job failed: {error}"),
        )
    })?;
    guard.armed = false;
    delivery_result?;
    result
}

struct CancelOnDrop {
    token: CancellationToken,
    armed: bool,
}
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if self.armed {
            self.token
                .request_cancel("generation RPC handler disconnected");
        }
    }
}

fn event_delivery_error(error: impl std::fmt::Display) -> ServiceError {
    ServiceError::new(
        ServiceErrorKind::Cancelled,
        format!("generation event delivery stopped: {error}"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    };
    use std::time::Duration;

    fn result() -> GenerateResult {
        GenerateResult {
            rendered_prompt: "fixture".into(),
            generated_token_ids: Vec::new(),
            generated_text: "fixture answer".into(),
        }
    }
    fn event(token_id: usize) -> GenerateEvent {
        GenerateEvent::Token {
            token_id,
            text: "fixture".into(),
        }
    }
    fn cancelled(token: &CancellationToken) -> Result<(), ServiceError> {
        token.bail_if_cancelled().map_err(event_delivery_error)
    }

    #[tokio::test(flavor = "current_thread")]
    async fn more_than_initial_vox_credit_streams_in_order_without_blocking_driver() {
        let observed = Arc::new(Mutex::new(Vec::new()));
        let capture = observed.clone();
        let returned = tokio::time::timeout(
            Duration::from_secs(2),
            run_generation_job(
                |_, sender| {
                    for token_id in 0..64 {
                        sender
                            .blocking_send(event(token_id))
                            .map_err(event_delivery_error)?;
                    }
                    Ok(result())
                },
                move |event| {
                    let capture = capture.clone();
                    async move {
                        tokio::task::yield_now().await;
                        let GenerateEvent::Token { token_id, .. } = event else {
                            panic!("fixture token")
                        };
                        capture.lock().unwrap().push(token_id);
                        Ok(())
                    }
                },
                std::future::pending(),
            ),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(returned, result());
        assert_eq!(*observed.lock().unwrap(), (0..64).collect::<Vec<_>>());
    }

    #[tokio::test]
    async fn cancellation_releases_a_producer_behind_a_full_event_queue() {
        let produced = Arc::new(AtomicUsize::new(0));
        let producer = produced.clone();
        let error = tokio::time::timeout(
            Duration::from_secs(2),
            run_generation_job(
                move |token, sender| {
                    for token_id in 0..1000 {
                        cancelled(&token)?;
                        sender
                            .blocking_send(event(token_id))
                            .map_err(event_delivery_error)?;
                        producer.fetch_add(1, Ordering::SeqCst);
                    }
                    Ok(result())
                },
                |_| std::future::pending::<Result<(), vox::TxError>>(),
                async {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                    "fixture cancellation".into()
                },
            ),
        )
        .await
        .unwrap()
        .unwrap_err();
        assert_eq!(error.kind, ServiceErrorKind::Cancelled);
        assert!(produced.load(Ordering::SeqCst) <= EVENT_BRIDGE_CAPACITY + 1);
    }

    #[tokio::test]
    async fn event_delivery_failure_cancels_and_joins_the_job() {
        let finished = Arc::new(AtomicBool::new(false));
        let job_finished = finished.clone();
        let error = tokio::time::timeout(
            Duration::from_secs(2),
            run_generation_job(
                move |token, sender| {
                    sender
                        .blocking_send(event(0))
                        .map_err(event_delivery_error)?;
                    while !token.is_cancelled() {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    job_finished.store(true, Ordering::SeqCst);
                    cancelled(&token)?;
                    Ok(result())
                },
                |_| std::future::ready(Err(vox::TxError::Unbound)),
                std::future::pending(),
            ),
        )
        .await
        .unwrap()
        .unwrap_err();
        assert_eq!(error.kind, ServiceErrorKind::Cancelled);
        assert!(finished.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn dropping_the_rpc_handler_cancels_its_detached_blocking_job() {
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (finished_tx, finished_rx) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(run_generation_job(
            move |token, _sender| {
                started_tx.send(()).unwrap();
                while !token.is_cancelled() {
                    std::thread::sleep(Duration::from_millis(1));
                }
                finished_tx.send(()).unwrap();
                cancelled(&token)?;
                Ok(result())
            },
            |_| std::future::ready(Ok(())),
            std::future::pending(),
        ));
        tokio::time::timeout(Duration::from_secs(2), started_rx)
            .await
            .unwrap()
            .unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        tokio::time::timeout(Duration::from_secs(2), finished_rx)
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn backend_failure_keeps_its_domain_error_kind() {
        let error = run_generation_job(
            |_, _| {
                Err(ServiceError::new(
                    ServiceErrorKind::InvalidRequest,
                    "fixture invalid prompt",
                ))
            },
            |_| std::future::ready(Ok(())),
            std::future::pending(),
        )
        .await
        .unwrap_err();
        assert_eq!(error.kind, ServiceErrorKind::InvalidRequest);
        assert_eq!(error.message, "fixture invalid prompt");
    }

    #[tokio::test]
    async fn server_shutdown_cancels_a_job_without_client_cancellation() {
        let shutdown = CancellationToken::new();
        let signal = shutdown.clone();
        let error = tokio::time::timeout(Duration::from_secs(2), async move {
            tokio::join!(
                run_generation_job(
                    |token, _sender| {
                        while !token.is_cancelled() {
                            std::thread::sleep(Duration::from_millis(1));
                        }
                        cancelled(&token)?;
                        Ok(result())
                    },
                    |_| std::future::ready(Ok(())),
                    cancelled_for_call(None, Some(shutdown)),
                ),
                async move {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                    signal.request_cancel("fixture server stopped");
                }
            )
            .0
        })
        .await
        .unwrap()
        .unwrap_err();
        assert_eq!(error.kind, ServiceErrorKind::Cancelled);
        assert_eq!(error.message, "local inference server is shutting down");
    }

    #[tokio::test]
    async fn shutdown_drain_follows_blocking_job_after_handler_abort() {
        let active = Arc::new(ActiveGenerations::default());
        let registration = active.begin(&CancellationToken::new()).unwrap();
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (permit_tx, permit_rx) = std::sync::mpsc::channel();
        let task = tokio::spawn(run_generation_job(
            move |token, _sender| {
                let _registration = registration;
                started_tx.send(()).unwrap();
                while !token.is_cancelled() {
                    std::thread::sleep(Duration::from_millis(1));
                }
                // Represent a kernel completing after cooperative cancellation.
                permit_rx.recv().unwrap();
                cancelled(&token)?;
                Ok(result())
            },
            |_| std::future::ready(Ok(())),
            std::future::pending(),
        ));
        tokio::time::timeout(Duration::from_secs(2), started_rx)
            .await
            .unwrap()
            .unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        let premature = tokio::time::timeout(Duration::from_millis(20), active.drain()).await;
        // Always release the blocking fixture, including on a failed assertion.
        permit_tx.send(()).unwrap();
        assert!(
            premature.is_err(),
            "shutdown returned before the native job finished"
        );
        tokio::time::timeout(Duration::from_secs(2), active.drain())
            .await
            .unwrap();
        assert_eq!(active.count.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn shutdown_rejects_new_generation_without_registering_work() {
        let active = Arc::new(ActiveGenerations::default());
        let shutdown = CancellationToken::new();
        shutdown.request_cancel("fixture stopped");
        let error = active.begin(&shutdown).err().unwrap();
        assert_eq!(error.kind, ServiceErrorKind::Cancelled);
        assert_eq!(active.count.load(Ordering::SeqCst), 0);
    }
}
