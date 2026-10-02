//! Parent transport: bounded frames, one active request, captured child stderr.
use super::protocol::{MAX_FRAME_BYTES, WorkerAction, WorkerCommand, WorkerEvent, WorkerReply};
use eyre::{Result, WrapErr, bail, ensure};
use std::{process::Stdio, time::Duration};
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, Command},
    sync::mpsc,
    task::JoinHandle,
};

pub(super) struct InteractiveEngine {
    child: Child,
    stdin: Option<ChildStdin>,
    replies: mpsc::Receiver<Result<WorkerReply>>,
    readers: Vec<JoinHandle<()>>,
    next_id: u64,
    active_id: Option<u64>,
    mode: Option<&'static str>,
    cancellation: teamy_cancellation::CancellationToken,
}

/// Read one frame without allowing an unterminated line to grow indefinitely.
async fn bounded_line<R: AsyncRead + Unpin>(
    reader: &mut R,
    limit: usize,
) -> Result<Option<Vec<u8>>> {
    let mut bytes = Vec::with_capacity(1024);
    loop {
        let byte = reader.read_u8().await;
        match byte {
            Ok(b'\n') => return Ok(Some(bytes)),
            Ok(byte) => {
                ensure!(bytes.len() < limit, "worker frame exceeds {limit} bytes");
                bytes.push(byte);
            }
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => {
                ensure!(bytes.is_empty(), "worker stream ended inside a frame");
                return Ok(None);
            }
            Err(error) => return Err(error.into()),
        }
    }
}

impl InteractiveEngine {
    pub(super) async fn spawn(cancellation: teamy_cancellation::CancellationToken) -> Result<Self> {
        cancellation.bail_if_cancelled()?;
        let mut command = Command::new(std::env::current_exe()?);
        command
            .arg("--internal-interactive-worker")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        command.env_remove("TEAMY_LLM_WORKER_LOG_FILE");
        command.env_remove("TEAMY_LLM_WORKER_STOP_AFTER");
        if let Some(filter) = crate::logging::worker_filter() {
            command.env("RUST_LOG", filter);
        }
        if let Some(spec) = crate::logging::worker_stop_after() {
            command.env("TEAMY_LLM_WORKER_STOP_AFTER", spec);
        }
        let mut child = command
            .spawn()
            .wrap_err("cannot start the resident inference worker")?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| eyre::eyre!("worker stdin unavailable"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| eyre::eyre!("worker stdout unavailable"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| eyre::eyre!("worker stderr unavailable"))?;
        let (sender, replies) = mpsc::channel(32);
        let reply_task = tokio::spawn(async move {
            let mut reader = BufReader::new(stdout);
            loop {
                let result = match bounded_line(&mut reader, MAX_FRAME_BYTES).await {
                    Ok(Some(bytes)) => super::protocol::parse_reply(&bytes),
                    Ok(None) => Err(eyre::eyre!("resident inference worker disconnected")),
                    Err(error) => Err(error),
                };
                let failed = result.is_err();
                if sender.send(result).await.is_err() || failed {
                    break;
                }
            }
        });
        let log_task = tokio::spawn(async move {
            // Chunk long native lines instead of stopping the pipe drain. Model
            // output is protocol data; this pipe contains diagnostics only.
            let mut reader = BufReader::new(stderr);
            let mut buffer = Vec::with_capacity(16 * 1024);
            let mut chunk = [0_u8; 4096];
            loop {
                match reader.read(&mut chunk).await {
                    Ok(0) => break,
                    Ok(count) => {
                        for byte in &chunk[..count] {
                            if (*byte == b'\n' || buffer.len() == 16 * 1024) && !buffer.is_empty() {
                                tracing::info!(target: "teamy_llm::worker", worker_log = %String::from_utf8_lossy(&buffer), "worker diagnostic");
                                buffer.clear();
                            }
                            if *byte != b'\n' {
                                buffer.push(*byte);
                            }
                        }
                    }
                    Err(error) => {
                        tracing::error!(%error, "cannot drain worker diagnostics");
                        break;
                    }
                }
            }
            if !buffer.is_empty() {
                tracing::info!(target: "teamy_llm::worker", worker_log = %String::from_utf8_lossy(&buffer), "worker diagnostic");
            }
        });
        let mut engine = Self {
            child,
            stdin: Some(stdin),
            replies,
            readers: vec![reply_task, log_task],
            next_id: 1,
            active_id: None,
            mode: None,
            cancellation: cancellation.clone(),
        };
        let ready = tokio::select! {
            _ = super::wait_cancelled(&cancellation) => { cancellation.bail_if_cancelled()?; unreachable!("cancelled token") },
            ready = tokio::time::timeout(Duration::from_secs(30), engine.next_event()) => ready.wrap_err("worker startup timed out")??,
        };
        ensure!(
            ready.id == 0 && matches!(ready.event, WorkerEvent::Ready),
            "worker did not send its ready handshake"
        );
        Ok(engine)
    }

    async fn send(&mut self, command: WorkerCommand) -> Result<()> {
        let frame = super::protocol::encode_command(&command)?;
        let inference = matches!(
            command.command,
            WorkerAction::Prompt { .. } | WorkerAction::Decide { .. }
        );
        let cancellation = self.cancellation.clone();
        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| eyre::eyre!("worker is shut down"))?;
        let write = async {
            stdin.write_all(&frame).await?;
            stdin.flush().await?;
            Ok::<(), eyre::Report>(())
        };
        tokio::select! {
            _ = super::wait_cancelled(&cancellation), if inference => { cancellation.bail_if_cancelled()?; unreachable!("cancelled token") },
            result = tokio::time::timeout(Duration::from_secs(5), write) => result.wrap_err("worker input write timed out")?,
        }
    }

    pub(super) async fn submit(&mut self, command: WorkerAction) -> Result<u64> {
        self.cancellation.bail_if_cancelled()?;
        ensure!(
            self.active_id.is_none(),
            "an inference request is already active"
        );
        ensure!(
            matches!(
                command,
                WorkerAction::Prompt { .. } | WorkerAction::Decide { .. }
            ),
            "submit expects a prompt or decision"
        );
        let mode = if matches!(command, WorkerAction::Prompt { .. }) {
            "prompt"
        } else {
            "decide"
        };
        if self.mode.is_some_and(|previous| previous != mode) {
            // Burn/CubeCL's device registry can retain allocator pools after
            // tensors drop; memory_cleanup explicitly gives no release guarantee.
            // Exit the owned process before allowing the other backend to load.
            let next_id = self.next_id;
            self.shutdown().await?;
            let mut replacement = Self::spawn(self.cancellation.clone()).await?;
            replacement.next_id = next_id;
            *self = replacement;
            tracing::debug!(mode, "restarted inference worker after mode change");
        }
        let id = self.next_id;
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or_else(|| eyre::eyre!("request IDs exhausted"))?;
        self.send(WorkerCommand {
            version: 1,
            id,
            command,
        })
        .await?;
        self.active_id = Some(id);
        self.mode = Some(mode);
        Ok(id)
    }

    pub(super) async fn cancel(&mut self, request_id: u64) -> Result<()> {
        ensure!(
            self.active_id == Some(request_id),
            "cancellation does not match the active request"
        );
        self.send(WorkerCommand {
            version: 1,
            id: 0,
            command: WorkerAction::Cancel { request_id },
        })
        .await
    }

    pub(super) async fn next_event(&mut self) -> Result<WorkerReply> {
        let reply = self
            .replies
            .recv()
            .await
            .ok_or_else(|| eyre::eyre!("worker reply channel closed"))??;
        ensure!(
            reply.version == 1,
            "unsupported worker reply version {}",
            reply.version
        );
        if matches!(reply.event, WorkerEvent::Ready) {
            ensure!(reply.id == 0, "invalid worker handshake ID");
        } else {
            ensure!(
                self.active_id == Some(reply.id),
                "unexpected worker reply ID {}",
                reply.id
            );
            if matches!(
                reply.event,
                WorkerEvent::Completed { .. } | WorkerEvent::Failed { .. }
            ) {
                self.active_id = None;
            }
        }
        Ok(reply)
    }

    pub(super) async fn shutdown(&mut self) -> Result<()> {
        if self.stdin.is_none() {
            return Ok(());
        }
        let send_result = self
            .send(WorkerCommand {
                version: 1,
                id: 0,
                command: WorkerAction::Shutdown,
            })
            .await;
        self.stdin.take();
        // Continue draining replies while waiting: the bounded token channel
        // must not hold the child alive during cooperative shutdown.
        let wait_result = tokio::time::timeout(Duration::from_secs(15), async {
            loop {
                tokio::select! {
                    status = self.child.wait() => break status,
                    _ = self.replies.recv() => {}
                }
            }
        })
        .await;
        let outcome = match wait_result {
            Ok(Ok(status)) if status.success() => send_result,
            Ok(Ok(status)) => Err(eyre::eyre!("worker exited with {status}")),
            Ok(Err(error)) => Err(error.into()),
            Err(_) => {
                self.child.kill().await?;
                bail!(
                    "worker did not finish cooperative shutdown within 15 seconds; terminated the owned child"
                );
            }
        };
        // Child exit can win the select while the bounded reply queue remains
        // full. Close its receiver before joining the producer, so a pending
        // send cannot deadlock cleanup after the timeout has ended.
        let reader_error = join_output_readers(&mut self.replies, &mut self.readers)
            .await
            .err();
        match (outcome, reader_error) {
            (Err(error), Some(cleanup)) => {
                Err(error.wrap_err(format!("worker reader cleanup also failed: {cleanup:#}")))
            }
            (Err(error), None) => Err(error),
            (Ok(()), Some(error)) => Err(error),
            (Ok(()), None) => Ok(()),
        }
    }
}

async fn join_output_readers(
    replies: &mut mpsc::Receiver<Result<WorkerReply>>,
    readers: &mut Vec<JoinHandle<()>>,
) -> Result<()> {
    replies.close();
    let mut failure = None;
    for mut reader in readers.drain(..) {
        match tokio::time::timeout(Duration::from_secs(2), &mut reader).await {
            Ok(Err(error)) => {
                failure.get_or_insert_with(|| eyre::eyre!(error));
            }
            Ok(Ok(())) => {}
            Err(_) => {
                reader.abort();
                failure.get_or_insert_with(|| {
                    eyre::eyre!("worker output reader did not finish after process exit")
                });
            }
        }
    }
    failure.map_or(Ok(()), Err)
}

impl Drop for InteractiveEngine {
    fn drop(&mut self) {
        // kill_on_drop owns only this child; cancel reader tasks on exceptional
        // parent exit so no channel or stderr task outlives the session.
        for reader in &self.readers {
            reader.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn frame_reader_rejects_oversize_and_truncated_frames() {
        assert!(bounded_line(&mut &b"abcd\n"[..], 3).await.is_err());
        assert!(bounded_line(&mut &b"abc"[..], 4).await.is_err());
        assert_eq!(
            bounded_line(&mut &b"abc\n"[..], 3).await.unwrap(),
            Some(b"abc".to_vec())
        );
        assert!(bounded_line(&mut &b""[..], 3).await.unwrap().is_none());
    }
    #[tokio::test]
    async fn closing_a_session_unblocks_a_reader_waiting_on_a_full_reply_queue() {
        let (sender, mut receiver) = mpsc::channel(1);
        sender
            .send(Ok(WorkerReply {
                version: 1,
                id: 0,
                event: WorkerEvent::Ready,
            }))
            .await
            .unwrap();
        let reader = tokio::spawn(async move {
            assert!(
                sender
                    .send(Ok(WorkerReply {
                        version: 1,
                        id: 1,
                        event: WorkerEvent::Token {
                            text: "queued".into()
                        }
                    }))
                    .await
                    .is_err()
            );
        });
        let mut readers = vec![reader];
        tokio::time::timeout(
            Duration::from_secs(1),
            join_output_readers(&mut receiver, &mut readers),
        )
        .await
        .unwrap()
        .unwrap();
        assert!(readers.is_empty());
    }
}
