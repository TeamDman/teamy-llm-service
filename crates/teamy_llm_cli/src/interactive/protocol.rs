//! Private child-process transport. Stdout carries bounded versioned NDJSON only.
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

use eyre::{Result, WrapErr, ensure};
use facet::Facet;
use teamy_llm_core::GenerateRequest;

use crate::cli::decide::{DecideCli, DecisionDevice, DecisionProvider};

/// Maximum UTF-8 JSON payload; the terminating LF is not part of this limit.
pub const MAX_FRAME_BYTES: usize = 2 * 1024 * 1024;
pub const PROTOCOL_VERSION: u32 = 1;

/// Inference options on the child-process wire. CLI `PathBuf` fields are opaque
/// to Facet JSON; explicit UTF-8 strings preserve their exact lexical spelling.
#[derive(Clone, Debug, PartialEq, Eq, Facet)]
#[facet(rename_all = "kebab-case", deny_unknown_fields)]
pub struct DecisionConfig {
    pub request: Option<String>,
    pub provider: DecisionProvider,
    pub model: Option<String>,
    pub model_dir: String,
    pub device: DecisionDevice,
    pub runtime_library: Option<String>,
    pub question: Option<String>,
    pub repeat: usize,
    pub include_encoding: bool,
    pub max_tokens: usize,
    pub head_tokens: usize,
    pub allow_cpu_fallback: bool,
    pub profile_prefix: Option<String>,
}

impl TryFrom<&DecideCli> for DecisionConfig {
    type Error = eyre::Report;

    fn try_from(options: &DecideCli) -> Result<Self> {
        options.validate()?;
        Ok(Self {
            request: options.request.clone(),
            provider: options.provider,
            model: options.model.clone(),
            model_dir: path_string(&options.model_dir, "model-dir")?,
            device: options.device,
            runtime_library: options
                .runtime_library
                .as_deref()
                .map(|path| path_string(path, "runtime-library"))
                .transpose()?,
            question: options.question.clone(),
            repeat: options.repeat,
            include_encoding: options.include_encoding,
            max_tokens: options.max_tokens,
            head_tokens: options.head_tokens,
            allow_cpu_fallback: options.allow_cpu_fallback,
            profile_prefix: options
                .profile_prefix
                .as_deref()
                .map(|path| path_string(path, "profile-prefix"))
                .transpose()?,
        })
    }
}

impl DecisionConfig {
    /// Restore the CLI's qualified options without normalization or lossy paths.
    pub fn into_cli(self) -> Result<DecideCli> {
        let options = DecideCli {
            request: self.request,
            provider: self.provider,
            model: self.model,
            model_dir: string_path(self.model_dir, "model-dir")?,
            device: self.device,
            runtime_library: self
                .runtime_library
                .map(|path| string_path(path, "runtime-library"))
                .transpose()?,
            question: self.question,
            repeat: self.repeat,
            include_encoding: self.include_encoding,
            max_tokens: self.max_tokens,
            head_tokens: self.head_tokens,
            allow_cpu_fallback: self.allow_cpu_fallback,
            profile_prefix: self
                .profile_prefix
                .map(|path| string_path(path, "profile-prefix"))
                .transpose()?,
        };
        options.validate()?;
        Ok(options)
    }
}

fn path_string(path: &Path, field: &str) -> Result<String> {
    let value = path
        .to_str()
        .ok_or_else(|| eyre::eyre!("interactive {field} must be a valid UTF-8 path"))?;
    ensure!(
        !value.contains('\0'),
        "interactive {field} must not contain NUL"
    );
    Ok(value.into())
}

fn string_path(value: String, field: &str) -> Result<PathBuf> {
    ensure!(
        !value.contains('\0'),
        "interactive {field} must not contain NUL"
    );
    Ok(value.into())
}

#[derive(Clone, Debug, PartialEq, Eq, Facet)]
#[facet(rename_all = "kebab-case", deny_unknown_fields)]
pub struct WorkerCommand {
    pub version: u32,
    /// Zero identifies control frames; inference requests use nonzero IDs.
    pub id: u64,
    pub command: WorkerAction,
}

#[repr(u8)]
#[derive(Clone, Debug, PartialEq, Eq, Facet)]
#[facet(rename_all = "kebab-case", deny_unknown_fields)]
pub enum WorkerAction {
    Prompt {
        request: GenerateRequest,
    },
    Decide {
        config: DecisionConfig,
        request_json: String,
    },
    Cancel {
        request_id: u64,
    },
    Shutdown,
}

#[derive(Clone, Debug, PartialEq, Facet)]
#[facet(rename_all = "kebab-case", deny_unknown_fields)]
pub struct WorkerReply {
    pub version: u32,
    pub id: u64,
    pub event: WorkerEvent,
}

#[repr(u8)]
#[derive(Clone, Debug, PartialEq, Facet)]
#[facet(rename_all = "kebab-case", deny_unknown_fields)]
pub enum WorkerEvent {
    Ready,
    Started {
        mode: String,
    },
    Token {
        text: String,
    },
    Completed {
        result_json: String,
        elapsed_ms: f64,
    },
    Failed {
        message: String,
        cancelled: bool,
    },
}

impl WorkerCommand {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.version == PROTOCOL_VERSION,
            "unsupported worker command version {}; expected {PROTOCOL_VERSION}",
            self.version
        );
        match &self.command {
            WorkerAction::Prompt { .. } | WorkerAction::Decide { .. } => {
                ensure!(self.id != 0, "worker inference request IDs must be nonzero");
            }
            WorkerAction::Cancel { request_id } => {
                ensure!(self.id == 0, "worker control frames require ID zero");
                ensure!(
                    *request_id != 0,
                    "cancellation requires a nonzero request ID"
                );
            }
            WorkerAction::Shutdown => {
                ensure!(self.id == 0, "worker control frames require ID zero");
            }
        }
        Ok(())
    }
}

impl WorkerReply {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.version == PROTOCOL_VERSION,
            "unsupported worker reply version {}; expected {PROTOCOL_VERSION}",
            self.version
        );
        if matches!(self.event, WorkerEvent::Ready) {
            ensure!(self.id == 0, "worker ready handshake requires ID zero");
        } else {
            ensure!(self.id != 0, "worker inference replies require nonzero IDs");
        }
        if let WorkerEvent::Completed { elapsed_ms, .. } = &self.event {
            ensure!(
                elapsed_ms.is_finite() && *elapsed_ms >= 0.0,
                "worker elapsed time must be finite and nonnegative"
            );
        }
        Ok(())
    }
}

/// Accept one payload, with an optional terminal LF or CRLF.
pub fn parse_command(bytes: &[u8]) -> Result<WorkerCommand> {
    let command: WorkerCommand =
        facet_json::from_slice(payload(bytes)?).wrap_err("invalid worker command")?;
    command.validate()?;
    Ok(command)
}

pub fn parse_reply(bytes: &[u8]) -> Result<WorkerReply> {
    let reply: WorkerReply =
        facet_json::from_slice(payload(bytes)?).wrap_err("invalid worker reply")?;
    reply.validate()?;
    Ok(reply)
}

/// Return one complete NDJSON frame, including its terminating LF.
pub fn encode_command(command: &WorkerCommand) -> Result<Vec<u8>> {
    command.validate()?;
    encode(command)
}

pub fn encode_reply(reply: &WorkerReply) -> Result<Vec<u8>> {
    reply.validate()?;
    encode(reply)
}

fn payload(bytes: &[u8]) -> Result<&[u8]> {
    let bytes = bytes.strip_suffix(b"\n").unwrap_or(bytes);
    let bytes = bytes.strip_suffix(b"\r").unwrap_or(bytes);
    ensure!(
        bytes.len() <= MAX_FRAME_BYTES,
        "worker frame exceeds {MAX_FRAME_BYTES} bytes"
    );
    ensure!(!bytes.is_empty(), "empty worker frame");
    ensure!(
        !bytes.contains(&b'\n') && !bytes.contains(&b'\r'),
        "worker frame contains more than one line"
    );
    Ok(bytes)
}

/// Bound serialization before allocating a larger payload or touching stdout.
fn encode<'facet, T: Facet<'facet>>(value: &T) -> Result<Vec<u8>> {
    let mut writer = FrameBuffer(Vec::with_capacity(1024));
    facet_json::to_writer_std(&mut writer, value).wrap_err("cannot encode bounded worker frame")?;
    ensure!(
        !writer.0.contains(&b'\n') && !writer.0.contains(&b'\r'),
        "encoded worker frame contains an unescaped newline"
    );
    writer.0.push(b'\n');
    Ok(writer.0)
}

struct FrameBuffer(Vec<u8>);
impl Write for FrameBuffer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > MAX_FRAME_BYTES.saturating_sub(self.0.len()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "worker frame exceeds 2 MiB limit",
            ));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Blocking reader used on a dedicated input thread. Never calls unbounded `read_line`.
pub fn read_frame(reader: &mut impl BufRead) -> Result<Option<Vec<u8>>> {
    read_frame_with_limit(reader, MAX_FRAME_BYTES)
}

fn read_frame_with_limit(reader: &mut impl BufRead, limit: usize) -> Result<Option<Vec<u8>>> {
    let mut frame = Vec::with_capacity(1024.min(limit));
    loop {
        let buffer = reader.fill_buf().wrap_err("cannot read worker frame")?;
        if buffer.is_empty() {
            ensure!(frame.is_empty(), "worker input ended inside a frame");
            return Ok(None);
        }
        let newline = buffer.iter().position(|byte| *byte == b'\n');
        let count = newline.unwrap_or(buffer.len());
        ensure!(
            count <= limit.saturating_sub(frame.len()),
            "worker frame exceeds {limit} bytes"
        );
        frame.extend_from_slice(&buffer[..count]);
        reader.consume(count + usize::from(newline.is_some()));
        if newline.is_some() {
            return Ok(Some(frame));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufReader, Cursor};
    use teamy_llm_core::{ModelSelector, ThinkingMode};

    fn prompt() -> WorkerCommand {
        WorkerCommand {
            version: 1,
            id: 7,
            command: WorkerAction::Prompt {
                request: GenerateRequest {
                    selector: ModelSelector::managed("example-model"),
                    system_prompt: None,
                    user_prompt: "line one\nline two\r\n\"quoted\"".into(),
                    thinking_mode: ThinkingMode::Direct,
                    max_new_tokens: 4,
                    timeout_ms: None,
                },
            },
        }
    }

    fn decision_options(provider: DecisionProvider) -> DecideCli {
        DecideCli {
            request: Some(r#"{"version":1}"#.into()),
            provider,
            model: Some("julia-1".into()),
            model_dir: PathBuf::from(r"fixtures\Julia café 日本語"),
            device: DecisionDevice::Cuda,
            runtime_library: (provider == DecisionProvider::Onnx)
                .then(|| PathBuf::from(r"fixtures\CUDA runtime 日本語\onnxruntime.dll")),
            question: Some("Which café option should be chosen?".into()),
            repeat: 2,
            include_encoding: true,
            max_tokens: 1024,
            head_tokens: 256,
            allow_cpu_fallback: provider == DecisionProvider::Onnx,
            profile_prefix: (provider == DecisionProvider::Onnx)
                .then(|| PathBuf::from(r"fixtures\profiles\café session")),
        }
    }

    #[test]
    fn decision_transport_round_trips_every_option_and_exact_unicode_paths() {
        for provider in [DecisionProvider::Native, DecisionProvider::Onnx] {
            let options = decision_options(provider);
            let config = DecisionConfig::try_from(&options).unwrap();
            assert_eq!(config.model_dir, r"fixtures\Julia café 日本語");
            let command = WorkerCommand {
                version: PROTOCOL_VERSION,
                id: 2,
                command: WorkerAction::Decide {
                    config,
                    request_json: "{\"text\":\"café 日本語\"}\n".into(),
                },
            };
            let encoded = encode_command(&command).unwrap();
            assert_eq!(encoded.iter().filter(|byte| **byte == b'\n').count(), 1);
            let parsed = parse_command(&encoded).unwrap();
            assert_eq!(parsed, command);
            let WorkerAction::Decide {
                config,
                request_json,
            } = parsed.command
            else {
                panic!("expected decision command")
            };
            assert_eq!(request_json, "{\"text\":\"café 日本語\"}\n");
            assert_eq!(config.into_cli().unwrap(), options);
        }
    }

    #[test]
    fn decision_transport_rejects_unknown_configuration_fields() {
        let command = WorkerCommand {
            version: PROTOCOL_VERSION,
            id: 2,
            command: WorkerAction::Decide {
                config: DecisionConfig::try_from(&decision_options(DecisionProvider::Native))
                    .unwrap(),
                request_json: "{}".into(),
            },
        };
        let encoded = String::from_utf8(encode_command(&command).unwrap()).unwrap();
        assert!(encoded.contains("\"model-dir\":"));
        assert!(encoded.contains("\"runtime-library\":"));
        let unknown = encoded.replace(
            "\"model-dir\":",
            "\"model_dir\":\"ignored typo\",\"model-dir\":",
        );
        assert!(parse_command(unknown.as_bytes()).is_err());
    }

    #[test]
    fn decision_transport_rejects_nul_paths_before_runtime_construction() {
        let mut options = decision_options(DecisionProvider::Native);
        options.model_dir = PathBuf::from("fixture\0model");
        assert!(DecisionConfig::try_from(&options).is_err());
        let mut config =
            DecisionConfig::try_from(&decision_options(DecisionProvider::Onnx)).unwrap();
        config.runtime_library = Some("fixture\0runtime.dll".into());
        assert!(config.into_cli().is_err());
    }

    #[cfg(any(unix, windows))]
    #[test]
    fn decision_transport_rejects_non_utf8_paths_without_replacement_characters() {
        #[cfg(unix)]
        let invalid = {
            use std::os::unix::ffi::OsStringExt;
            PathBuf::from(std::ffi::OsString::from_vec(vec![0xff]))
        };
        #[cfg(windows)]
        let invalid = {
            use std::os::windows::ffi::OsStringExt;
            PathBuf::from(std::ffi::OsString::from_wide(&[0xd800]))
        };
        let mut options = decision_options(DecisionProvider::Native);
        options.model_dir = invalid.clone();
        assert!(DecisionConfig::try_from(&options).is_err());
        let mut options = decision_options(DecisionProvider::Onnx);
        options.runtime_library = Some(invalid.clone());
        assert!(DecisionConfig::try_from(&options).is_err());
        options.runtime_library = Some(PathBuf::from("fixture/runtime.dll"));
        options.profile_prefix = Some(invalid);
        assert!(DecisionConfig::try_from(&options).is_err());
    }

    #[test]
    fn frames_round_trip_without_splitting_user_text() {
        let command = prompt();
        let encoded = encode_command(&command).unwrap();
        assert_eq!(encoded.iter().filter(|byte| **byte == b'\n').count(), 1);
        assert_eq!(parse_command(&encoded).unwrap(), command);
        let replies = [
            WorkerReply {
                version: 1,
                id: 0,
                event: WorkerEvent::Ready,
            },
            WorkerReply {
                version: 1,
                id: 7,
                event: WorkerEvent::Started {
                    mode: "prompt".into(),
                },
            },
            WorkerReply {
                version: 1,
                id: 7,
                event: WorkerEvent::Token {
                    text: "hi\nthere".into(),
                },
            },
            WorkerReply {
                version: 1,
                id: 7,
                event: WorkerEvent::Completed {
                    result_json: "{\"answer\":1}".into(),
                    elapsed_ms: 2.5,
                },
            },
            WorkerReply {
                version: 1,
                id: 7,
                event: WorkerEvent::Failed {
                    message: "cancelled".into(),
                    cancelled: true,
                },
            },
        ];
        for reply in replies {
            assert_eq!(parse_reply(&encode_reply(&reply).unwrap()).unwrap(), reply);
        }
    }

    #[test]
    fn rejects_schema_drift_and_control_id_confusion() {
        let encoded = String::from_utf8(encode_command(&prompt()).unwrap()).unwrap();
        assert!(
            parse_command(encoded.replace("\"version\":1", "\"version\":2").as_bytes()).is_err()
        );
        assert!(
            parse_command(
                encoded
                    .replace("\"version\":1", "\"typo\":true,\"version\":1")
                    .as_bytes()
            )
            .is_err()
        );
        assert!(parse_command(&[0xff]).is_err());
        assert!(parse_command(b"{}\n{}\n").is_err());
        let mut command = prompt();
        command.id = 0;
        assert!(encode_command(&command).is_err());
        command.command = WorkerAction::Cancel { request_id: 0 };
        assert!(encode_command(&command).is_err());
        command.command = WorkerAction::Shutdown;
        assert!(encode_command(&command).is_ok());
        command.id = 7;
        assert!(encode_command(&command).is_err());
    }

    #[test]
    fn enforces_limits_before_unbounded_input_or_output_growth() {
        let mut reader = BufReader::with_capacity(2, Cursor::new(b"abc\ndef\n"));
        assert_eq!(
            read_frame_with_limit(&mut reader, 3).unwrap(),
            Some(b"abc".to_vec())
        );
        assert_eq!(
            read_frame_with_limit(&mut reader, 3).unwrap(),
            Some(b"def".to_vec())
        );
        assert!(read_frame_with_limit(&mut reader, 3).unwrap().is_none());
        assert!(read_frame_with_limit(&mut Cursor::new(b"abcd\n"), 3).is_err());
        assert!(read_frame_with_limit(&mut Cursor::new(b"abc"), 3).is_err());
        assert!(parse_reply(&vec![b' '; MAX_FRAME_BYTES + 1]).is_err());
        let reply = WorkerReply {
            version: 1,
            id: 7,
            event: WorkerEvent::Token {
                text: "x".repeat(MAX_FRAME_BYTES),
            },
        };
        assert!(encode_reply(&reply).is_err());
        let mut buffer = FrameBuffer(Vec::new());
        assert!(buffer.write_all(&vec![0; MAX_FRAME_BYTES + 1]).is_err());
        assert!(buffer.0.is_empty());
    }
}
