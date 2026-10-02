//! Only an explicit, solitary `-` consumes stdin. Input is data, never a command.
use std::io::Read;

use eyre::{Result, WrapErr, ensure};
use teamy_cancellation::CancellationToken;

pub(super) const MAX_PROMPT_BYTES: usize = 2 * 1024 * 1024;

fn uses_stdin(arguments: &[String]) -> bool {
    arguments.len() == 1 && arguments[0] == "-"
}

fn read_prompt(reader: impl Read) -> Result<String> {
    let mut bytes = Vec::new();
    reader
        .take((MAX_PROMPT_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .wrap_err("cannot read prompt from stdin")?;
    ensure!(
        bytes.len() <= MAX_PROMPT_BYTES,
        "stdin prompt exceeds the 2 MiB limit"
    );
    String::from_utf8(bytes).wrap_err("stdin prompt must be valid UTF-8")
}

pub(super) async fn resolve_prompt(
    arguments: Vec<String>,
    cancellation: &CancellationToken,
) -> Result<String> {
    cancellation.bail_if_cancelled()?;
    if !uses_stdin(&arguments) {
        return Ok(arguments.join(" "));
    }
    // A detached OS reader lets cancellation end the CLI even if a pipe's
    // producer leaves stdin open. Tokio's blocking pool waits for its tasks at
    // shutdown, so it must not own this potentially indefinite read.
    let (sender, receiver) = tokio::sync::oneshot::channel();
    std::thread::Builder::new()
        .name("llm-prompt-stdin".into())
        .spawn(move || {
            let result = read_prompt(std::io::stdin().lock());
            let _ = sender.send(result);
        })
        .wrap_err("cannot start stdin reader")?;
    tokio::select! {
        result = receiver => result.wrap_err("stdin reader stopped unexpectedly")?,
        _ = super::streaming::wait_cancelled(cancellation) => {
            cancellation.bail_if_cancelled()?;
            unreachable!("cancellation waiter only completes for a cancelled token")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_solitary_dash_selects_stdin() {
        assert!(uses_stdin(&["-".into()]));
        for arguments in [vec![], vec!["--".into()], vec!["a".into(), "-".into()]] {
            assert!(!uses_stdin(&arguments));
        }
    }

    #[test]
    fn stdin_preserves_unicode_whitespace_and_literal_shell_text() {
        let text = "  café\n$(not-a-command)\r\n ";
        assert_eq!(read_prompt(text.as_bytes()).unwrap(), text);
    }

    #[test]
    fn stdin_accepts_the_limit_and_rejects_overflow_and_invalid_utf8() {
        let at_limit = vec![b'x'; MAX_PROMPT_BYTES];
        assert_eq!(
            read_prompt(at_limit.as_slice()).unwrap().len(),
            MAX_PROMPT_BYTES
        );
        let too_large = vec![b'x'; MAX_PROMPT_BYTES + 1];
        assert!(read_prompt(too_large.as_slice()).is_err());
        assert!(read_prompt(&[0xff][..]).is_err());
    }

    #[tokio::test]
    async fn ordinary_arguments_do_not_read_stdin() {
        assert_eq!(
            resolve_prompt(vec!["a".into(), "-".into()], &CancellationToken::new())
                .await
                .unwrap(),
            "a -"
        );
    }
}
