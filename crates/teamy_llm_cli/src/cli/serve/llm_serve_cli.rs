use std::time::Instant;

use crate::cli::{model_selection::ModelSelectionArgs, prompt::generation_request};
use eyre::{Result, WrapErr, ensure};
use facet::Facet;
use figue as args;
use teamy_cancellation::CancellationToken;
use teamy_llm_service::TeamyLlmService;
/// Serve resident local models over Vox IPC.
#[derive(Debug, Facet)]
pub struct ServeCli {
    /// Local service address. This command keeps model runtimes resident.
    #[facet(args::named, default = default_local_address())]
    pub address: String,
    /// Explicit model to prewarm before accepting requests. Omit for lazy loading.
    #[facet(flatten, default)]
    pub selection: ModelSelectionArgs,
    /// Positive total input and output capacity for the prewarmed model session.
    /// Requires --model or --model-dir; clients should request the same capacity.
    #[facet(args::named)]
    pub context_tokens: Option<u32>,
}

pub fn default_local_address() -> String {
    #[cfg(windows)]
    {
        r"local://\\.\pipe\teamy-llm-service".into()
    }
    #[cfg(not(windows))]
    {
        "local://teamy-llm-service".into()
    }
}

/// This CLI deliberately exposes local IPC only, even though Vox supports TCP.
/// On Windows, remote UNC pipe servers and the invalid short pipe name are rejected.
pub(crate) fn validate_local_address(address: &str) -> Result<()> {
    let endpoint = address.strip_prefix("local://");
    ensure!(endpoint.is_some(), "service address must use local:// IPC");
    let endpoint = endpoint.expect("scheme checked above");
    ensure!(
        !endpoint.is_empty(),
        "local service address requires an endpoint"
    );
    ensure!(
        !endpoint.chars().any(char::is_control),
        "local service address contains a control character"
    );
    #[cfg(windows)]
    {
        let name = endpoint.strip_prefix(r"\\.\pipe\");
        ensure!(
            name.is_some(),
            "Windows local IPC requires local://\\\\.\\pipe\\<name>"
        );
        let name = name.expect("local named-pipe prefix checked above");
        ensure!(
            !name.is_empty(),
            "Windows local IPC requires a nonempty pipe name"
        );
        ensure!(
            !name.contains(['\\', '/']),
            "Windows pipe name must not contain path separators"
        );
    }
    Ok(())
}

impl ServeCli {
    #[tracing::instrument(name = "llm_serve", skip_all)]
    pub async fn invoke(
        self,
        service: TeamyLlmService,
        cancellation: CancellationToken,
    ) -> Result<()> {
        validate_local_address(&self.address)?;
        self.selection.to_selector()?;
        ensure!(
            self.context_tokens != Some(0),
            "--context-tokens must be at least 1"
        );
        let prewarm = self.selection.model.is_some() || self.selection.model_dir.is_some();
        ensure!(
            self.context_tokens.is_none() || prewarm,
            "serve --context-tokens requires an explicit --model or --model-dir to prewarm"
        );
        cancellation.bail_if_cancelled()?;
        if prewarm {
            let mut request = generation_request(
                &service,
                &self.selection,
                None,
                false,
                Some(1),
                None,
                "Warmup.".into(),
            )?;
            request.context_tokens = self.context_tokens;
            let started = Instant::now();
            let warm_service = service.clone();
            let warm_cancellation = cancellation.clone();
            let span = tracing::info_span!(
                "llm_resident_prewarm", model = self.selection.model.as_deref().unwrap_or("explicit-directory"),
                context_tokens = ?self.context_tokens,
            );
            tokio::task::spawn_blocking(move || {
                span.in_scope(|| {
                    warm_service.generate_with_cancellation(
                        request,
                        Some(warm_cancellation),
                        &mut |_| Ok(()),
                    )
                })
            })
            .await
            .wrap_err("resident prewarm worker failed")??;
            cancellation.bail_if_cancelled()?;
            tracing::info!(
                model = self.selection.model.as_deref().unwrap_or("explicit-directory"),
                context_tokens = ?self.context_tokens,
                warmup_ms = started.elapsed().as_secs_f64() * 1000.0,
                "resident model ready; starting local IPC listener"
            );
        } else {
            tracing::info!("starting local IPC listener; models will load lazily on first request");
        }
        tracing::info!(address = %self.address, "serving Teamy LLM over Vox local IPC");
        // The service helper cancels and drains active native jobs as well as
        // stopping the listener; dropping only Vox's listener leaves its
        // spawned connection handlers alive during runtime shutdown.
        teamy_llm_service::vox_api::serve_local_with_cancellation(
            &self.address,
            service,
            cancellation.clone(),
        )
        .await
        .wrap_err("local inference server stopped")?;
        cancellation.bail_if_cancelled()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_endpoint_is_local_and_valid_for_this_platform() {
        assert!(validate_local_address(&default_local_address()).is_ok());
        #[cfg(windows)]
        assert_eq!(
            default_local_address(),
            r"local://\\.\pipe\teamy-llm-service"
        );
    }

    #[test]
    fn nonlocal_and_control_character_endpoints_are_rejected() {
        for address in [
            "tcp://127.0.0.1:1234",
            "http://localhost",
            "localhost:1234",
            "local://",
            "local://unsafe\npipe",
        ] {
            assert!(validate_local_address(address).is_err(), "{address:?}");
        }
    }

    #[cfg(windows)]
    #[test]
    fn windows_pipe_names_cannot_select_a_remote_server_or_directory() {
        for address in [
            "local://teamy-llm-service",
            r"local://\\other-server\pipe\llm",
            r"local://\\.\pipe\",
            r"local://\\.\pipe\parent\child",
            r"local://\\.\pipe\parent/child",
        ] {
            assert!(validate_local_address(address).is_err(), "{address:?}");
        }
    }
}
