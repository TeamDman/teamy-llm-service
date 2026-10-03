//! Process-owned compilation cache. A miss recompiles; model weights never live here.
use std::{
    fs::{self, File, OpenOptions},
    io::Read,
    path::Path,
    sync::{Arc, Mutex, OnceLock},
};

use cubek::cubecl::config::{CubeClRuntimeConfig, RuntimeConfig, cache::CacheConfig};
use eyre::{WrapErr, ensure};
use sha2::{Digest, Sha256};
use teamy_cancellation::CancellationToken;

use crate::cuda_runtime::CudaRuntimeIdentity;

// The lease outlives every compute worker, including workers retained by CubeCL globals.
static CACHE_LEASE: OnceLock<Option<File>> = OnceLock::new();
static CACHE_INIT: Mutex<()> = Mutex::new(());

pub(crate) fn initialize(
    identity: &CudaRuntimeIdentity,
    cancellation: &CancellationToken,
) -> eyre::Result<()> {
    cancellation.bail_if_cancelled()?;
    let _initialization = CACHE_INIT
        .lock()
        .map_err(|_| eyre::eyre!("Julia cache initialization lock poisoned"))?;
    cancellation.bail_if_cancelled()?;
    if CACHE_LEASE.get().is_none() {
        let lease = match configure(identity, cancellation) {
            Ok(lease) => lease,
            Err(error) => {
                // A cancelled setup must remain retryable in a resident process.
                cancellation.bail_if_cancelled()?;
                let mut state = CubeClRuntimeConfig::storage().lock();
                if state.is_none() {
                    // Do not fall through to CubeCL's current-directory configuration discovery.
                    *state = Some(Arc::new(CubeClRuntimeConfig::default()));
                }
                tracing::warn!(error = %error, "Julia kernel persistence unavailable; kernels will compile in memory");
                None
            }
        };
        // configure may already have published persistence. Retain its lease even
        // if cancellation arrived immediately after that successful publication.
        let _ = CACHE_LEASE.set(lease);
    }
    cancellation.bail_if_cancelled()?;
    let state = CubeClRuntimeConfig::storage().lock();
    ensure!(
        CACHE_LEASE.get().is_some_and(Option::is_some)
            || state
                .as_ref()
                .is_none_or(|config| config.compilation.cache.is_none()),
        "CubeCL already uses a foreign persistent cache; initialize Julia before other CUDA providers"
    );
    Ok(())
}

#[tracing::instrument(level = "debug", name = "julia_kernel_cache_setup", skip_all)]
fn configure(
    identity: &CudaRuntimeIdentity,
    cancellation: &CancellationToken,
) -> eyre::Result<Option<File>> {
    if let Some(config) = CubeClRuntimeConfig::storage().lock().as_ref() {
        ensure!(
            config.compilation.cache.is_none(),
            "CubeCL already uses a foreign persistent cache; initialize Julia before other CUDA providers"
        );
        tracing::debug!("CubeCL configuration already initialized with persistence disabled");
        return Ok(None);
    }
    cancellation.bail_if_cancelled()?;
    let mut digest = Sha256::new();
    digest.update(b"teamy-julia-cubecl-cache-v2-cubin-fp32-simpleunit");
    hash_file(&mut digest, &std::env::current_exe()?, cancellation)?;
    digest.update(identity.driver_api_version.to_le_bytes());
    digest.update(identity.nvrtc_major.to_le_bytes());
    digest.update(identity.nvrtc_minor.to_le_bytes());
    digest.update((identity.devices_sm.len() as u64).to_le_bytes());
    for (major, minor) in &identity.devices_sm {
        digest.update(major.to_le_bytes());
        digest.update(minor.to_le_bytes());
    }
    // SDK binaries and every header's contents participate, rather than trusting timestamps.
    // Keep the selected toolkit immutable during execution, like the model acquisition.
    hash_path(&mut digest, &identity.nvrtc_path);
    hash_file(&mut digest, &identity.nvrtc_path, cancellation)?;
    hash_path(&mut digest, &identity.nvrtc_builtins_path);
    hash_file(&mut digest, &identity.nvrtc_builtins_path, cancellation)?;
    hash_headers(&mut digest, &identity.include_path, cancellation)?;
    let key = format!("{:x}", digest.finalize());
    let home = teamy_llm_burn_qwen35::paths::CacheHome::resolve()?;
    ensure!(
        home.is_absolute(),
        "Julia kernel cache home must be absolute"
    );
    let root = home.join("julia-kernels").join(key);
    fs::create_dir_all(&root).wrap_err("Create Julia kernel cache namespace")?;
    let lease = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(root.join("compilation.lock"))
        .wrap_err("Open Julia kernel cache lease")?;
    lease
        .try_lock()
        .wrap_err("Another process owns the Julia kernel cache")?;
    cancellation.bail_if_cancelled()?;
    let mut state = CubeClRuntimeConfig::storage().lock();
    if let Some(config) = state.as_ref() {
        ensure!(
            config.compilation.cache.is_none(),
            "CubeCL initialized a foreign persistent cache during Julia setup"
        );
        return Ok(None);
    }
    let mut config = CubeClRuntimeConfig::default();
    config.compilation.cache = Some(CacheConfig::File(root));
    // Assign through the same mutex as get(), so concurrent initialization cannot race set().
    *state = Some(Arc::new(config));
    tracing::debug!("Julia kernel persistence enabled for this executable and CUDA toolkit");
    Ok(Some(lease))
}

fn hash_file(
    digest: &mut Sha256,
    path: &Path,
    cancellation: &CancellationToken,
) -> eyre::Result<()> {
    let mut file = File::open(path)?;
    let length = file.metadata()?.len();
    ensure!(
        length <= 512 * 1024 * 1024,
        "Cache identity file exceeds bound"
    );
    digest.update(length.to_le_bytes());
    let mut buffer = [0; 64 * 1024];
    let mut read_bytes = 0_u64;
    loop {
        cancellation.bail_if_cancelled()?;
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        read_bytes += count as u64;
        ensure!(
            read_bytes <= length,
            "Cache identity file grew during hashing"
        );
        digest.update(&buffer[..count]);
    }
    ensure!(
        read_bytes == length && file.metadata()?.len() == length,
        "Cache identity file changed length during hashing"
    );
    Ok(())
}

fn hash_path(digest: &mut Sha256, path: &Path) {
    let bytes = path.as_os_str().as_encoded_bytes();
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
}

fn hash_headers(
    digest: &mut Sha256,
    root: &Path,
    cancellation: &CancellationToken,
) -> eyre::Result<()> {
    let mut pending = vec![root.to_owned()];
    let mut files = Vec::new();
    let mut entries = 0;
    let mut total_bytes = 0_u64;
    while let Some(directory) = pending.pop() {
        cancellation.bail_if_cancelled()?;
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            entries += 1;
            ensure!(
                entries <= 20_000,
                "CUDA include tree exceeds cache identity bound"
            );
            let kind = entry.file_type()?;
            ensure!(
                !kind.is_symlink(),
                "CUDA include symlinks require a separate cache identity policy"
            );
            if kind.is_dir() {
                pending.push(entry.path());
            } else if kind.is_file() {
                total_bytes = total_bytes
                    .checked_add(entry.metadata()?.len())
                    .ok_or_else(|| eyre::eyre!("CUDA include byte count overflow"))?;
                ensure!(
                    total_bytes <= 512 * 1024 * 1024,
                    "CUDA include tree exceeds cache byte bound"
                );
                files.push(entry.path());
            }
        }
    }
    files.sort();
    for file in files {
        cancellation.bail_if_cancelled()?;
        hash_path(digest, &file);
        hash_file(digest, &file, cancellation)?;
    }
    Ok(())
}
