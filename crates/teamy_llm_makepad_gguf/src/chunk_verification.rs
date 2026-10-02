//! Fresh, parallel verification against a commitment pinned by the application build.
//!
//! Independent chunk hashes do not reconstruct the ordinary whole-file SHA-256. The trusted
//! table must be produced while the same guarded file also matches its frozen whole-file SHA,
//! then its canonical digest must be pinned in reviewed source. A mutable table's assertion of
//! the whole-file SHA is never an authentication mechanism. Callers retain their read-only
//! file/mapping guard until all use of the verified bytes has finished.

use std::ffi::OsStr;
use std::io::Read;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::Instant;

use eyre::{Result, WrapErr, ensure, eyre};
use sha2::{Digest, Sha256};
use teamy_cancellation::CancellationToken;

pub const CHUNK_TABLE_VERSION: u32 = 1;
pub const DEFAULT_CHUNK_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_VERIFICATION_WORKERS: usize = 32;
const VERIFY_WORKERS_ENV: &str = "TEAMY_LLM_GGUF_VERIFY_WORKERS";
const HASH_UPDATE_BYTES: usize = 1024 * 1024;
const DIGEST_BYTES: usize = 32;
const TABLE_DOMAIN: &[u8] = b"teamy-gguf-sha256-chunks-v1\0";

/// The digest of this descriptor/table must come from trusted, compiled application data.
///
/// The fields are not self-authenticating: never take `table_sha256` from the mutable table
/// being checked. A disk table is usable only with an independently compiled digest.
#[derive(Clone, Copy, Debug)]
pub struct TrustedChunkCommitment<'a> {
    pub version: u32,
    pub chunk_bytes: usize,
    pub total_bytes: u64,
    pub whole_sha256: &'a str,
    pub table_sha256: &'a str,
    /// Ordered, concatenated 32-byte SHA-256 digests, including the exact final short chunk.
    pub digests: &'a [u8],
}

#[derive(Clone, Debug)]
pub struct ChunkVerificationReceipt {
    pub version: u32,
    pub expected_whole_sha256: String,
    pub table_sha256: String,
    pub verified_bytes: u64,
    pub verified_chunks: usize,
    pub worker_count: usize,
    pub elapsed_ms: f64,
}

/// An authoring result, not an independently trusted runtime commitment.
#[derive(Clone, Debug)]
pub struct GeneratedChunkTable {
    pub version: u32,
    pub chunk_bytes: usize,
    pub total_bytes: u64,
    pub whole_sha256: String,
    pub table_sha256: String,
    pub digests: Vec<u8>,
    pub elapsed_ms: f64,
}

impl GeneratedChunkTable {
    /// This view is appropriate for authoring tests only until its digest has been pinned.
    #[must_use]
    pub fn commitment(&self) -> TrustedChunkCommitment<'_> {
        TrustedChunkCommitment {
            version: self.version,
            chunk_bytes: self.chunk_bytes,
            total_bytes: self.total_bytes,
            whole_sha256: &self.whole_sha256,
            table_sha256: &self.table_sha256,
            digests: &self.digests,
        }
    }
}

/// Verify every byte, in bounded parallel workers, before any caller uses the model.
///
/// This authenticates freshly computed chunk hashes against the compiled commitment. It does
/// not report a newly recomputed ordinary whole-file SHA-256. The file/mapping remains guarded
/// by the caller, including after this function returns. By default the worker count is bounded
/// by both the available logical CPUs and 32; an explicit process override can select a smaller
/// count or request 32 without changing the verification contract.
pub fn verify_chunks(
    bytes: &[u8],
    commitment: &TrustedChunkCommitment<'_>,
    cancellation: &CancellationToken,
) -> Result<ChunkVerificationReceipt> {
    let available = thread::available_parallelism().map_or(1, usize::from);
    let value = std::env::var_os(VERIFY_WORKERS_ENV);
    let workers = resolve_verification_workers(value.as_deref(), available)?;
    verify_chunks_with_workers(bytes, commitment, cancellation, workers)
}

fn parse_verification_workers(value: Option<&str>) -> Result<Option<usize>> {
    match value {
        None => Ok(None),
        Some("1") => Ok(Some(1)),
        Some("4") => Ok(Some(4)),
        Some("8") => Ok(Some(8)),
        Some("12") => Ok(Some(12)),
        Some("16") => Ok(Some(16)),
        Some("32") => Ok(Some(32)),
        Some(_) => Err(eyre!(
            "{VERIFY_WORKERS_ENV} must be exactly 1, 4, 8, 12, 16 or 32; verification cannot be disabled"
        )),
    }
}

fn resolve_verification_workers(value: Option<&OsStr>, available: usize) -> Result<usize> {
    let value = value
        .map(|value| {
            value
                .to_str()
                .ok_or_else(|| eyre!("{VERIFY_WORKERS_ENV} must be valid UTF-8"))
        })
        .transpose()?;
    Ok(parse_verification_workers(value)?
        .unwrap_or_else(|| available.clamp(1, MAX_VERIFICATION_WORKERS)))
}

fn verify_chunks_with_workers(
    bytes: &[u8],
    commitment: &TrustedChunkCommitment<'_>,
    cancellation: &CancellationToken,
    requested_workers: usize,
) -> Result<ChunkVerificationReceipt> {
    let started = Instant::now();
    cancellation.bail_if_cancelled()?;
    let count = validate_descriptor(commitment)?;
    ensure!(
        u64::try_from(bytes.len())? == commitment.total_bytes,
        "GGUF chunk verification size mismatch: expected {} bytes, received {}",
        commitment.total_bytes,
        bytes.len()
    );
    let table_sha256 = canonical_table_sha256(
        commitment.version,
        commitment.chunk_bytes,
        commitment.total_bytes,
        commitment.whole_sha256,
        commitment.digests,
    )?;
    ensure!(
        decode_sha256(&table_sha256)? == decode_sha256(commitment.table_sha256)?,
        "GGUF chunk table does not match its compiled commitment"
    );
    let worker_count = requested_workers
        .clamp(1, MAX_VERIFICATION_WORKERS)
        .min(count);
    let next = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let verified = AtomicUsize::new(0);
    thread::scope(|scope| -> Result<()> {
        let mut handles = Vec::with_capacity(worker_count);
        for worker in 0..worker_count {
            let handle = thread::Builder::new()
                .name(format!("gguf-verify-{worker}"))
                .spawn_scoped(scope, || -> Result<()> {
                let result = (|| {
                    loop {
                        if stop.load(Ordering::Acquire) {
                            return Ok(());
                        }
                        cancellation.bail_if_cancelled()?;
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        if index >= count {
                            return Ok(());
                        }
                        let start = index.checked_mul(commitment.chunk_bytes)
                            .ok_or_else(|| eyre!("GGUF chunk offset overflow"))?;
                        let end = start.saturating_add(commitment.chunk_bytes).min(bytes.len());
                        let mut hash = Sha256::new();
                        for piece in bytes[start..end].chunks(HASH_UPDATE_BYTES) {
                            cancellation.bail_if_cancelled()?;
                            if stop.load(Ordering::Acquire) {
                                return Ok(());
                            }
                            hash.update(piece);
                        }
                        let actual = hash.finalize();
                        let expected = &commitment.digests[index * DIGEST_BYTES..(index + 1) * DIGEST_BYTES];
                        ensure!(
                            actual.as_slice() == expected,
                            "GGUF chunk SHA-256 mismatch at chunk {index}, byte range {start}..{end}"
                        );
                        verified.fetch_add(1, Ordering::Relaxed);
                    }
                })();
                if result.is_err() {
                    stop.store(true, Ordering::Release);
                }
                result
                });
            match handle {
                Ok(handle) => handles.push(handle),
                Err(error) => {
                    stop.store(true, Ordering::Release);
                    return Err(error)
                        .wrap_err("Starting a bounded GGUF verification worker failed");
                }
            }
        }
        // Always join every worker before releasing the caller's borrowed mapped bytes.
        let mut failure = None;
        for handle in handles {
            let result = handle
                .join()
                .unwrap_or_else(|_| Err(eyre!("GGUF chunk verification worker panicked")));
            if let Err(error) = result {
                stop.store(true, Ordering::Release);
                if failure.is_none() {
                    failure = Some(error);
                }
            }
        }
        if let Some(error) = failure {
            return Err(error);
        }
        Ok(())
    })?;
    cancellation.bail_if_cancelled()?;
    ensure!(
        verified.load(Ordering::Relaxed) == count,
        "GGUF chunk verification stopped before checking every chunk"
    );
    Ok(ChunkVerificationReceipt {
        version: CHUNK_TABLE_VERSION,
        expected_whole_sha256: commitment.whole_sha256.to_ascii_lowercase(),
        table_sha256,
        verified_bytes: commitment.total_bytes,
        verified_chunks: count,
        worker_count,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
    })
}

/// Make an ordered digest table only after the same streamed bytes match the frozen whole SHA.
///
/// The caller must hold an ordinary-write/delete-excluding file guard throughout this pass.
/// The resulting table still needs an independently trusted, compiled commitment before use
/// in a different process. No model bytes are copied into this result.
pub fn generate_chunk_table<R: Read>(
    reader: &mut R,
    expected_bytes: u64,
    expected_whole_sha256: &str,
    chunk_bytes: usize,
    cancellation: &CancellationToken,
) -> Result<GeneratedChunkTable> {
    let started = Instant::now();
    cancellation.bail_if_cancelled()?;
    let expected_whole = decode_sha256(expected_whole_sha256)?;
    let count = chunk_count(expected_bytes, chunk_bytes)?;
    let capacity = count
        .checked_mul(DIGEST_BYTES)
        .ok_or_else(|| eyre!("GGUF digest table size overflow"))?;
    let mut digests = Vec::with_capacity(capacity);
    let mut whole = Sha256::new();
    let mut chunk = Sha256::new();
    let mut chunk_used = 0usize;
    let mut total_bytes = 0u64;
    let mut buffer = vec![0; HASH_UPDATE_BYTES.min(chunk_bytes)];
    loop {
        cancellation.bail_if_cancelled()?;
        let read = reader
            .read(&mut buffer)
            .wrap_err("Reading guarded GGUF for chunk table failed")?;
        if read == 0 {
            break;
        }
        total_bytes = total_bytes
            .checked_add(u64::try_from(read)?)
            .ok_or_else(|| eyre!("GGUF authoring byte count overflow"))?;
        ensure!(
            total_bytes <= expected_bytes,
            "GGUF authoring source is larger than the frozen size"
        );
        whole.update(&buffer[..read]);
        let mut offset = 0usize;
        while offset < read {
            cancellation.bail_if_cancelled()?;
            let take = (chunk_bytes - chunk_used).min(read - offset);
            chunk.update(&buffer[offset..offset + take]);
            offset += take;
            chunk_used += take;
            if chunk_used == chunk_bytes {
                digests.extend_from_slice(&chunk.finalize_reset());
                chunk_used = 0;
            }
        }
    }
    cancellation.bail_if_cancelled()?;
    ensure!(
        total_bytes == expected_bytes,
        "GGUF authoring size mismatch: expected {expected_bytes}, received {total_bytes}"
    );
    if chunk_used != 0 {
        digests.extend_from_slice(&chunk.finalize());
    }
    let actual_whole = whole.finalize();
    ensure!(
        actual_whole.as_slice() == expected_whole,
        "GGUF authoring whole-file SHA-256 does not match the frozen artifact"
    );
    ensure!(
        digests.len() == capacity,
        "GGUF authoring produced an incomplete chunk table"
    );
    let whole_sha256 = expected_whole_sha256.to_ascii_lowercase();
    let table_sha256 = canonical_table_sha256(
        CHUNK_TABLE_VERSION,
        chunk_bytes,
        total_bytes,
        &whole_sha256,
        &digests,
    )?;
    Ok(GeneratedChunkTable {
        version: CHUNK_TABLE_VERSION,
        chunk_bytes,
        total_bytes,
        whole_sha256,
        table_sha256,
        digests,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
    })
}

/// Canonical commitment: domain, LE version/chunk size/file size, whole SHA bytes, LE count,
/// then the ordered raw chunk digest bytes. This binds metadata as well as the digest table.
pub fn canonical_table_sha256(
    version: u32,
    chunk_bytes: usize,
    total_bytes: u64,
    whole_sha256: &str,
    digests: &[u8],
) -> Result<String> {
    let count = chunk_count(total_bytes, chunk_bytes)?;
    ensure!(
        version == CHUNK_TABLE_VERSION,
        "Unsupported GGUF chunk table version {version}"
    );
    ensure!(
        digests.len()
            == count
                .checked_mul(DIGEST_BYTES)
                .ok_or_else(|| eyre!("GGUF table size overflow"))?,
        "GGUF digest table length does not match its descriptor"
    );
    let mut hash = Sha256::new();
    hash.update(TABLE_DOMAIN);
    hash.update(version.to_le_bytes());
    hash.update(u64::try_from(chunk_bytes)?.to_le_bytes());
    hash.update(total_bytes.to_le_bytes());
    hash.update(decode_sha256(whole_sha256)?);
    hash.update(u64::try_from(count)?.to_le_bytes());
    hash.update(digests);
    Ok(format!("{:x}", hash.finalize()))
}

fn chunk_count(total_bytes: u64, chunk_bytes: usize) -> Result<usize> {
    ensure!(
        total_bytes != 0,
        "GGUF chunk commitment cannot describe an empty model"
    );
    ensure!(chunk_bytes != 0, "GGUF chunk size must be positive");
    usize::try_from(total_bytes.div_ceil(u64::try_from(chunk_bytes)?))
        .wrap_err("GGUF chunk count does not fit this platform")
}

fn validate_descriptor(commitment: &TrustedChunkCommitment<'_>) -> Result<usize> {
    ensure!(
        commitment.version == CHUNK_TABLE_VERSION,
        "Unsupported GGUF chunk table version {}",
        commitment.version
    );
    decode_sha256(commitment.whole_sha256)?;
    decode_sha256(commitment.table_sha256)?;
    let count = chunk_count(commitment.total_bytes, commitment.chunk_bytes)?;
    ensure!(
        commitment.digests.len()
            == count
                .checked_mul(DIGEST_BYTES)
                .ok_or_else(|| eyre!("GGUF table size overflow"))?,
        "GGUF digest table length does not match its descriptor"
    );
    Ok(count)
}

fn decode_sha256(value: &str) -> Result<[u8; DIGEST_BYTES]> {
    ensure!(
        value.len() == DIGEST_BYTES * 2,
        "SHA-256 commitment must contain exactly 64 hex digits"
    );
    let mut result = [0; DIGEST_BYTES];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let digit = |byte| -> Result<u8> {
            match byte {
                b'0'..=b'9' => Ok(byte - b'0'),
                b'a'..=b'f' => Ok(byte - b'a' + 10),
                b'A'..=b'F' => Ok(byte - b'A' + 10),
                _ => Err(eyre!("SHA-256 commitment contains a non-hex digit")),
            }
        };
        result[index] = (digit(pair[0])? << 4) | digit(pair[1])?;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn authored(bytes: &[u8], chunk_bytes: usize) -> GeneratedChunkTable {
        let whole = format!("{:x}", Sha256::digest(bytes));
        generate_chunk_table(
            &mut Cursor::new(bytes),
            bytes.len() as u64,
            &whole,
            chunk_bytes,
            &CancellationToken::new(),
        )
        .unwrap()
    }

    #[test]
    fn verification_worker_override_is_exact_bounded_and_never_disables_verification() {
        assert_eq!(parse_verification_workers(None).unwrap(), None);
        for (text, count) in [
            ("1", 1),
            ("4", 4),
            ("8", 8),
            ("12", 12),
            ("16", 16),
            ("32", 32),
        ] {
            assert_eq!(parse_verification_workers(Some(text)).unwrap(), Some(count));
            assert_eq!(
                resolve_verification_workers(Some(OsStr::new(text)), 32).unwrap(),
                count
            );
        }
        for value in [
            "", "0", "2", "3", "17", "31", "33", "64", "off", "false", " 8", "8 ", "08", "+8", "-1",
        ] {
            assert!(parse_verification_workers(Some(value)).is_err());
        }
        assert_eq!(resolve_verification_workers(None, 64).unwrap(), 32);
        assert_eq!(resolve_verification_workers(None, 32).unwrap(), 32);
        assert_eq!(resolve_verification_workers(None, 24).unwrap(), 24);
        assert_eq!(resolve_verification_workers(None, 16).unwrap(), 16);
        assert_eq!(resolve_verification_workers(None, 12).unwrap(), 12);
        assert_eq!(resolve_verification_workers(None, 1).unwrap(), 1);
        assert_eq!(resolve_verification_workers(None, 0).unwrap(), 1);
    }

    #[cfg(windows)]
    #[test]
    fn verification_worker_override_rejects_non_utf8_without_hashing() {
        use std::os::windows::ffi::OsStringExt;
        let value = std::ffi::OsString::from_wide(&[0xd800]);
        assert!(
            resolve_verification_workers(Some(&value), 16)
                .unwrap_err()
                .to_string()
                .contains("valid UTF-8")
        );
    }

    #[cfg(unix)]
    #[test]
    fn verification_worker_override_rejects_non_utf8_without_hashing() {
        use std::os::unix::ffi::OsStrExt;
        let value = OsStr::from_bytes(&[0xff]);
        assert!(
            resolve_verification_workers(Some(value), 16)
                .unwrap_err()
                .to_string()
                .contains("valid UTF-8")
        );
    }

    #[test]
    fn all_bytes_and_exact_last_chunk_are_verified_in_parallel() {
        let bytes = b"abcdefghijk";
        let table = authored(bytes, 3);
        assert_eq!(table.digests.len(), 4 * DIGEST_BYTES);
        assert_eq!(
            &table.digests[3 * DIGEST_BYTES..],
            Sha256::digest(b"jk").as_slice()
        );
        let receipt =
            verify_chunks_with_workers(bytes, &table.commitment(), &CancellationToken::new(), 16)
                .unwrap();
        assert_eq!(receipt.verified_bytes, 11);
        assert_eq!(receipt.verified_chunks, 4);
        assert_eq!(receipt.worker_count, 4);
    }

    #[test]
    fn modifications_in_every_chunk_including_the_last_are_rejected() {
        let bytes = b"abcdefghijk";
        let table = authored(bytes, 3);
        for index in [0, 3, 6, 10] {
            let mut changed = bytes.to_vec();
            changed[index] ^= 1;
            let error = verify_chunks_with_workers(
                &changed,
                &table.commitment(),
                &CancellationToken::new(),
                3,
            )
            .unwrap_err();
            assert!(error.to_string().contains("chunk SHA-256 mismatch"));
        }
    }

    #[test]
    fn exact_chunk_boundary_does_not_create_an_empty_final_digest() {
        let bytes = b"abcdefghi";
        let table = authored(bytes, 3);
        assert_eq!(table.digests.len(), 3 * DIGEST_BYTES);
        verify_chunks(bytes, &table.commitment(), &CancellationToken::new()).unwrap();
    }

    #[test]
    fn truncation_and_appended_bytes_are_rejected() {
        let bytes = b"abcdefg";
        let table = authored(bytes, 3);
        assert!(
            verify_chunks(&bytes[..6], &table.commitment(), &CancellationToken::new())
                .unwrap_err()
                .to_string()
                .contains("size mismatch")
        );
        assert!(
            verify_chunks(b"abcdefgh", &table.commitment(), &CancellationToken::new())
                .unwrap_err()
                .to_string()
                .contains("size mismatch")
        );
    }

    #[test]
    fn reordered_or_replaced_digest_table_cannot_use_the_pinned_root() {
        let bytes = b"abcdefg";
        let table = authored(bytes, 3);
        let mut replaced = table.digests.clone();
        replaced[0] ^= 1;
        let mut commitment = table.commitment();
        commitment.digests = &replaced;
        assert!(
            verify_chunks(bytes, &commitment, &CancellationToken::new())
                .unwrap_err()
                .to_string()
                .contains("compiled commitment")
        );
        let mut reordered = table.digests.clone();
        reordered[..64].rotate_left(32);
        commitment.digests = &reordered;
        assert!(
            verify_chunks(bytes, &commitment, &CancellationToken::new())
                .unwrap_err()
                .to_string()
                .contains("compiled commitment")
        );
    }

    #[test]
    fn canonical_root_binds_size_chunk_shape_and_whole_hash() {
        let table = authored(b"abcdefg", 3);
        let root = &table.table_sha256;
        assert_ne!(
            canonical_table_sha256(1, 4, 9, &table.whole_sha256, &table.digests).unwrap(),
            *root
        );
        assert_ne!(
            canonical_table_sha256(1, 3, 8, &table.whole_sha256, &table.digests).unwrap(),
            *root
        );
        assert_ne!(
            canonical_table_sha256(1, 3, 7, &"00".repeat(32), &table.digests).unwrap(),
            *root
        );
    }

    #[test]
    fn unknown_version_missing_digests_and_invalid_root_are_rejected() {
        let table = authored(b"abcdefg", 3);
        let mut commitment = table.commitment();
        commitment.version = 2;
        assert!(verify_chunks(b"abcdefg", &commitment, &CancellationToken::new()).is_err());
        commitment = table.commitment();
        commitment.digests = &table.digests[..64];
        assert!(verify_chunks(b"abcdefg", &commitment, &CancellationToken::new()).is_err());
        commitment = table.commitment();
        commitment.table_sha256 = "invalid";
        assert!(verify_chunks(b"abcdefg", &commitment, &CancellationToken::new()).is_err());
    }

    #[test]
    fn cancelled_verification_never_returns_a_success_receipt() {
        let table = authored(b"abcdefg", 3);
        let cancellation = CancellationToken::new();
        cancellation.request_cancel("fixture cancelled");
        assert!(
            verify_chunks(b"abcdefg", &table.commitment(), &cancellation)
                .unwrap_err()
                .to_string()
                .contains("fixture cancelled")
        );
    }

    #[test]
    fn generation_requires_the_actual_frozen_whole_hash_and_size() {
        let bytes = b"abcdefg";
        assert!(
            generate_chunk_table(
                &mut Cursor::new(bytes),
                7,
                &"00".repeat(32),
                3,
                &CancellationToken::new()
            )
            .unwrap_err()
            .to_string()
            .contains("whole-file SHA-256")
        );
        let whole = format!("{:x}", Sha256::digest(bytes));
        assert!(
            generate_chunk_table(
                &mut Cursor::new(bytes),
                6,
                &whole,
                3,
                &CancellationToken::new()
            )
            .is_err()
        );
        assert!(
            generate_chunk_table(
                &mut Cursor::new(bytes),
                8,
                &whole,
                3,
                &CancellationToken::new()
            )
            .is_err()
        );
    }

    #[test]
    fn cancellation_is_observed_between_generator_reads() {
        struct CancelAfterRead {
            cancellation: CancellationToken,
        }
        impl Read for CancelAfterRead {
            fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
                buffer[0] = b'a';
                self.cancellation.request_cancel("cancel after first read");
                Ok(1)
            }
        }
        let cancellation = CancellationToken::new();
        let mut reader = CancelAfterRead {
            cancellation: cancellation.clone(),
        };
        let expected = format!("{:x}", Sha256::digest(b"a"));
        assert!(
            generate_chunk_table(&mut reader, 1, &expected, 1, &cancellation)
                .unwrap_err()
                .to_string()
                .contains("cancel after first read")
        );
    }

    #[test]
    fn zero_sizes_and_malformed_hex_do_not_panic() {
        assert!(canonical_table_sha256(1, 0, 1, &"00".repeat(32), &[]).is_err());
        assert!(canonical_table_sha256(1, 1, 0, &"00".repeat(32), &[]).is_err());
        assert!(decode_sha256(&"zz".repeat(32)).is_err());
        assert!(decode_sha256(&"é".repeat(32)).is_err());
    }
}
