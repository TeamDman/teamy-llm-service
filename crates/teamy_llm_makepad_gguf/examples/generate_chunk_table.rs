//! Author an Orca chunk table after a guarded, ordinary full-SHA verification.
//!
//! Output hash data contains no source path. The resulting table is not trusted at runtime
//! until its canonical commitment has been independently pinned in reviewed application code.

use std::ffi::OsString;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use eyre::{Result, WrapErr, bail, ensure, eyre};
use teamy_cancellation::CancellationToken;
use teamy_llm_makepad_gguf::chunk_verification::{DEFAULT_CHUNK_BYTES, generate_chunk_table};
use teamy_llm_makepad_gguf::{ORCA_GGUF_BYTES, ORCA_GGUF_SHA256};

fn main() -> Result<()> {
    let (source, table_output, receipt_output) = arguments()?;
    ensure!(
        table_output != receipt_output,
        "Table and receipt outputs must differ"
    );
    ensure!(
        !table_output.exists(),
        "Table output already exists; authoring never overwrites it"
    );
    ensure!(
        !receipt_output.exists(),
        "Receipt output already exists; authoring never overwrites it"
    );
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.share_mode(1); // FILE_SHARE_READ: exclude ordinary writes and replacement.
    }
    let mut guarded = options
        .open(&source)
        .wrap_err("Opening source with a read-only guard failed")?;
    ensure!(
        guarded.metadata()?.len() == ORCA_GGUF_BYTES,
        "Source size differs from the frozen Orca artifact"
    );
    let table = generate_chunk_table(
        &mut guarded,
        ORCA_GGUF_BYTES,
        ORCA_GGUF_SHA256,
        DEFAULT_CHUNK_BYTES,
        &CancellationToken::new(),
    )?;
    // This branch is reached only after the same pass matches the existing frozen whole SHA.
    write_new(&table_output, &table.digests)?;
    let receipt = format!(
        "{{\n  \"version\": {},\n  \"whole_sha256\": \"{}\",\n  \"total_bytes\": {},\n  \"chunk_bytes\": {},\n  \"chunk_count\": {},\n  \"canonical_table_sha256\": \"{}\",\n  \"elapsed_ms\": {:.3}\n}}\n",
        table.version,
        table.whole_sha256,
        table.total_bytes,
        table.chunk_bytes,
        table.digests.len() / 32,
        table.table_sha256,
        table.elapsed_ms,
    );
    write_new(&receipt_output, receipt.as_bytes())?;
    print!("{receipt}");
    Ok(())
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file: File = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .wrap_err("Creating a new authoring output failed")?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn arguments() -> Result<(PathBuf, PathBuf, PathBuf)> {
    let mut source = None;
    let mut table = None;
    let mut receipt = None;
    let mut args = std::env::args_os().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--help" || flag == "-h" {
            println!(
                "generate_chunk_table --model <existing-orca.gguf> --table-output <new.bin> --receipt-output <new.json>"
            );
            std::process::exit(0);
        }
        let slot = match flag.to_str() {
            Some("--model") => &mut source,
            Some("--table-output") => &mut table,
            Some("--receipt-output") => &mut receipt,
            _ => bail!("Unknown authoring option; use --help"),
        };
        ensure!(slot.is_none(), "Duplicate authoring option");
        let value: OsString = args
            .next()
            .ok_or_else(|| eyre!("Authoring option is missing its path"))?;
        *slot = Some(PathBuf::from(value));
    }
    Ok((
        source.ok_or_else(|| eyre!("--model is required"))?,
        table.ok_or_else(|| eyre!("--table-output is required"))?,
        receipt.ok_or_else(|| eyre!("--receipt-output is required"))?,
    ))
}
