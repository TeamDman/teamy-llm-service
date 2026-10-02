#![cfg(windows)]

use makepad_ai_loader::MappedRegion;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

fn fixture_path() -> PathBuf {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "makepad-windows-mmap-café-{}-{timestamp}-{}.bin",
        std::process::id(),
        NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)
    ))
}

struct Fixture {
    path: PathBuf,
}

impl Fixture {
    fn new(bytes: &[u8]) -> Self {
        let path = fixture_path();
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .expect("create unique fixture");
        file.write_all(bytes).expect("write fixture");
        Self { path }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        // Remove only the exact temporary file created by this fixture.
        let _ = std::fs::remove_file(&self.path);
    }
}

#[test]
fn maps_binary_bytes_and_exact_non_page_length() {
    let bytes = (0..4101)
        .map(|index| (index % 256) as u8)
        .collect::<Vec<_>>();
    let fixture = Fixture::new(&bytes);
    let mapping = MappedRegion::map(&fixture.path).expect("map fixture");
    assert_eq!(mapping.len(), bytes.len());
    assert!(!mapping.is_empty());
    assert_eq!(mapping.as_slice(), bytes);
    mapping.prefault();
    assert_eq!(mapping.as_slice(), bytes);
    assert_eq!(
        mapping.residency(),
        (0, 0),
        "Windows residency is unmeasured"
    );
}

#[test]
fn one_byte_has_no_exposed_padded_tail() {
    let fixture = Fixture::new(&[0]);
    let mapping = MappedRegion::map_file(&fixture.path).expect("map one byte");
    assert_eq!(mapping.len(), 1);
    assert_eq!(mapping.as_slice(), &[0]);
    mapping.prefault();
}

#[test]
fn empty_file_is_rejected_without_retaining_a_lock() {
    let fixture = Fixture::new(&[]);
    let error = MappedRegion::map_file(&fixture.path).expect_err("reject empty file");
    assert!(error.contains("empty file"));
    std::fs::write(&fixture.path, [1]).expect("failed mapping releases file");
}

#[test]
fn missing_file_reports_an_error() {
    let path = fixture_path();
    assert!(!path.exists());
    let error = MappedRegion::map_file(&path).expect_err("reject absent file");
    assert!(error.contains("failed to open"));
}

#[test]
fn live_mapping_denies_write_delete_and_replacement_but_allows_readers() {
    let fixture = Fixture::new(b"original weights");
    let mapping = MappedRegion::map_file(&fixture.path).expect("map fixture");
    let reader = File::open(&fixture.path).expect("additional reader is allowed");
    drop(reader);
    assert!(OpenOptions::new().write(true).open(&fixture.path).is_err());
    assert!(OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(&fixture.path)
        .is_err());
    assert!(std::fs::remove_file(&fixture.path).is_err());
    let replacement = fixture_path();
    assert!(std::fs::rename(&fixture.path, &replacement).is_err());
    assert!(!replacement.exists());
    assert_eq!(mapping.as_slice(), b"original weights");

    drop(mapping);
    std::fs::write(&fixture.path, b"new weights").expect("drop permits writing again");
    std::fs::remove_file(&fixture.path).expect("drop permits deleting again");
}

#[test]
fn preexisting_writer_prevents_mapping() {
    let fixture = Fixture::new(b"original");
    let writer = OpenOptions::new()
        .write(true)
        .open(&fixture.path)
        .expect("open writer before mapping");
    assert!(MappedRegion::map_file(&fixture.path).is_err());
    drop(writer);
    let mapping = MappedRegion::map_file(&fixture.path).expect("map after writer closes");
    assert_eq!(mapping.as_slice(), b"original");
}

#[test]
fn shared_mapping_keeps_file_locked_until_last_owner_drops() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<MappedRegion>();
    let fixture = Fixture::new(b"shared immutable weights");
    let mapping = Arc::new(MappedRegion::map_file(&fixture.path).expect("map fixture"));
    let another_owner = Arc::clone(&mapping);
    drop(mapping);
    assert!(OpenOptions::new().write(true).open(&fixture.path).is_err());
    let copied = std::thread::spawn(move || another_owner.as_slice().to_vec())
        .join()
        .expect("mapping is readable on another thread");
    assert_eq!(copied, b"shared immutable weights");
    std::fs::write(&fixture.path, b"unlocked").expect("last owner releases file");
}
