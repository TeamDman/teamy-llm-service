//! Model-free regression fixtures for retaining the verified view without reopening its path.

use std::fs;
use std::sync::Arc;

use makepad_ai_llm::{GgufFile, GgufWeightLayout, MappedRegion, TensorType};
use tempfile::TempDir;

fn fixture() -> (TempDir, GgufFile, Arc<MappedRegion>, Vec<u8>) {
    // A valid tiny GGUF v3 containing one F32[2] tensor. fs::write closes its writable handle
    // before the Windows FILE_SHARE_READ mapping is acquired.
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("fixture.gguf");
    let name = b"fixture";
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes()); // tensor count
    bytes.extend_from_slice(&0u64.to_le_bytes()); // metadata count
    bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
    bytes.extend_from_slice(name);
    bytes.extend_from_slice(&1u32.to_le_bytes()); // tensor rank
    bytes.extend_from_slice(&2u64.to_le_bytes()); // F32 element count
    bytes.extend_from_slice(&(TensorType::F32 as u32).to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes()); // tensor offset relative to data
    bytes.resize(bytes.len().next_multiple_of(32), 0);
    bytes.extend_from_slice(&1.25f32.to_le_bytes());
    bytes.extend_from_slice(&(-3.0f32).to_le_bytes());
    fs::write(&source, &bytes).unwrap();
    let mut metadata = GgufFile::open(&source).unwrap();
    let mapping = Arc::new(MappedRegion::map_file(&source).unwrap());
    // A successful explicit-view load must not need this path, or quietly use it as fallback.
    metadata.path = directory.path().join("metadata-only-no-such-file.gguf");
    assert!(!metadata.path.exists());
    (directory, metadata, mapping, bytes)
}

#[test]
fn supplied_mapping_identity_data_and_lifetime_survive_a_nonexistent_metadata_path() {
    let (directory, metadata, mapping, bytes) = fixture();
    let layout = GgufWeightLayout::from_tensors(metadata.tensors.clone()).unwrap();
    let weak = Arc::downgrade(&mapping);
    let weights = layout
        .load_from_mapped_region(&metadata, 64, mapping.clone())
        .unwrap();
    assert!(Arc::ptr_eq(weights.ctx.ro_region().unwrap(), &mapping));
    assert_eq!(weights.ctx.ro_split(), bytes.len());
    assert_eq!(weights.ctx.data_at(0, bytes.len()).unwrap(), bytes);
    let id = weights.require_tensor_id("fixture").unwrap();
    let tensor = weights.ctx.tensor(id).unwrap();
    assert_eq!(
        tensor.data_offset,
        Some(usize::try_from(metadata.data_offset).unwrap())
    );
    drop(mapping);
    assert!(
        weak.upgrade().is_some(),
        "weight context must retain the same view"
    );
    let data = weights
        .ctx
        .data_at(usize::try_from(metadata.data_offset).unwrap(), 8)
        .unwrap();
    assert_eq!(&data[..4], &1.25f32.to_le_bytes());
    assert_eq!(&data[4..], &(-3.0f32).to_le_bytes());
    drop(weights);
    assert!(
        weak.upgrade().is_none(),
        "last weight owner must release the view"
    );
    drop(directory);
}

#[test]
fn supplied_mapping_length_mismatch_is_rejected_before_any_path_fallback() {
    let (directory, mut metadata, mapping, _) = fixture();
    let layout = GgufWeightLayout::from_tensors(metadata.tensors.clone()).unwrap();
    metadata.file_size += 1;
    let error = layout
        .load_from_mapped_region(&metadata, 64, mapping.clone())
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("mapped weight view length differs from parsed GGUF")
    );
    assert!(!metadata.path.exists());
    assert_eq!(Arc::strong_count(&mapping), 1);
    drop(mapping);
    drop(directory);
}

#[test]
fn supplied_mapping_out_of_range_tensor_is_rejected_without_reopening() {
    let (directory, metadata, mapping, _) = fixture();
    let mut tensor = metadata.tensors[0].clone();
    tensor.offset = metadata.file_size;
    let layout = GgufWeightLayout::from_tensors([tensor]).unwrap();
    let error = layout
        .load_from_mapped_region(&metadata, 64, mapping.clone())
        .unwrap_err();
    assert!(error.to_string().contains("tensor range"));
    assert!(error.to_string().contains("exceeds file size"));
    assert!(!metadata.path.exists());
    assert_eq!(Arc::strong_count(&mapping), 1);
    drop(mapping);
    drop(directory);
}
