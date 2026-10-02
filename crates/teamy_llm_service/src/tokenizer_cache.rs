//! Reuse an unchanged tokenizer across resident requests without changing encode options.
//!
//! Length and modification time are freshness hints, not content integrity checks. A caller
//! that replaces a tokenizer while preserving both must explicitly invalidate its entry.

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::SystemTime,
};

use eyre::{WrapErr, ensure};
use tokenizers::Tokenizer;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FileStamp {
    length: u64,
    modified: SystemTime,
}

impl FileStamp {
    fn read(path: &Path) -> eyre::Result<Self> {
        let metadata = fs::metadata(path).wrap_err("Read tokenizer cache metadata")?;
        ensure!(metadata.is_file(), "Tokenizer cache path must be a file");
        Ok(Self {
            length: metadata.len(),
            modified: metadata
                .modified()
                .wrap_err("Read tokenizer modification time")?,
        })
    }
}

struct CacheEntry {
    stamp: FileStamp,
    tokenizer: Arc<Tokenizer>,
}

/// Clones share entries. Loading and invalidation are serialized to prevent duplicate
/// first loads or insertion of a stale entry after an explicit invalidation.
#[derive(Clone, Default)]
pub struct TokenizerCache {
    entries: Arc<Mutex<HashMap<PathBuf, CacheEntry>>>,
}

impl TokenizerCache {
    /// Return the unchanged tokenizer for this absolute path and current file stamp.
    ///
    /// This method does not change truncation, padding, special tokens or post-processing;
    /// the caller retains control over `encode(prompt, true)` and decoding options.
    /// A failed or changing replacement is not served through an older cached entry.
    pub fn get(&self, path: &Path) -> eyre::Result<Arc<Tokenizer>> {
        // Absolute paths avoid aliasing a relative spelling after a working-directory change.
        // This is lexical resolution, not a symlink/security-boundary assertion.
        let key = std::path::absolute(path).wrap_err("Resolve tokenizer cache path")?;
        let mut entries = self
            .entries
            .lock()
            .map_err(|_| eyre::eyre!("Tokenizer cache mutex is poisoned"))?;
        let stamp = match FileStamp::read(&key) {
            Ok(stamp) => stamp,
            Err(error) => {
                entries.remove(&key);
                return Err(error);
            }
        };
        if let Some(entry) = entries.get(&key)
            && entry.stamp == stamp
        {
            tracing::debug!(
                cache_hit = true,
                tokenizer_bytes = stamp.length,
                "Tokenizer cache hit"
            );
            return Ok(Arc::clone(&entry.tokenizer));
        }
        entries.remove(&key);
        let _span =
            tracing::info_span!("llm_tokenizer_load", tokenizer_bytes = stamp.length).entered();
        let tokenizer = Arc::new(
            Tokenizer::from_file(&key)
                .map_err(|error| eyre::eyre!("Load cached tokenizer: {error}"))?,
        );
        ensure!(
            stamp == FileStamp::read(&key)?,
            "Tokenizer file changed while loading; retry with a stable file"
        );
        entries.insert(
            key,
            CacheEntry {
                stamp,
                tokenizer: Arc::clone(&tokenizer),
            },
        );
        Ok(tokenizer)
    }

    /// Evict this path if present, even when its file no longer exists.
    /// Existing `Arc` owners remain usable; the next `get` reads the file afresh.
    pub fn invalidate(&self, path: &Path) -> eyre::Result<bool> {
        let key = std::path::absolute(path).wrap_err("Resolve tokenizer cache path")?;
        let mut entries = self
            .entries
            .lock()
            .map_err(|_| eyre::eyre!("Tokenizer cache mutex is poisoned"))?;
        Ok(entries.remove(&key).is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs::FileTimes, time::Duration};
    use tokenizers::processors::template::TemplateProcessing;

    fn write_fixture(path: &Path, hello_id: u32, extra_word: bool) {
        let world_id = if hello_id == 1 { 2 } else { 1 };
        let extra = if extra_word {
            ",\"longer_fixture_word\":4"
        } else {
            ""
        };
        let json = format!(
            r#"{{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{{"type":"WhitespaceSplit"}},"post_processor":null,"decoder":null,"model":{{"type":"WordLevel","vocab":{{"<unk>":0,"hello":{hello_id},"world":{world_id},"<bos>":3{extra}}},"unk_token":"<unk>"}}}}"#
        );
        let mut tokenizer = Tokenizer::from_bytes(json.as_bytes()).expect("fixture tokenizer");
        tokenizer.with_post_processor(Some(
            TemplateProcessing::builder()
                .try_single("<bos> $A")
                .expect("fixture template")
                .special_tokens(vec![("<bos>", 3)])
                .build()
                .expect("fixture processor"),
        ));
        tokenizer
            .save(path, false)
            .expect("write fixture tokenizer");
    }

    #[test]
    fn same_file_is_shared_and_special_token_encoding_is_preserved() {
        let directory = tempfile::tempdir().expect("fixture directory");
        let path = directory.path().join("tokenizer.json");
        write_fixture(&path, 1, false);
        let cache = TokenizerCache::default();
        let first = cache.get(&path).expect("first load");
        let again = cache.get(&path).expect("cache hit");
        let shared = cache.clone().get(&path).expect("shared cache hit");
        assert!(Arc::ptr_eq(&first, &again));
        assert!(Arc::ptr_eq(&first, &shared));
        let uncached = Tokenizer::from_file(&path).expect("independent fixture load");
        let expected = uncached.encode("hello", true).expect("uncached encode");
        let actual = first.encode("hello", true).expect("cached encode");
        assert_eq!(actual.get_ids(), expected.get_ids());
        assert_eq!(actual.get_ids(), &[3, 1]);
        assert_eq!(first.encode("hello", false).unwrap().get_ids(), &[1]);
    }

    #[test]
    fn changed_file_length_reloads_new_tokenizer_contents() {
        let directory = tempfile::tempdir().expect("fixture directory");
        let path = directory.path().join("tokenizer.json");
        write_fixture(&path, 1, false);
        let cache = TokenizerCache::default();
        let first = cache.get(&path).expect("first load");
        let length = fs::metadata(&path).unwrap().len();
        write_fixture(&path, 2, true);
        assert_ne!(fs::metadata(&path).unwrap().len(), length);
        let replacement = cache.get(&path).expect("replacement load");
        assert!(!Arc::ptr_eq(&first, &replacement));
        assert_eq!(
            replacement.encode("hello", true).unwrap().get_ids(),
            &[3, 2]
        );
        assert_eq!(first.encode("hello", true).unwrap().get_ids(), &[3, 1]);
    }

    #[test]
    fn changed_modification_stamp_reloads_even_when_length_is_unchanged() {
        let directory = tempfile::tempdir().expect("fixture directory");
        let path = directory.path().join("tokenizer.json");
        write_fixture(&path, 1, false);
        let cache = TokenizerCache::default();
        let first = cache.get(&path).expect("first load");
        let original = FileStamp::read(&path).unwrap();
        let changed_time = original.modified + Duration::from_secs(10);
        fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_times(FileTimes::new().set_modified(changed_time))
            .unwrap();
        let changed = FileStamp::read(&path).unwrap();
        assert_eq!(changed.length, original.length);
        assert_ne!(changed.modified, original.modified);
        let replacement = cache.get(&path).expect("reload touched file");
        assert!(!Arc::ptr_eq(&first, &replacement));
    }

    #[test]
    fn explicit_invalidation_is_shared_and_forces_a_fresh_load() {
        let directory = tempfile::tempdir().expect("fixture directory");
        let path = directory.path().join("tokenizer.json");
        write_fixture(&path, 1, false);
        let cache = TokenizerCache::default();
        let first = cache.get(&path).expect("first load");
        assert!(
            cache
                .clone()
                .invalidate(&path)
                .expect("invalidate shared entry")
        );
        assert!(!cache.invalidate(&path).expect("entry already absent"));
        let replacement = cache.get(&path).expect("load after invalidation");
        assert!(!Arc::ptr_eq(&first, &replacement));
    }

    #[test]
    fn malformed_replacement_does_not_return_a_stale_cached_tokenizer() {
        let directory = tempfile::tempdir().expect("fixture directory");
        let path = directory.path().join("tokenizer.json");
        write_fixture(&path, 1, false);
        let cache = TokenizerCache::default();
        let first = cache.get(&path).expect("first load");
        fs::write(&path, "invalid tokenizer json").unwrap();
        assert!(cache.get(&path).is_err());
        assert!(cache.get(&path).is_err());
        write_fixture(&path, 2, true);
        let repaired = cache.get(&path).expect("load repaired file");
        assert!(!Arc::ptr_eq(&first, &repaired));
        assert_eq!(repaired.encode("hello", true).unwrap().get_ids(), &[3, 2]);
    }
}
