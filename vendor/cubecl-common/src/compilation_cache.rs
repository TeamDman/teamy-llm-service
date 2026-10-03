use std::{
    boxed::Box,
    cell::RefCell,
    fs::{self, File},
    io::{self, Cursor, Write},
    path::{Path, PathBuf},
    string::{String, ToString},
    vec::Vec,
};

use ciborium::{de::from_reader, ser::into_writer};
use hashbrown::HashMap;
use serde::Serialize;

use crate::cache::{
    Cache, CacheError, CacheKey, CacheOption, CacheValue, Entry, sanitize_path_segment,
};

/// The in-memory cache used by the chunked kernel cache.
/// Box ensures values aren't moved when inserting new elements, so we don't need to keep it
/// locked for reads
type InMemoryCache<K, V> = RefCell<HashMap<K, Box<V>>>;

/// A chunked cache for compilation artifacts. Uses a human readable table of contents, with binary
/// storage for the compiled kernel.
#[derive(Debug)]
pub struct CompilationCache<K: CacheKey, V: CacheValue> {
    toc: Cache<K, String>,
    in_memory_cache: InMemoryCache<K, V>,
    current_chunk: Option<File>,
    current_chunk_path_normalized: String,
    cache_root: PathBuf,
}

/// Error related to caching.
#[derive(Debug)]
pub enum CompilationCacheError<K: Serialize, V: Serialize> {
    /// Can't insert an entry with the same key, but different value.
    #[allow(missing_docs)]
    DuplicatedKey {
        key: K,
        value_previous: V,
        value_updated: V,
    },
    /// The table of contents cache had an error
    #[allow(missing_docs)]
    TocError(CacheError<K, String>),
    /// Persistence failed; the entry remains usable in the in-memory cache.
    IoError(io::Error),
    /// The value could not be serialized for persistence.
    SerializationError(String),
}

impl<K: CacheKey, V: CacheValue> CompilationCache<K, V> {
    /// Create a new cache and load the data from the provided path if it exists.
    #[cfg_attr(feature="tracing", tracing::instrument(
        level = "trace",
        skip(path),
        fields(path = ?path.as_ref())))]
    pub fn new<P: AsRef<Path>>(path: P, option: CacheOption) -> Self {
        let (_, name, version, root, _) = option.clone().resolve();
        let path = path.as_ref();
        let toc_path = path.join("toc");
        // `.cbor` suffix (was `.bin` with bincode) invalidates old on-disk chunks.
        let chunk_path = Path::new("chunk0.cbor"); // Split later

        let cache_root = get_persistent_cache_root(path, root, name, version);
        let chunk_path = get_persistent_chunk_file_path(chunk_path, &cache_root);

        let in_memory_cache = InMemoryCache::default();
        let toc = Cache::new(toc_path, option);

        // A damaged cache is expendable. Leave it untouched and continue without
        // persistence instead of appending new entries after a malformed record.
        let readable = match chunk_path.try_exists() {
            Ok(true) => match Self::read_chunk(&chunk_path, &in_memory_cache) {
                Ok(()) => true,
                Err(err) => {
                    log::warn!(
                        "Unable to read compilation cache {chunk_path:?}; persistence disabled: {err}"
                    );
                    false
                }
            },
            Ok(false) => true,
            Err(err) => {
                log::warn!(
                    "Unable to inspect compilation cache {chunk_path:?}; persistence disabled: {err}"
                );
                false
            }
        };
        let current_chunk = if readable {
            match open_chunk_writable(&chunk_path) {
                Ok(file) => Some(file),
                Err(err) => {
                    log::warn!(
                        "Unable to open compilation cache {chunk_path:?}; persistence disabled: {err}"
                    );
                    None
                }
            }
        } else {
            None
        };

        Self {
            toc,
            in_memory_cache,
            current_chunk,
            current_chunk_path_normalized: normalized_path(
                chunk_path
                    .strip_prefix(&cache_root)
                    .expect("Should contain root"),
            ),
            cache_root,
        }
    }

    /// Fetch an item from the cache.
    pub fn get(&self, key: &K) -> Option<&V> {
        if let Some(value) = self.get_ref_unsafe(key) {
            return Some(value);
        }
        let chunk = self.toc.get(key)?;
        // TOC values are cache data, never authority to read outside this root.
        let mut components = Path::new(chunk).components();
        if !matches!(components.next(), Some(std::path::Component::Normal(_)))
            || components.next().is_some()
        {
            log::warn!("Ignoring compilation cache entry with invalid chunk location");
            return None;
        }
        if let Err(err) = Self::read_chunk(&self.cache_root.join(chunk), &self.in_memory_cache) {
            log::warn!("Unable to read compilation cache chunk; treating entry as a miss: {err}");
            return None;
        }
        self.get_ref_unsafe(key)
    }

    /// Unsafely construct a reference of lifetime 'self, ignoring the read guard.
    /// # Safety
    ///
    /// This is only safe because we never remove or update elements, so boxes remain valid for the
    /// entire lifetime of `self`
    fn get_ref_unsafe<'a>(&'a self, key: &K) -> Option<&'a V> {
        let cache = self.in_memory_cache.borrow();
        if let Some(value) = cache.get(key) {
            let value = unsafe { core::mem::transmute::<&'_ V, &'a V>(&**value) };
            Some(value)
        } else {
            None
        }
    }

    fn read_chunk(chunk: &Path, cache: &InMemoryCache<K, V>) -> io::Result<()> {
        let data = fs::read(chunk)?;
        let mut cursor = Cursor::new(data);
        // Collect new entries first so we only need to lock once everything is loaded
        let mut new_entries = HashMap::<K, Box<V>>::new();
        loop {
            let pos = cursor.position() as usize;
            let total_len = cursor.get_ref().len();
            if pos >= total_len {
                break;
            }
            let entry = from_reader::<Entry<K, V>, _>(&mut cursor)
                .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err.to_string()))?;
            if let Some(existing) = new_entries.get(&entry.key)
                && existing.as_ref() != &entry.value
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Conflicting compilation cache records",
                ));
            }
            new_entries
                .entry(entry.key)
                .or_insert_with(|| Box::new(entry.value));
        }

        let mut cache = cache.borrow_mut();
        for (key, value) in &new_entries {
            if let Some(existing) = cache.get(key)
                && existing.as_ref() != value.as_ref()
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Conflicting compilation cache records",
                ));
            }
        }
        for (key, value) in new_entries {
            // Existing boxes must never be replaced: get_ref_unsafe exposes references
            // that remain valid for this cache's entire lifetime.
            cache.entry(key).or_insert(value);
        }
        Ok(())
    }

    /// Insert a new item to the cache.
    ///
    /// Conflicting values return an error. Persistence failures retain the live
    /// in-memory entry and disable further writes to the damaged chunk.
    pub fn insert(&mut self, key: K, value: V) -> Result<(), CompilationCacheError<K, V>> {
        if let Some(existing) = self.get_ref_unsafe(&key) {
            if existing != &value {
                return Err(CompilationCacheError::DuplicatedKey {
                    key,
                    value_previous: existing.clone(),
                    value_updated: value,
                });
            } else {
                return Ok(());
            }
        }

        let entry = Entry {
            key: key.clone(),
            value,
        };
        let persistence = if let Some(chunk) = &mut self.current_chunk {
            let mut bytes = Vec::new();
            into_writer(&entry, &mut bytes)
                .map_err(|err| CompilationCacheError::SerializationError(err.to_string()))
                .and_then(|()| {
                    chunk
                        .write_all(&bytes)
                        .map_err(CompilationCacheError::IoError)
                })
                .map(|()| true)
        } else {
            Ok(false)
        };
        self.in_memory_cache
            .borrow_mut()
            .insert(entry.key, Box::new(entry.value));
        match persistence {
            Ok(true) => self
                .toc
                .insert(key, self.current_chunk_path_normalized.clone())
                .map_err(CompilationCacheError::TocError)?,
            Ok(false) => {}
            Err(err) => {
                // A short/failed append may leave a torn record. Do not write more
                // to this chunk, and do not advertise the failed append in the TOC.
                self.current_chunk = None;
                return Err(err);
            }
        }

        Ok(())
    }
}

fn get_persistent_cache_root(
    path_partial: impl AsRef<Path>,
    root: PathBuf,
    name: String,
    version: String,
) -> PathBuf {
    let path_partial = path_partial.as_ref();
    let mut path = root
        .join(sanitize_path_segment(&name))
        .join(sanitize_path_segment(&version));

    for segment in path_partial.iter() {
        // Skip the name directory since it resets the previous path segments.
        if segment == "/" {
            continue;
        }
        path = path.join(sanitize_path_segment(segment.to_str().unwrap()));
    }

    std::path::absolute(path).expect("Not empty, so can't fail")
}

fn get_persistent_chunk_file_path<P: AsRef<Path>>(path_partial: P, chunks_root: &Path) -> PathBuf {
    let path_partial: &Path = path_partial.as_ref();

    let mut path = chunks_root.to_path_buf();

    for segment in path_partial.iter() {
        // Skip the name directory since it resets the previous path segments.
        if segment == "/" {
            continue;
        }
        path = path.join(sanitize_path_segment(segment.to_str().unwrap()));
    }

    std::path::absolute(path).expect("Not empty, so can't fail")
}

fn normalized_path(path: &Path) -> String {
    let path = path.to_string_lossy().to_string();
    path.replace("\\", "/")
}

fn open_chunk_writable(path: &Path) -> io::Result<File> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    File::options().append(true).create(true).open(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::format;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct TestRoot(PathBuf);

    impl TestRoot {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "cubecl-compilation-cache-{}-{nonce}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed),
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn options(&self) -> CacheOption {
            CacheOption::default()
                .root(&self.0)
                .name("test")
                .version("v1")
        }

        fn chunk(&self) -> PathBuf {
            self.0.join("test/v1/ptx/chunk0.cbor")
        }
    }

    impl Drop for TestRoot {
        fn drop(&mut self) {
            // This unique directory is created and owned entirely by the fixture.
            fs::remove_dir_all(&self.0).ok();
        }
    }

    #[test]
    fn valid_entries_survive_a_fresh_cache_instance() {
        let root = TestRoot::new();
        let mut cache = CompilationCache::<u32, String>::new("ptx", root.options());
        cache.insert(1, "compiled-kernel".into()).unwrap();
        drop(cache);
        let reloaded = CompilationCache::<u32, String>::new("ptx", root.options());
        assert_eq!(
            reloaded.get(&1).map(String::as_str),
            Some("compiled-kernel")
        );
    }

    #[test]
    fn corrupted_chunk_is_untouched_and_persistence_is_disabled() {
        let root = TestRoot::new();
        fs::create_dir_all(root.chunk().parent().unwrap()).unwrap();
        let mut corrupted = Vec::new();
        into_writer(
            &Entry {
                key: 1_u32,
                value: "prefix".to_string(),
            },
            &mut corrupted,
        )
        .unwrap();
        corrupted.push(0xa1); // An incomplete CBOR map following a valid record.
        fs::write(root.chunk(), &corrupted).unwrap();

        let mut cache = CompilationCache::<u32, String>::new("ptx", root.options());
        assert!(cache.current_chunk.is_none());
        assert!(cache.get(&1).is_none());
        cache.insert(2, "live-only".into()).unwrap();
        assert_eq!(cache.get(&2).map(String::as_str), Some("live-only"));
        assert_eq!(fs::read(root.chunk()).unwrap(), corrupted);
    }

    #[test]
    fn unreadable_chunk_is_a_memory_only_cache() {
        let root = TestRoot::new();
        fs::create_dir_all(root.chunk()).unwrap();
        let mut cache = CompilationCache::<u32, String>::new("ptx", root.options());
        assert!(cache.current_chunk.is_none());
        cache.insert(1, "live-only".into()).unwrap();
        assert_eq!(cache.get(&1).map(String::as_str), Some("live-only"));
        assert!(root.chunk().is_dir());
    }

    #[test]
    fn missing_chunk_named_by_toc_is_a_miss() {
        let root = TestRoot::new();
        let mut cache = CompilationCache::<u32, String>::new("ptx", root.options());
        cache.toc.insert(1, "missing.cbor".into()).unwrap();
        assert!(cache.get(&1).is_none());
    }

    #[test]
    fn toc_cannot_read_a_chunk_outside_the_cache_root() {
        let root = TestRoot::new();
        let mut cache = CompilationCache::<u32, String>::new("ptx", root.options());
        let outside = cache.cache_root.parent().unwrap().join("outside.cbor");
        let mut bytes = Vec::new();
        into_writer(
            &Entry {
                key: 1_u32,
                value: "outside".to_string(),
            },
            &mut bytes,
        )
        .unwrap();
        fs::write(outside, bytes).unwrap();
        cache.toc.insert(1, "../outside.cbor".into()).unwrap();
        assert!(cache.get(&1).is_none());
    }

    #[test]
    fn failed_append_retains_memory_entry_and_disables_further_writes() {
        let root = TestRoot::new();
        let mut cache = CompilationCache::<u32, String>::new("ptx", root.options());
        cache.current_chunk = Some(File::open(root.chunk()).unwrap()); // Read-only handle.
        assert!(matches!(
            cache.insert(1, "live-only".into()),
            Err(CompilationCacheError::IoError(_)),
        ));
        assert!(cache.current_chunk.is_none());
        assert_eq!(cache.get(&1).map(String::as_str), Some("live-only"));
        assert!(cache.toc.get(&1).is_none());
        cache.insert(2, "also-live".into()).unwrap();
        assert_eq!(cache.get(&2).map(String::as_str), Some("also-live"));
        assert!(fs::read(root.chunk()).unwrap().is_empty());
    }

    #[test]
    fn refreshing_a_chunk_preserves_existing_box_addresses() {
        let root = TestRoot::new();
        fs::create_dir_all(root.chunk().parent().unwrap()).unwrap();
        let memory = InMemoryCache::<u32, String>::default();
        memory.borrow_mut().insert(1, Box::new("first".into()));
        let address = memory.borrow().get(&1).unwrap().as_ref() as *const String;
        let mut bytes = Vec::new();
        for (key, value) in [(1_u32, "first"), (2, "second")] {
            into_writer(
                &Entry {
                    key,
                    value: value.to_string(),
                },
                &mut bytes,
            )
            .unwrap();
        }
        fs::write(root.chunk(), bytes).unwrap();
        CompilationCache::<u32, String>::read_chunk(&root.chunk(), &memory).unwrap();
        let refreshed = memory.borrow();
        assert_eq!(
            refreshed.get(&1).unwrap().as_ref() as *const String,
            address
        );
        assert_eq!(refreshed.get(&2).unwrap().as_str(), "second");
    }

    #[test]
    fn conflicting_records_do_not_publish_any_entries() {
        let root = TestRoot::new();
        fs::create_dir_all(root.chunk().parent().unwrap()).unwrap();
        let memory = InMemoryCache::<u32, String>::default();
        let mut bytes = Vec::new();
        for value in ["first", "conflict"] {
            into_writer(
                &Entry {
                    key: 1_u32,
                    value: value.to_string(),
                },
                &mut bytes,
            )
            .unwrap();
        }
        fs::write(root.chunk(), bytes).unwrap();
        assert!(CompilationCache::<u32, String>::read_chunk(&root.chunk(), &memory).is_err());
        assert!(memory.borrow().is_empty());
    }
}
