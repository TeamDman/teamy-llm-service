//! Explicit model preference is independent of download/registration order.

use std::{
    fs::{self, File},
    io::{self, Read, Write},
    path::{Path, PathBuf},
};

use facet::Facet;
use teamy_llm_burn_qwen35::{
    model::{
        DEFAULT_LLM_MODEL_NAME, HF_CONFIG_FILE_NAME, KnownLlmModel, LlmManagedModelMetadata,
        MODEL_METADATA_FILE_NAME, TOKENIZER_CONFIG_FILE_NAME, TOKENIZER_FILE_NAME,
        inspect_model_dir, known_llm_model, list_registered_model_dirs, managed_model_dir,
    },
    paths::{AppHome, CacheHome},
};
use teamy_llm_core::{ServiceError, ServiceErrorKind};

const DEFAULT_MODEL_FILE: &str = "default-model.json";
const MAX_RECORD_BYTES: u64 = 4096;

/// A selection report, rather than a claim that a CUDA runtime has been loaded.
#[derive(Clone, Debug, PartialEq, Eq, Facet)]
#[facet(rename_all = "kebab-case")]
pub struct ModelDefaultInfo {
    pub version: u32,
    pub model_name: String,
    pub model_dir: String,
    /// `explicit`, `registered`, or `built-in`.
    pub origin: String,
}

#[derive(Clone, Debug, Facet)]
#[facet(rename_all = "kebab-case", deny_unknown_fields)]
struct DefaultModelRecord {
    version: u32,
    model_name: String,
}

pub(super) fn default_model(
    app_home: &AppHome,
    cache_home: &CacheHome,
) -> Result<ModelDefaultInfo, ServiceError> {
    if let Some(record) = read_record(app_home)? {
        let known = record_model(&record)?;
        let root = managed_model_dir(cache_home, known.name);
        validate_prepared_model(&root, known)?;
        return Ok(selection_info(known.name, &root, "explicit"));
    }
    let registered = list_registered_model_dirs(app_home).map_err(internal)?;
    if let Some(root) = registered.first() {
        let artifacts = inspect_model_dir(root).map_err(not_prepared)?;
        return Ok(selection_info(
            &artifacts.metadata.model_name,
            root,
            "registered",
        ));
    }
    Ok(selection_info(
        DEFAULT_LLM_MODEL_NAME,
        &managed_model_dir(cache_home, DEFAULT_LLM_MODEL_NAME),
        "built-in",
    ))
}

pub(super) fn set_default_model(
    app_home: &AppHome,
    cache_home: &CacheHome,
    model: &str,
) -> Result<ModelDefaultInfo, ServiceError> {
    let known = known_llm_model(model).ok_or_else(|| {
        invalid(format!(
            "Unknown managed model `{model}`; use `teamy-llm model list` to select a model name"
        ))
    })?;
    let root = managed_model_dir(cache_home, known.name);
    validate_prepared_model(&root, known)?;
    let record = DefaultModelRecord {
        version: 1,
        model_name: known.name.into(),
    };
    write_record(app_home, &record)?;
    Ok(selection_info(known.name, &root, "explicit"))
}

/// A resident request must not repeatedly parse the large tokenizer just to select its model.
/// These presence/metadata checks retain fail-closed selection; runtime loading still performs
/// the complete tokenizer and frozen weight identity checks.
pub(super) fn explicit_model_dir(
    app_home: &AppHome,
    cache_home: &CacheHome,
) -> Result<Option<PathBuf>, ServiceError> {
    let Some(record) = read_record(app_home)? else {
        return Ok(None);
    };
    let known = record_model(&record)?;
    let root = managed_model_dir(cache_home, known.name);
    validate_model_identity_and_presence(&root, known)?;
    Ok(Some(root))
}

fn selection_info(model: &str, root: &Path, origin: &str) -> ModelDefaultInfo {
    ModelDefaultInfo {
        version: 1,
        model_name: model.into(),
        model_dir: root.display().to_string(),
        origin: origin.into(),
    }
}

fn read_record(app_home: &AppHome) -> Result<Option<DefaultModelRecord>, ServiceError> {
    let mut file = match File::open(app_home.file_path(DEFAULT_MODEL_FILE)) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(internal(error)),
    };
    let mut bytes = Vec::new();
    Read::by_ref(&mut file)
        .take(MAX_RECORD_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(internal)?;
    if bytes.len() as u64 > MAX_RECORD_BYTES {
        return Err(invalid("default-model.json exceeds its 4096-byte limit"));
    }
    let record: DefaultModelRecord = facet_json::from_slice(&bytes).map_err(|error| {
        invalid(format!(
            "Invalid default-model.json: {error}; use `teamy-llm model default set <model>` to replace it"
        ))
    })?;
    if record.version != 1 {
        return Err(invalid(format!(
            "Unsupported default-model.json version {}; supported version is 1",
            record.version
        )));
    }
    // An unknown name must not quietly reinstate the old registration-based fallback.
    record_model(&record)?;
    Ok(Some(record))
}

fn record_model(record: &DefaultModelRecord) -> Result<&'static KnownLlmModel, ServiceError> {
    known_llm_model(&record.model_name).ok_or_else(|| {
        invalid(format!(
            "default-model.json selects unknown managed model `{}`; choose an installed model explicitly",
            record.model_name
        ))
    })
}

fn validate_prepared_model(root: &Path, known: &KnownLlmModel) -> Result<(), ServiceError> {
    validate_model_identity_and_presence(root, known)?;
    let artifacts = inspect_model_dir(root).map_err(not_prepared)?;
    validate_metadata(&artifacts.metadata, known)
}

fn validate_model_identity_and_presence(
    root: &Path,
    known: &KnownLlmModel,
) -> Result<(), ServiceError> {
    let metadata: LlmManagedModelMetadata = facet_json::from_slice(
        &fs::read(root.join(MODEL_METADATA_FILE_NAME)).map_err(not_prepared)?,
    )
    .map_err(not_prepared)?;
    validate_metadata(&metadata, known)?;
    let model_file = if known.name == teamy_llm_makepad_gguf::ORCA_MODEL_NAME {
        known.model_file_name
    } else {
        teamy_llm_burn_qwen35::model::MODEL_FILE_NAME
    };
    for file in [
        model_file,
        TOKENIZER_FILE_NAME,
        TOKENIZER_CONFIG_FILE_NAME,
        HF_CONFIG_FILE_NAME,
    ] {
        if !root.join(file).is_file() {
            return Err(not_prepared(format!(
                "Selected default model `{}` is missing required artifact `{file}`; prepare it or select another model",
                known.name
            )));
        }
    }
    Ok(())
}

fn validate_metadata(
    metadata: &LlmManagedModelMetadata,
    known: &KnownLlmModel,
) -> Result<(), ServiceError> {
    if metadata.model_name != known.name
        || metadata.architecture != known.architecture
        || metadata.quantization != known.quantization
        || metadata.model_file_name != known.model_file_name
    {
        return Err(not_prepared(format!(
            "Managed directory for `{}` has incompatible model identity or metadata",
            known.name
        )));
    }
    Ok(())
}

fn write_record(app_home: &AppHome, record: &DefaultModelRecord) -> Result<(), ServiceError> {
    let mut bytes = facet_json::to_vec_pretty(record).map_err(internal)?;
    bytes.push(b'\n');
    app_home.ensure_dir().map_err(internal)?;
    // Same-directory staging keeps replacement on one filesystem. Never remove the old record.
    let mut staging = tempfile::Builder::new()
        .prefix("default-model.")
        .suffix(".tmp")
        .tempfile_in(&app_home.0)
        .map_err(internal)?;
    staging.write_all(&bytes).map_err(internal)?;
    staging.flush().map_err(internal)?;
    staging.as_file().sync_all().map_err(internal)?;
    let mut staging = staging.into_temp_path();
    atomic_replace(&staging, &app_home.file_path(DEFAULT_MODEL_FILE)).map_err(internal)?;
    staging.disable_cleanup(true);
    #[cfg(unix)]
    File::open(&app_home.0)
        .and_then(|directory| directory.sync_all())
        .map_err(internal)?;
    Ok(())
}

#[cfg(windows)]
fn atomic_replace(staging: &Path, destination: &Path) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        FILE_ATTRIBUTE_NORMAL, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH, MoveFileExW,
        SetFileAttributesW,
    };
    fn wide(path: &Path) -> io::Result<Vec<u16>> {
        let mut value = path.as_os_str().encode_wide().collect::<Vec<_>>();
        if value.contains(&0) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Default model path contains a NUL character",
            ));
        }
        value.push(0);
        Ok(value)
    }
    let staging = wide(staging)?;
    let destination = wide(destination)?;
    // NamedTempFile uses FILE_ATTRIBUTE_TEMPORARY; clear that hint before publishing a durable
    // configuration file. A failure here affects only the new staging file.
    if unsafe { SetFileAttributesW(staging.as_ptr(), FILE_ATTRIBUTE_NORMAL) } == 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: both buffers are NUL-terminated and live for the call. MoveFileExW creates no
    // owned handle; write-through replacement never deletes the destination beforehand.
    let succeeded = unsafe {
        MoveFileExW(
            staging.as_ptr(),
            destination.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if succeeded == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(not(windows))]
fn atomic_replace(staging: &Path, destination: &Path) -> io::Result<()> {
    fs::rename(staging, destination)
}

fn invalid(message: impl Into<String>) -> ServiceError {
    ServiceError::new(ServiceErrorKind::InvalidRequest, message)
}

fn internal(error: impl std::fmt::Display) -> ServiceError {
    ServiceError::new(
        ServiceErrorKind::Internal,
        format!("Default model configuration: {error}"),
    )
}

fn not_prepared(error: impl std::fmt::Display) -> ServiceError {
    ServiceError::new(
        ServiceErrorKind::ModelNotPrepared,
        format!("Default model is not prepared: {error}"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TeamyLlmService;
    use teamy_llm_burn_qwen35::model::{
        KNOWN_LLM_MODELS, MODEL_FILE_NAME, add_registered_model_dir, write_model_metadata,
    };
    use teamy_llm_core::{ModelSelector, PrepareModelRequest};
    use tokenizers::{Tokenizer, models::bpe::BPE};

    fn fixture_model(cache: &CacheHome, index: usize) -> PathBuf {
        let known = &KNOWN_LLM_MODELS[index];
        let root = managed_model_dir(cache, known.name);
        fs::create_dir_all(&root).unwrap();
        let model_file = if index == 1 {
            known.model_file_name
        } else {
            MODEL_FILE_NAME
        };
        fs::write(root.join(model_file), b"synthetic inventory fixture").unwrap();
        Tokenizer::new(BPE::default())
            .save(root.join(TOKENIZER_FILE_NAME), false)
            .unwrap();
        fs::write(root.join(TOKENIZER_CONFIG_FILE_NAME), b"{}").unwrap();
        fs::write(root.join(HF_CONFIG_FILE_NAME), b"{}").unwrap();
        write_model_metadata(&root, known, false).unwrap();
        root
    }

    fn homes(root: &Path) -> (AppHome, CacheHome) {
        (AppHome(root.join("app")), CacheHome(root.join("cache")))
    }

    #[test]
    fn no_record_preserves_built_in_fallback_without_writes() {
        let temp = tempfile::tempdir().unwrap();
        let (app, cache) = homes(temp.path());
        let service = TeamyLlmService::new(app.clone(), cache.clone());
        let selected = service.default_model().unwrap();
        assert_eq!(selected.model_name, DEFAULT_LLM_MODEL_NAME);
        assert_eq!(selected.origin, "built-in");
        assert_eq!(
            service
                .resolve_model_dir(&ModelSelector::default())
                .unwrap(),
            managed_model_dir(&cache, DEFAULT_LLM_MODEL_NAME)
        );
        assert!(!app.0.exists());
        assert!(!cache.0.exists());
    }

    #[test]
    fn no_record_preserves_first_registered_directory() {
        let temp = tempfile::tempdir().unwrap();
        let (app, cache) = homes(temp.path());
        let old = fixture_model(&cache, 0);
        let new = fixture_model(&cache, 1);
        add_registered_model_dir(&app, &old).unwrap();
        add_registered_model_dir(&app, &new).unwrap();
        let service = TeamyLlmService::new(app.clone(), cache);
        let selected = service.default_model().unwrap();
        assert_eq!(selected.origin, "registered");
        assert_eq!(selected.model_name, KNOWN_LLM_MODELS[1].name);
        assert_eq!(
            service
                .resolve_model_dir(&ModelSelector::default())
                .unwrap(),
            new
        );
        assert!(!app.file_path(DEFAULT_MODEL_FILE).exists());
    }

    #[test]
    fn selection_persists_and_overrides_registration_order() {
        let temp = tempfile::tempdir().unwrap();
        let (app, cache) = homes(temp.path());
        let old = fixture_model(&cache, 0);
        let new = fixture_model(&cache, 1);
        add_registered_model_dir(&app, &old).unwrap();
        let service = TeamyLlmService::new(app.clone(), cache.clone());
        let selected = service.set_default_model(KNOWN_LLM_MODELS[1].name).unwrap();
        assert_eq!(selected.origin, "explicit");
        add_registered_model_dir(&app, &old).unwrap();
        let restarted = TeamyLlmService::new(app, cache);
        assert_eq!(restarted.default_model().unwrap(), selected);
        assert_eq!(
            restarted
                .resolve_model_dir(&ModelSelector::default())
                .unwrap(),
            new
        );
    }

    #[test]
    fn selected_name_is_canonical_and_record_does_not_store_a_directory() {
        let temp = tempfile::tempdir().unwrap();
        let (app, cache) = homes(temp.path());
        fixture_model(&cache, 1);
        let service = TeamyLlmService::new(app.clone(), cache);
        let selected = service
            .set_default_model(&format!(" {} ", KNOWN_LLM_MODELS[1].name.to_uppercase()))
            .unwrap();
        assert_eq!(selected.model_name, KNOWN_LLM_MODELS[1].name);
        let bytes = fs::read_to_string(app.file_path(DEFAULT_MODEL_FILE)).unwrap();
        assert!(bytes.contains("\"model-name\""));
        assert!(!bytes.contains("model-dir"));
        assert!(!bytes.contains(temp.path().to_str().unwrap()));
    }

    #[test]
    fn rejected_selection_does_not_create_or_replace_configuration() {
        let temp = tempfile::tempdir().unwrap();
        let (app, cache) = homes(temp.path());
        fixture_model(&cache, 0);
        let service = TeamyLlmService::new(app.clone(), cache.clone());
        for name in ["", "../unknown", KNOWN_LLM_MODELS[1].name] {
            assert!(service.set_default_model(name).is_err());
        }
        assert!(!app.0.exists());
        service.set_default_model(KNOWN_LLM_MODELS[0].name).unwrap();
        let original = fs::read(app.file_path(DEFAULT_MODEL_FILE)).unwrap();
        let wrong = fixture_model(&cache, 1);
        write_model_metadata(&wrong, &KNOWN_LLM_MODELS[0], false).unwrap();
        assert!(service.set_default_model(KNOWN_LLM_MODELS[1].name).is_err());
        assert!(service.set_default_model("unknown").is_err());
        assert_eq!(
            fs::read(app.file_path(DEFAULT_MODEL_FILE)).unwrap(),
            original
        );
    }

    #[test]
    fn valid_selection_atomically_replaces_previous_selection() {
        let temp = tempfile::tempdir().unwrap();
        let (app, cache) = homes(temp.path());
        fixture_model(&cache, 0);
        fixture_model(&cache, 1);
        let service = TeamyLlmService::new(app.clone(), cache);
        service.set_default_model(KNOWN_LLM_MODELS[0].name).unwrap();
        service.set_default_model(KNOWN_LLM_MODELS[1].name).unwrap();
        assert_eq!(
            service.default_model().unwrap().model_name,
            KNOWN_LLM_MODELS[1].name
        );
        let files = fs::read_dir(&app.0)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(
            files.len(),
            1,
            "staging files must not remain after success"
        );
    }

    #[test]
    fn explicit_selector_bypasses_default_and_does_not_mutate_it() {
        let temp = tempfile::tempdir().unwrap();
        let (app, cache) = homes(temp.path());
        let old = fixture_model(&cache, 0);
        fixture_model(&cache, 1);
        let service = TeamyLlmService::new(app.clone(), cache);
        service.set_default_model(KNOWN_LLM_MODELS[1].name).unwrap();
        let original = fs::read(app.file_path(DEFAULT_MODEL_FILE)).unwrap();
        assert_eq!(
            service
                .resolve_model_dir(&ModelSelector::managed(KNOWN_LLM_MODELS[0].name))
                .unwrap(),
            old
        );
        let custom = temp.path().join("explicit-path");
        assert_eq!(
            service
                .resolve_model_dir(&ModelSelector::explicit_dir(custom.display().to_string()))
                .unwrap(),
            custom
        );
        assert_eq!(
            fs::read(app.file_path(DEFAULT_MODEL_FILE)).unwrap(),
            original
        );
    }

    #[test]
    fn invalid_records_do_not_fall_back_and_can_be_explicitly_replaced() {
        let temp = tempfile::tempdir().unwrap();
        let (app, cache) = homes(temp.path());
        let old = fixture_model(&cache, 0);
        add_registered_model_dir(&app, &old).unwrap();
        let service = TeamyLlmService::new(app.clone(), cache);
        for record in [
            "{",
            "{\"version\":2,\"model-name\":\"qwopus-3.5-9b-coder-q4-k-m\"}",
            "{\"version\":1,\"model-name\":\"unknown\"}",
            "{\"version\":1,\"model-name\":\"qwopus-3.5-9b-coder-q4-k-m\",\"model-dir\":\"other\"}",
        ] {
            fs::write(app.file_path(DEFAULT_MODEL_FILE), record).unwrap();
            assert_eq!(
                service.default_model().unwrap_err().kind,
                ServiceErrorKind::InvalidRequest
            );
            assert!(
                service
                    .resolve_model_dir(&ModelSelector::default())
                    .is_err()
            );
            assert_eq!(
                service
                    .resolve_model_dir(&ModelSelector::managed(KNOWN_LLM_MODELS[0].name))
                    .unwrap(),
                old
            );
        }
        service.set_default_model(KNOWN_LLM_MODELS[0].name).unwrap();
        assert_eq!(service.default_model().unwrap().origin, "explicit");
    }

    #[test]
    fn missing_selected_artifact_does_not_fall_back_to_registered_model() {
        let temp = tempfile::tempdir().unwrap();
        let (app, cache) = homes(temp.path());
        let old = fixture_model(&cache, 0);
        let new = fixture_model(&cache, 1);
        add_registered_model_dir(&app, &old).unwrap();
        let service = TeamyLlmService::new(app, cache);
        service.set_default_model(KNOWN_LLM_MODELS[1].name).unwrap();
        fs::remove_file(new.join(TOKENIZER_FILE_NAME)).unwrap();
        assert_eq!(
            service.default_model().unwrap_err().kind,
            ServiceErrorKind::ModelNotPrepared
        );
        assert_eq!(
            service
                .resolve_model_dir(&ModelSelector::default())
                .unwrap_err()
                .kind,
            ServiceErrorKind::ModelNotPrepared
        );
    }

    #[test]
    fn oversized_default_record_fails_without_consulting_registration() {
        let temp = tempfile::tempdir().unwrap();
        let (app, cache) = homes(temp.path());
        let old = fixture_model(&cache, 0);
        add_registered_model_dir(&app, &old).unwrap();
        fs::write(app.file_path(DEFAULT_MODEL_FILE), vec![b' '; 4097]).unwrap();
        let service = TeamyLlmService::new(app, cache);
        let error = service.default_model().unwrap_err();
        assert_eq!(error.kind, ServiceErrorKind::InvalidRequest);
        assert!(error.message.contains("4096-byte"));
        assert!(
            service
                .resolve_model_dir(&ModelSelector::default())
                .is_err()
        );
    }

    #[tokio::test]
    async fn preparing_another_existing_model_preserves_explicit_default() {
        let temp = tempfile::tempdir().unwrap();
        let (app, cache) = homes(temp.path());
        fixture_model(&cache, 0);
        fixture_model(&cache, 1);
        let service = TeamyLlmService::new(app.clone(), cache);
        let selected = service.set_default_model(KNOWN_LLM_MODELS[1].name).unwrap();
        let record = fs::read(app.file_path(DEFAULT_MODEL_FILE)).unwrap();
        service
            .prepare_model(
                PrepareModelRequest {
                    model: KNOWN_LLM_MODELS[0].name.into(),
                    overwrite: false,
                    include_mmproj: false,
                    export_burn_text: false,
                    burn_text_dtype: None,
                    download_main_model_file: false,
                },
                |_| Ok(()),
            )
            .await
            .unwrap();
        assert_eq!(service.default_model().unwrap(), selected);
        assert_eq!(fs::read(app.file_path(DEFAULT_MODEL_FILE)).unwrap(), record);
    }

    #[test]
    fn default_uses_the_runtime_cache_home_instead_of_persisted_paths() {
        let temp = tempfile::tempdir().unwrap();
        let (app, cache) = homes(temp.path());
        fixture_model(&cache, 1);
        TeamyLlmService::new(app.clone(), cache)
            .set_default_model(KNOWN_LLM_MODELS[1].name)
            .unwrap();
        let other_cache = CacheHome(temp.path().join("other-cache"));
        let new = fixture_model(&other_cache, 1);
        let service = TeamyLlmService::new(app, other_cache);
        assert_eq!(
            service
                .resolve_model_dir(&ModelSelector::default())
                .unwrap(),
            new
        );
    }

    #[cfg(windows)]
    #[test]
    fn failed_atomic_replacement_preserves_previous_record() {
        use std::os::windows::fs::OpenOptionsExt;
        let temp = tempfile::tempdir().unwrap();
        let (app, cache) = homes(temp.path());
        fixture_model(&cache, 0);
        fixture_model(&cache, 1);
        let service = TeamyLlmService::new(app.clone(), cache);
        service.set_default_model(KNOWN_LLM_MODELS[0].name).unwrap();
        let original = fs::read(app.file_path(DEFAULT_MODEL_FILE)).unwrap();
        let locked = fs::OpenOptions::new()
            .read(true)
            .share_mode(0)
            .open(app.file_path(DEFAULT_MODEL_FILE))
            .unwrap();
        assert!(service.set_default_model(KNOWN_LLM_MODELS[1].name).is_err());
        drop(locked);
        assert_eq!(
            fs::read(app.file_path(DEFAULT_MODEL_FILE)).unwrap(),
            original
        );
        assert_eq!(
            service.default_model().unwrap().model_name,
            KNOWN_LLM_MODELS[0].name
        );
    }
}
