//! A recorded native Julia location, independent of the text-generation default.
use std::fs::File;
use std::io::{ErrorKind, Read};
use std::path::{Path, PathBuf};

use eyre::{Result, WrapErr, ensure, eyre};
use facet::Facet;
use teamy_llm_burn_qwen35::paths::AppHome;

use super::DecisionProvider;

const RECORD_NAME: &str = "julia-native-model.json";
const MAX_RECORD_BYTES: u64 = 128 * 1024;

#[derive(Facet)]
#[facet(rename_all = "kebab-case", deny_unknown_fields)]
struct NativeJuliaLocation {
    version: u32,
    model_dir: String,
}

pub(super) fn resolve(explicit: Option<&Path>, provider: DecisionProvider) -> Result<PathBuf> {
    if let Some(path) = explicit {
        return Ok(path.to_path_buf());
    }
    ensure!(
        provider == DecisionProvider::Native,
        "ONNX requires an explicit --model-dir; the recorded Julia location is native-only"
    );
    let app_home = AppHome::resolve().wrap_err("cannot resolve Julia configuration directory")?;
    read_record(&app_home.file_path(RECORD_NAME))
}

fn read_record(path: &Path) -> Result<PathBuf> {
    let file = File::open(path).map_err(|error| {
        if error.kind() == ErrorKind::NotFound {
            eyre!("Julia's native model location is not configured; supply --model-dir")
        } else {
            eyre!(error).wrap_err("cannot read recorded Julia model location")
        }
    })?;
    let mut bytes = Vec::new();
    file.take(MAX_RECORD_BYTES + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= MAX_RECORD_BYTES,
        "Julia location record exceeds 128 KiB"
    );
    let text = std::str::from_utf8(&bytes).wrap_err("Julia location record must be UTF-8")?;
    let record: NativeJuliaLocation =
        facet_json::from_str(text).wrap_err("invalid Julia location record")?;
    ensure!(
        record.version == 1,
        "unsupported Julia location record version; expected 1"
    );
    ensure!(
        !record.model_dir.is_empty() && !record.model_dir.contains('\0'),
        "recorded Julia model directory must be nonempty and contain no NUL"
    );
    let model_dir = PathBuf::from(record.model_dir);
    ensure!(
        model_dir.is_absolute(),
        "recorded Julia model directory must be absolute"
    );
    Ok(model_dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_record(root: &Path, text: &str) -> PathBuf {
        let record = root.join(RECORD_NAME);
        std::fs::write(&record, text).unwrap();
        record
    }

    #[test]
    fn explicit_locations_do_not_require_configuration_or_provider_defaults() {
        let explicit = Path::new("fixtures/Julia café 日本語");
        for provider in [DecisionProvider::Native, DecisionProvider::Onnx] {
            assert_eq!(resolve(Some(explicit), provider).unwrap(), explicit);
        }
        assert!(
            resolve(None, DecisionProvider::Onnx)
                .unwrap_err()
                .to_string()
                .contains("explicit --model-dir")
        );
    }

    #[test]
    fn recorded_absolute_location_preserves_unicode_without_artifact_reads() {
        let root = tempfile::tempdir().unwrap();
        let model_dir = root.path().join("nonexistent Julia café 日本語");
        let json = facet_json::to_string(&NativeJuliaLocation {
            version: 1,
            model_dir: model_dir.to_str().unwrap().into(),
        })
        .unwrap();
        let record = write_record(root.path(), &json);
        assert_eq!(read_record(&record).unwrap(), model_dir);
        assert!(
            !model_dir.exists(),
            "resolving configuration must not read or create artifacts"
        );
    }

    #[test]
    fn invalid_records_fail_instead_of_searching_or_using_generation_defaults() {
        let root = tempfile::tempdir().unwrap();
        let record = root.path().join(RECORD_NAME);
        assert!(
            read_record(&record)
                .unwrap_err()
                .to_string()
                .contains("--model-dir")
        );
        for text in [
            r#"{"version":2,"model-dir":"/absolute"}"#,
            r#"{"version":1,"model-dir":"relative"}"#,
            r#"{"version":1,"model-dir":""}"#,
            r#"{"version":1,"model-dir":"\u0000"}"#,
            r#"{"version":1,"model-dir":"/absolute","typo":true}"#,
        ] {
            std::fs::write(&record, text).unwrap();
            assert!(read_record(&record).is_err(), "{text}");
        }
        std::fs::write(&record, [0xff]).unwrap();
        assert!(read_record(&record).is_err());
        std::fs::write(&record, vec![b' '; MAX_RECORD_BYTES as usize + 1]).unwrap();
        assert!(
            read_record(&record)
                .unwrap_err()
                .to_string()
                .contains("128 KiB")
        );
    }
}
