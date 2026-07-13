use directories_next::ProjectDirs;
use eyre::bail;
use std::env;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use tracing::warn;

pub const APP_HOME_ENV_VAR: &str = "TEAMY_LLM_SERVICE_HOME_DIR";
pub const APP_HOME_DIR_NAME: &str = "Teamy-LLM-Service";
pub const APP_CACHE_ENV_VAR: &str = "TEAMY_LLM_SERVICE_CACHE_DIR";
pub const APP_CACHE_DIR_NAME: &str = "Teamy-LLM-Service";

pub static APP_HOME: LazyLock<AppHome> = LazyLock::new(|| match AppHome::resolve() {
    Ok(home) => home,
    Err(error) => {
        warn!("failed to resolve Teamy LLM service app home: {error}");
        AppHome(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
    }
});

pub static CACHE_DIR: LazyLock<CacheHome> = LazyLock::new(|| match CacheHome::resolve() {
    Ok(home) => home,
    Err(error) => {
        warn!("failed to resolve Teamy LLM service cache home: {error}");
        CacheHome(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
    }
});

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AppHome(pub PathBuf);

impl AppHome {
    #[must_use]
    pub fn file_path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }

    pub fn ensure_dir(&self) -> eyre::Result<()> {
        std::fs::create_dir_all(&self.0)?;
        Ok(())
    }

    pub fn resolve() -> eyre::Result<Self> {
        if let Ok(override_dir) = env::var(APP_HOME_ENV_VAR) {
            return Ok(Self(PathBuf::from(override_dir)));
        }
        if let Some(project_dirs) = ProjectDirs::from("", "TeamDman", APP_HOME_DIR_NAME) {
            Ok(Self(project_dirs.config_dir().to_path_buf()))
        } else {
            bail!("Could not determine config directory")
        }
    }

    #[must_use]
    pub fn is_default(&self) -> bool {
        self.0 == APP_HOME.0
    }
}

impl Deref for AppHome {
    type Target = Path;

    fn deref(&self) -> &Self::Target {
        self.0.as_path()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CacheHome(pub PathBuf);

impl CacheHome {
    pub fn resolve() -> eyre::Result<Self> {
        if let Ok(override_dir) = env::var(APP_CACHE_ENV_VAR) {
            return Ok(Self(PathBuf::from(override_dir)));
        }
        if let Some(project_dirs) = ProjectDirs::from("", "TeamDman", APP_CACHE_DIR_NAME) {
            Ok(Self(project_dirs.cache_dir().to_path_buf()))
        } else {
            bail!("Could not determine cache directory")
        }
    }
}

impl Deref for CacheHome {
    type Target = Path;

    fn deref(&self) -> &Self::Target {
        self.0.as_path()
    }
}
