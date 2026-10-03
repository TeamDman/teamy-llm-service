//! Julia's CUDA prerequisite checks through native APIs, without command subprocesses.
//!
//! Driver API versions describe supported CUDA versions, not NVIDIA release build numbers:
//! <https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__VERSION.html>.
//! NVRTC versions come from the selected compiler DLL's `nvrtcVersion` function:
//! <https://docs.nvidia.com/cuda/nvrtc/index.html#general-information-query>.

use std::{path::PathBuf, sync::Arc};

#[cfg(any(windows, test))]
use eyre::{bail, ensure};
#[cfg(any(windows, test))]
use teamy_llm_burn_qwen35::burn_text::{REQUIRED_CUDA_MAJOR_VERSION, REQUIRED_CUDA_MINOR_VERSION};

/// Facts about the loaded runtime, suitable for a kernel-cache compatibility namespace.
/// The process retains the exact driver, NVRTC and builtins DLLs behind these facts.
#[derive(Debug)]
pub struct CudaRuntimeIdentity {
    /// CUDA Driver API encoding: `1000 * major + 10 * minor`.
    pub driver_api_version: i32,
    pub nvrtc_major: i32,
    pub nvrtc_minor: i32,
    /// Compute capability pairs, in CUDA's actual device ordinal order.
    pub devices_sm: Vec<(i32, i32)>,
    pub toolkit_root: PathBuf,
    pub nvrtc_path: PathBuf,
    pub nvrtc_builtins_path: PathBuf,
    pub include_path: PathBuf,
    #[cfg(windows)]
    _libraries: windows_runtime::RuntimeLibraries,
}

/// Verify the selected native CUDA runtime once per process and retain its DLL handles.
/// CUDA selection never silently falls back to CPU. No context, kernels or model are created here.
/// Changing `CUDA_PATH` after initialization is rejected rather than mixing runtime identities.
pub fn verify_native_cuda_runtime() -> eyre::Result<Arc<CudaRuntimeIdentity>> {
    #[cfg(windows)]
    {
        windows_runtime::verify()
    }
    #[cfg(not(windows))]
    {
        eyre::bail!(
            "Native Julia CUDA runtime verification currently requires the Windows CUDA toolkit layout; select the explicit CPU reference device or implement the platform's native runtime verification"
        )
    }
}

#[cfg(any(windows, test))]
fn validate_driver_version(raw: i32) -> eyre::Result<()> {
    ensure!(
        raw >= 1000 && raw % 10 == 0,
        "CUDA driver returned an invalid Driver API version: {raw}"
    );
    let major = raw / 1000;
    let minor = (raw % 1000) / 10;
    validate_cuda_version(major, minor, "CUDA driver support")
}

#[cfg(any(windows, test))]
fn validate_cuda_version(major: i32, minor: i32, source: &str) -> eyre::Result<()> {
    ensure!(
        major > 0 && (0..100).contains(&minor),
        "{source} returned an invalid CUDA version: {major}.{minor}"
    );
    let required = (
        REQUIRED_CUDA_MAJOR_VERSION as i32,
        REQUIRED_CUDA_MINOR_VERSION as i32,
    );
    ensure!(
        (major, minor) >= required,
        "{source} requires CUDA {}.{} or newer, but found {major}.{minor}",
        required.0,
        required.1
    );
    Ok(())
}

#[cfg(any(windows, test))]
fn validate_device_count(count: i32) -> eyre::Result<usize> {
    ensure!(
        (1..=256).contains(&count),
        "Native Julia CUDA requires 1–256 visible CUDA devices, but the driver reported {count}"
    );
    Ok(count as usize)
}

#[cfg(any(windows, test))]
fn validate_compute_capability(major: i32, minor: i32, ordinal: i32) -> eyre::Result<()> {
    ensure!(
        (1..100).contains(&major) && (0..100).contains(&minor),
        "CUDA device {ordinal} returned an invalid compute capability: {major}.{minor}"
    );
    Ok(())
}

#[cfg(any(windows, test))]
fn api_success(status: i32, operation: &str) -> eyre::Result<()> {
    if status != 0 {
        bail!("Native Julia CUDA prerequisite check {operation} failed with status {status}");
    }
    Ok(())
}

#[cfg(any(windows, test))]
fn validate_compiler_identity(verified: [usize; 2], actual: [usize; 2]) -> eyre::Result<()> {
    ensure!(
        verified
            .iter()
            .chain(actual.iter())
            .all(|address| *address != 0),
        "NVRTC compiler identity contains a null function address"
    );
    ensure!(
        verified == actual,
        "cudarc selected a different NVRTC compiler than the verified CUDA_PATH DLL; restart with a consistent runtime library configuration"
    );
    Ok(())
}

#[cfg(windows)]
mod windows_runtime {
    use std::{os::windows::ffi::OsStrExt, path::Path, sync::Mutex};

    use eyre::WrapErr;
    use libloading::os::windows::{
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LOAD_LIBRARY_SEARCH_SYSTEM32, Library,
    };
    use teamy_llm_burn_qwen35::burn_text::{
        REQUIRED_CUDA_NVRTC_BUILTINS_DLL_NAME, REQUIRED_CUDA_NVRTC_DLL_NAME,
    };

    use super::*;

    // Keep DLLs alive after the caller drops its Arc. CubeCL subsequently opens these same DLLs.
    // Serialize first initialization; a failed check leaves the cache empty and permits retry.
    static VERIFIED_RUNTIME: Mutex<Option<Arc<CudaRuntimeIdentity>>> = Mutex::new(None);

    #[derive(Debug)]
    pub(super) struct RuntimeLibraries {
        _driver: Library,
        _nvrtc: Library,
        _builtins: Library,
    }

    #[derive(Debug)]
    struct ToolkitPaths {
        root: PathBuf,
        nvrtc: PathBuf,
        builtins: PathBuf,
        include: PathBuf,
    }

    impl ToolkitPaths {
        fn from_environment() -> eyre::Result<Self> {
            let configured = std::env::var_os("CUDA_PATH").ok_or_else(|| {
                eyre::eyre!(
                    "CUDA {}.{} is required; CUDA_PATH is not set",
                    REQUIRED_CUDA_MAJOR_VERSION,
                    REQUIRED_CUDA_MINOR_VERSION
                )
            })?;
            let root = std::fs::canonicalize(PathBuf::from(configured))
                .wrap_err("Cannot resolve the configured CUDA_PATH toolkit directory")?;
            let runtime = root.join("bin").join("x64");
            let nvrtc = canonical_file(&runtime.join(REQUIRED_CUDA_NVRTC_DLL_NAME), "NVRTC")?;
            let builtins = canonical_file(
                &runtime.join(REQUIRED_CUDA_NVRTC_BUILTINS_DLL_NAME),
                "NVRTC builtins",
            )?;
            let include = std::fs::canonicalize(root.join("include"))
                .wrap_err("Cannot resolve CUDA_PATH/include for native Julia kernel compilation")?;
            ensure!(
                include.is_dir(),
                "CUDA toolkit include path is not a directory"
            );
            // Validate the declared include root before it participates in the cache identity.
            canonical_file(&include.join("cuda.h"), "CUDA toolkit header cuda.h")?;
            Ok(Self {
                root,
                nvrtc,
                builtins,
                include,
            })
        }
    }

    fn canonical_file(path: &Path, purpose: &str) -> eyre::Result<PathBuf> {
        ensure!(
            path.is_file(),
            "Required CUDA {purpose} file is missing at {}",
            path.display()
        );
        std::fs::canonicalize(path)
            .wrap_err_with(|| format!("Cannot resolve CUDA {purpose} at {}", path.display()))
    }

    pub(super) fn verify() -> eyre::Result<Arc<CudaRuntimeIdentity>> {
        let paths = ToolkitPaths::from_environment()?;
        let mut cached = VERIFIED_RUNTIME
            .lock()
            .map_err(|_| eyre::eyre!("Native Julia CUDA verification lock was poisoned"))?;
        if let Some(identity) = cached.as_ref() {
            ensure!(
                identity.toolkit_root == paths.root
                    && identity.nvrtc_path == paths.nvrtc
                    && identity.nvrtc_builtins_path == paths.builtins
                    && identity.include_path == paths.include,
                "CUDA_PATH changed after native Julia CUDA initialization; restart the process before selecting another toolkit"
            );
            return Ok(Arc::clone(identity));
        }

        let _span = tracing::debug_span!("julia_native_cuda_verify").entered();
        // SYSTEM32-only lookup prevents the working directory or CUDA_PATH from selecting a
        // replacement driver DLL. All toolkit DLLs use canonical absolute paths instead.
        let driver =
            unsafe { Library::load_with_flags("nvcuda.dll", LOAD_LIBRARY_SEARCH_SYSTEM32) }
                .wrap_err("Cannot load the system NVIDIA CUDA driver DLL nvcuda.dll")?;
        let builtins = load_toolkit_library(&paths.builtins)?;
        let nvrtc = load_toolkit_library(&paths.nvrtc)?;

        type DriverGetVersion = unsafe extern "system" fn(*mut i32) -> i32;
        type NvrtcVersion = unsafe extern "system" fn(*mut i32, *mut i32) -> i32;
        type Init = unsafe extern "system" fn(u32) -> i32;
        type DeviceGetCount = unsafe extern "system" fn(*mut i32) -> i32;
        type DeviceGet = unsafe extern "system" fn(*mut i32, i32) -> i32;
        type DeviceGetAttribute = unsafe extern "system" fn(*mut i32, i32, i32) -> i32;

        // ABI signatures and attribute values match NVIDIA's Driver API and the pinned cudarc
        // bindings. Each loaded symbol is called only while its owning Library remains alive.
        let mut driver_api_version = 0;
        let mut nvrtc_major = 0;
        let mut nvrtc_minor = 0;
        let mut count = 0;
        let devices_sm = unsafe {
            let get_driver_version = driver
                .get::<DriverGetVersion>(b"cuDriverGetVersion\0")
                .wrap_err("CUDA driver DLL does not export cuDriverGetVersion")?;
            api_success(
                get_driver_version(&mut driver_api_version),
                "cuDriverGetVersion",
            )?;
            validate_driver_version(driver_api_version)?;

            let nvrtc_version = nvrtc
                .get::<NvrtcVersion>(b"nvrtcVersion\0")
                .wrap_err("Selected NVRTC DLL does not export nvrtcVersion")?;
            api_success(
                nvrtc_version(&mut nvrtc_major, &mut nvrtc_minor),
                "nvrtcVersion",
            )?;
            validate_cuda_version(nvrtc_major, nvrtc_minor, "Selected NVRTC runtime")?;

            // Cudarc has its own process-wide generic-library lookup. Configure its search path
            // and prove that it selects the verified compiler before driver initialization or
            // publishing the kernel-cache identity; matching version numbers alone are weaker.
            configure_runtime_search_directory(&paths.nvrtc)?;
            verify_cudarc_compiler_matches(&nvrtc)?;

            let init = driver
                .get::<Init>(b"cuInit\0")
                .wrap_err("CUDA driver DLL does not export cuInit")?;
            api_success(init(0), "cuInit")?;
            let get_count = driver
                .get::<DeviceGetCount>(b"cuDeviceGetCount\0")
                .wrap_err("CUDA driver DLL does not export cuDeviceGetCount")?;
            api_success(get_count(&mut count), "cuDeviceGetCount")?;
            let bounded_count = validate_device_count(count)?;
            let get_device = driver
                .get::<DeviceGet>(b"cuDeviceGet\0")
                .wrap_err("CUDA driver DLL does not export cuDeviceGet")?;
            let get_attribute = driver
                .get::<DeviceGetAttribute>(b"cuDeviceGetAttribute\0")
                .wrap_err("CUDA driver DLL does not export cuDeviceGetAttribute")?;
            const COMPUTE_CAPABILITY_MAJOR: i32 = 75;
            const COMPUTE_CAPABILITY_MINOR: i32 = 76;
            let mut capabilities = Vec::with_capacity(bounded_count);
            for ordinal in 0..count {
                let mut device = 0;
                let mut major = 0;
                let mut minor = 0;
                api_success(get_device(&mut device, ordinal), "cuDeviceGet")?;
                api_success(
                    get_attribute(&mut major, COMPUTE_CAPABILITY_MAJOR, device),
                    "cuDeviceGetAttribute(COMPUTE_CAPABILITY_MAJOR)",
                )?;
                api_success(
                    get_attribute(&mut minor, COMPUTE_CAPABILITY_MINOR, device),
                    "cuDeviceGetAttribute(COMPUTE_CAPABILITY_MINOR)",
                )?;
                validate_compute_capability(major, minor, ordinal)?;
                capabilities.push((major, minor));
            }
            capabilities
        };

        tracing::debug!(
            driver_api_version,
            nvrtc_major,
            nvrtc_minor,
            devices_sm = ?devices_sm,
            "Native Julia CUDA prerequisites verified without subprocesses"
        );
        let identity = Arc::new(CudaRuntimeIdentity {
            driver_api_version,
            nvrtc_major,
            nvrtc_minor,
            devices_sm,
            toolkit_root: paths.root,
            nvrtc_path: paths.nvrtc,
            nvrtc_builtins_path: paths.builtins,
            include_path: paths.include,
            _libraries: RuntimeLibraries {
                _driver: driver,
                _nvrtc: nvrtc,
                _builtins: builtins,
            },
        });
        *cached = Some(Arc::clone(&identity));
        Ok(identity)
    }

    fn verify_cudarc_compiler_matches(verified: &Library) -> eyre::Result<()> {
        // These NVRTC C signatures match the pinned cudarc bindings. On the supported Windows
        // x64 toolkit, the C and system calling conventions use the same native ABI. Resolving
        // symbols does not compile a program or initialize a CUDA context.
        type NvrtcVersion = unsafe extern "C" fn(*mut i32, *mut i32) -> i32;
        type NvrtcCompileProgram = unsafe extern "C" fn(
            cudarc::nvrtc::sys::nvrtcProgram,
            i32,
            *const *const std::ffi::c_char,
        ) -> i32;
        let actual = std::panic::catch_unwind(|| unsafe { cudarc::nvrtc::sys::culib() })
            .map_err(|_| eyre::eyre!("cudarc could not initialize its native NVRTC library"))?;
        // SAFETY: these symbols use NVIDIA's declared function signatures and both libraries
        // stay alive while their addresses are read. Neither foreign function is called here.
        let (verified_addresses, actual_addresses) = unsafe {
            let verified_version = verified
                .get::<NvrtcVersion>(b"nvrtcVersion\0")
                .wrap_err("Verified NVRTC DLL does not export nvrtcVersion")?;
            let verified_compile = verified
                .get::<NvrtcCompileProgram>(b"nvrtcCompileProgram\0")
                .wrap_err("Verified NVRTC DLL does not export nvrtcCompileProgram")?;
            let actual_version = actual
                .get::<NvrtcVersion>(b"nvrtcVersion\0")
                .wrap_err("cudarc's NVRTC library does not export nvrtcVersion")?;
            let actual_compile = actual
                .get::<NvrtcCompileProgram>(b"nvrtcCompileProgram\0")
                .wrap_err("cudarc's NVRTC library does not export nvrtcCompileProgram")?;
            (
                [*verified_version as usize, *verified_compile as usize],
                [*actual_version as usize, *actual_compile as usize],
            )
        };
        validate_compiler_identity(verified_addresses, actual_addresses)
    }

    fn load_toolkit_library(path: &Path) -> eyre::Result<Library> {
        // The selected library's directory and System32 suffice for NVRTC/builtins dependencies;
        // don't consult the current working directory or arbitrary PATH entries.
        unsafe {
            Library::load_with_flags(
                path,
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32,
            )
        }
        .wrap_err_with(|| format!("Cannot load required CUDA runtime DLL {}", path.display()))
    }

    fn configure_runtime_search_directory(nvrtc_path: &Path) -> eyre::Result<()> {
        let directory = nvrtc_path.parent().ok_or_else(|| {
            eyre::eyre!("Verified NVRTC library does not have a parent runtime directory")
        })?;
        let mut wide: Vec<u16> = directory.as_os_str().encode_wide().collect();
        ensure!(
            !wide.contains(&0),
            "CUDA runtime directory contains an embedded NUL"
        );
        // canonicalize returns verbatim paths on Windows; SetDllDirectoryW accepts them.
        wide.push(0);
        // SAFETY: a terminated, initialized UTF-16 path remains alive for the entire call.
        if unsafe { SetDllDirectoryW(wide.as_ptr()) } == 0 {
            return Err(std::io::Error::last_os_error())
                .wrap_err("Cannot configure the verified CUDA runtime DLL search directory");
        }
        Ok(())
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn SetDllDirectoryW(path: *const u16) -> i32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn driver_api_encoding_and_required_boundary_are_checked() {
        for version in [13030, 13040, 14000] {
            validate_driver_version(version).expect("supported Driver API version");
        }
        for version in [12090, 13000, 13020] {
            assert!(validate_driver_version(version).is_err());
        }
        // A driver release such as 580 is not a CUDA Driver API version.
        for version in [-13030, 0, 580, 13031] {
            assert!(validate_driver_version(version).is_err());
        }
    }

    #[test]
    fn nvrtc_version_validation_rejects_unsupported_or_malformed_versions() {
        for (major, minor) in [(13, 3), (13, 4), (14, 0)] {
            validate_cuda_version(major, minor, "NVRTC").expect("supported NVRTC version");
        }
        for (major, minor) in [(12, 9), (13, 2), (0, 3), (13, -1), (13, 100)] {
            assert!(validate_cuda_version(major, minor, "NVRTC").is_err());
        }
    }

    #[test]
    fn zero_and_unbounded_device_counts_fail_before_allocation() {
        assert_eq!(validate_device_count(1).expect("one device"), 1);
        assert_eq!(
            validate_device_count(256).expect("bounded device count"),
            256
        );
        for count in [-1, 0, 257, i32::MAX] {
            assert!(validate_device_count(count).is_err());
        }
    }

    #[test]
    fn compute_capability_is_queried_data_not_a_fixed_4090_architecture() {
        for (major, minor) in [(8, 9), (9, 0), (12, 0)] {
            validate_compute_capability(major, minor, 0).expect("actual architecture");
        }
        for (major, minor) in [(0, 0), (-1, 0), (8, -1), (8, 100)] {
            assert!(validate_compute_capability(major, minor, 0).is_err());
        }
    }

    #[test]
    fn failed_native_api_statuses_cannot_be_treated_as_success() {
        api_success(0, "cuInit").expect("CUDA_SUCCESS");
        for status in [1, 100, 803, -1] {
            let error = api_success(status, "cuInit").expect_err("driver failure");
            assert!(error.to_string().contains("cuInit"));
            assert!(error.to_string().contains(&status.to_string()));
        }
    }

    #[test]
    fn cudarc_must_share_both_verified_compiler_entry_points() {
        validate_compiler_identity([11, 22], [11, 22]).expect("same retained compiler DLL");
        for actual in [[33, 22], [11, 44], [33, 44], [0, 22]] {
            assert!(validate_compiler_identity([11, 22], actual).is_err());
        }
        assert!(validate_compiler_identity([0, 0], [0, 0]).is_err());
    }
}
