use cubecl_common::backtrace::BackTrace;
use cubecl_cpp::formatter::format_cpp;
use cubecl_cpp::{cuda::arch::CudaArchitecture, shared::CompilationOptions};
use cubecl_runtime::{
    compiler::CompilationError,
    validation::{validate_cube_dim, validate_units},
};

use super::{
    cache_artifact::{CacheArtifactFormat, validate_cache_entry, write_checksum_input},
    storage::gpu::GpuResource,
};
use crate::{CudaCompiler, compute::stream::Stream};
use crate::{
    CudaComputeKernel,
    install::{cccl_include_path, include_path},
};
use cubecl_core::{
    compilation_cache::CompilationCache,
    hash::StableHash,
    server::ResourceLimitError,
    {ir::DeviceProperties, prelude::*},
};
use cubecl_runtime::timestamp_profiler::TimestampProfiler;
use cubecl_runtime::{compiler::CubeTask, logging::ServerLogger};
use cudarc::driver::DriverError;
use cudarc::driver::sys::CUfunc_st;
use cudarc::driver::sys::{CUctx_st, CUfunction_attribute, CUtensorMap};
use std::collections::HashMap;
use std::ffi::CString;
use std::ffi::c_char;
use std::str::FromStr;
use std::sync::Arc;
use std::{ffi::CStr, os::raw::c_void};

use core::hash::Hasher;
use cubecl_common::cache::CacheOption;
use cubecl_common::hash::StableHasher;

#[derive(Debug)]
pub(crate) struct CudaContext {
    pub context: *mut CUctx_st,
    pub module_names: HashMap<KernelId, CompiledKernel>,
    ptx_cache: Option<CompilationCache<StableHash, PtxCacheEntry>>,
    pub timestamps: TimestampProfiler,
    pub arch: CudaArchitecture,
    pub compilation_options: CompilationOptions,
    pub properties: DeviceProperties,
}

fn malformed_artifact(reason: impl Into<String>) -> CompilationError {
    CompilationError::Generic {
        reason: reason.into(),
        backtrace: BackTrace::capture(),
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(remote = "CacheArtifactFormat", rename_all = "lowercase")]
enum CacheArtifactFormatSerde {
    Ptx,
    Cubin,
}

/// Extract the selected NVRTC artifact using the exact same compilation flags.
/// Native CUBIN is opt-in; the default path remains upstream's PTX output.
unsafe fn compiled_image(
    program: cudarc::nvrtc::sys::nvrtcProgram,
) -> Result<Vec<c_char>, CompilationError> {
    #[cfg(feature = "teamy-cubin-cache")]
    {
        let mut size = 0;
        // SAFETY: `program` is a successfully compiled, still-live NVRTC program.
        unsafe { cudarc::nvrtc::sys::nvrtcGetCUBINSize(program, &mut size) }
            .result()
            .map_err(|err| malformed_artifact(format!("Unable to obtain CUBIN size: {err:?}")))?;
        if size == 0 {
            return Err(malformed_artifact("NVRTC returned an empty CUBIN image"));
        }
        let mut image = vec![0; size];
        // SAFETY: The vector holds exactly the byte count returned by NVRTC.
        unsafe { cudarc::nvrtc::sys::nvrtcGetCUBIN(program, image.as_mut_ptr()) }
            .result()
            .map_err(|err| malformed_artifact(format!("Unable to obtain CUBIN image: {err:?}")))?;
        Ok(image)
    }
    #[cfg(not(feature = "teamy-cubin-cache"))]
    {
        // SAFETY: `program` is a successfully compiled, still-live NVRTC program.
        unsafe { cudarc::nvrtc::result::get_ptx(program) }
            .map_err(|err| malformed_artifact(format!("Unable to obtain PTX image: {err:?}")))
    }
}

#[derive(Debug)]
pub struct CompiledKernel {
    cube_dim: CubeDim,
    shared_mem_bytes: usize,
    func: *mut CUfunc_st,
}

#[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq, Clone)]
pub struct PtxCacheEntry {
    #[serde(default, with = "CacheArtifactFormatSerde")]
    format: CacheArtifactFormat,
    #[serde(default)]
    checksum: Option<[u8; 16]>,
    entrypoint_name: String,
    shared_mem_bytes: usize,
    // Preserve upstream's CBOR field name; `format` defines the module image type.
    ptx: Vec<std::ffi::c_char>,
}

impl PtxCacheEntry {
    /// XXH3-128 detects accidental cache bitrot, not adversarial modification.
    /// Bind the expected kernel key and every load-affecting field to the image.
    fn bitrot_checksum(&self, expected_kernel_hash: StableHash) -> [u8; 16] {
        let mut hasher = StableHasher::new();
        write_checksum_input(
            expected_kernel_hash,
            self.format,
            &self.entrypoint_name,
            self.shared_mem_bytes as u64,
            bytemuck::cast_slice(&self.ptx),
            |bytes| hasher.write(bytes),
        );
        hasher.finalize().to_le_bytes()
    }

    fn checksum_matches(&self, expected_kernel_hash: StableHash) -> bool {
        self.checksum == Some(self.bitrot_checksum(expected_kernel_hash))
    }
}

impl CudaContext {
    pub fn new(
        compilation_options: CompilationOptions,
        properties: DeviceProperties,
        context: *mut CUctx_st,
        arch: CudaArchitecture,
    ) -> Self {
        Self {
            context,
            module_names: HashMap::new(),
            ptx_cache: {
                use cubecl_runtime::config::RuntimeConfig;
                let config = cubecl_runtime::config::CubeClRuntimeConfig::get();
                if let Some(cache) = &config.compilation.cache {
                    let root = cache.root();
                    Some(CompilationCache::new(
                        CacheArtifactFormat::selected().directory(),
                        CacheOption::default().name("cuda").root(root),
                    ))
                } else {
                    None
                }
            },
            arch,
            timestamps: TimestampProfiler::default(),
            compilation_options,
            properties,
        }
    }

    /// Switches the current CUDA context to this context.
    pub fn unsafe_set_current(&self) -> Result<(), DriverError> {
        // SAFETY: `self.context` is a valid CUDA context obtained from `primary_ctx::retain`
        // during server initialization and remains valid for the server's lifetime.
        unsafe { cudarc::driver::result::ctx::set_current(self.context) }
    }

    pub fn compile_kernel(
        &mut self,
        kernel_id: &KernelId,
        kernel: Box<dyn CubeTask<CudaCompiler>>,
        mode: ExecutionMode,
        logger: Arc<ServerLogger>,
    ) -> Result<(), LaunchError> {
        let hash = if let Some(cache) = &self.ptx_cache {
            let hash = kernel_id.stable_hash();

            if let Some(entry) = cache.get(&hash) {
                if entry.format == CacheArtifactFormat::selected() && entry.checksum_matches(hash) {
                    log::trace!("Using CUDA {:?} cache", entry.format);

                    match self.load_ptx(
                        entry.ptx.clone(),
                        entry.format,
                        kernel_id.clone(),
                        entry.entrypoint_name.clone(),
                        kernel_id.cube_dim,
                        entry.shared_mem_bytes,
                    ) {
                        Ok(()) => return Ok(()),
                        Err(err) => log::warn!(
                            "Unable to load cached CUDA artifact; compiling the kernel freshly: {err:?}"
                        ),
                    }
                } else {
                    log::warn!(
                        "Ignoring CUDA cache entry with incompatible format or missing/mismatched bitrot checksum"
                    );
                }
            }
            Some(hash)
        } else {
            None
        };

        log::trace!("Compiling kernel");

        validate_cube_dim(&self.properties, kernel_id)?;
        validate_units(&self.properties, kernel_id)?;

        let mut kernel_compiled = kernel.compile(
            &mut Default::default(),
            &self.compilation_options,
            mode,
            kernel.address_type(),
        )?;

        self.validate_shared(&kernel_compiled.repr)?;

        if logger.compilation_activated() {
            kernel_compiled.debug_info = Some(DebugInformation::new("cpp", kernel_id.clone()));

            if let Ok(formatted) = format_cpp(&kernel_compiled.source) {
                kernel_compiled.source = formatted;
            }
        }

        let cube_dim = kernel_compiled.cube_dim;
        let arch = if self.arch.version >= 90 {
            format!("--gpu-architecture=sm_{}a", self.arch)
        } else {
            format!("--gpu-architecture=sm_{}", self.arch)
        };

        let include_path = include_path();
        let include_option = format!("--include-path={}", include_path.to_str().unwrap());
        let cccl_include_path = cccl_include_path();
        let cccl_include_option = format!("--include-path={}", cccl_include_path.to_str().unwrap());
        let mut options = vec![arch.as_str(), include_option.as_str(), "-lineinfo"];
        if cccl_include_path.exists() {
            options.push(&cccl_include_option);
        }

        logger.log_compilation(&kernel_compiled);

        // SAFETY: Calling NVRTC FFI to create, compile, and extract a module image.
        // The `CString` source is null-terminated and outlives the program. On compilation
        // failure, the error log is retrieved and reported before returning.
        let ptx = unsafe {
            // I'd like to set the name to the kernel name, but keep getting UTF-8 errors so let's
            // leave it `None` for now
            let source = CString::from_str(&kernel_compiled.source).unwrap();
            let program =
                cudarc::nvrtc::result::create_program(source.as_c_str(), None).map_err(|err| {
                    CompilationError::Generic {
                        reason: format!("{err:?}"),
                        backtrace: BackTrace::capture(),
                    }
                })?;
            if cudarc::nvrtc::result::compile_program(program, &options).is_err() {
                let log_raw = cudarc::nvrtc::result::get_program_log(program).map_err(|err| {
                    CompilationError::Generic {
                        reason: format!("{err:?}"),
                        backtrace: BackTrace::capture(),
                    }
                })?;

                let log_ptr = log_raw.as_ptr();
                let log = CStr::from_ptr(log_ptr).to_str().unwrap();
                let mut message = "[Compilation Error] ".to_string();
                for line in log.split('\n') {
                    if !line.is_empty() {
                        message += format!("\n    {line}").as_str();
                    }
                }
                let source = kernel
                    .compile(
                        &mut Default::default(),
                        &self.compilation_options,
                        mode,
                        kernel.address_type(),
                    )?
                    .source;
                Err(CompilationError::Generic {
                    reason: format!("{message}\n[Source]  \n{source}"),
                    backtrace: BackTrace::capture(),
                })?;
            };
            compiled_image(program)?
        };

        let repr = kernel_compiled.repr.unwrap();

        if let Some(cache) = &mut self.ptx_cache {
            let kernel_hash = hash.unwrap();
            let mut entry = PtxCacheEntry {
                format: CacheArtifactFormat::selected(),
                checksum: None,
                entrypoint_name: kernel_compiled.entrypoint_name.clone(),
                shared_mem_bytes: repr.shared_memory_size(),
                ptx: ptx.clone(),
            };
            entry.checksum = Some(entry.bitrot_checksum(kernel_hash));
            let result = cache.insert(kernel_hash, entry);
            if let Err(err) = result {
                // DuplicatedKey includes complete images; avoid dumping kernel bytes.
                match err {
                    cubecl_core::compilation_cache::CompilationCacheError::DuplicatedKey {
                        ..
                    } => {
                        log::warn!(
                            "Unable to save CUDA artifact: an incompatible cache entry already exists"
                        );
                    }
                    other => log::warn!("Unable to save the CUDA artifact {other:?}"),
                }
            }
        }

        self.load_ptx(
            ptx,
            CacheArtifactFormat::selected(),
            kernel_id.clone(),
            kernel_compiled.entrypoint_name,
            cube_dim,
            repr.shared_memory_size(),
        )?;
        Ok(())
    }

    fn load_ptx(
        &mut self,
        ptx: Vec<c_char>,
        format: CacheArtifactFormat,
        kernel_id: KernelId,
        entrypoint_name: String,
        cube_dim: CubeDim,
        shared_mem_bytes: usize,
    ) -> Result<(), CompilationError> {
        let func_name =
            validate_cache_entry(&ptx, format, &entrypoint_name).map_err(malformed_artifact)?;
        // SAFETY: The image is terminated PTX or structurally bounded ELF64 CUBIN.
        // The buffer remains alive during loading; the driver validates CUDA-specific
        // module semantics. `func_name` is a validated null-terminated `CString`.
        let func = unsafe {
            let module = cudarc::driver::result::module::load_data(ptx.as_ptr() as *const _)
                .map_err(|err| CompilationError::Generic {
                    reason: format!("Unable to load the CUDA {format:?} image: {err:?}"),
                    backtrace: BackTrace::capture(),
                })?;

            cudarc::driver::result::module::get_function(module, func_name).map_err(|err| {
                CompilationError::Generic {
                    reason: format!("Unable to fetch the function from the module: {err:?}"),
                    backtrace: BackTrace::capture(),
                }
            })?
        };

        self.module_names.insert(
            kernel_id.clone(),
            CompiledKernel {
                cube_dim,
                shared_mem_bytes,
                func,
            },
        );

        Ok(())
    }

    pub fn execute_task(
        &mut self,
        stream: &mut Stream,
        kernel_id: KernelId,
        dispatch_count: (u32, u32, u32),
        tensor_maps: &[CUtensorMap],
        resources: &[GpuResource],
        const_info: Option<*mut c_void>,
    ) -> Result<(), LaunchError> {
        let mut bindings = tensor_maps
            .iter()
            .map(|map| map as *const _ as *mut c_void)
            .collect::<Vec<_>>();
        bindings.extend(resources.iter().map(|memory| memory.binding));
        bindings.extend(const_info);

        let kernel = self.module_names.get(&kernel_id).unwrap();
        let cube_dim = kernel.cube_dim;
        // SAFETY: `kernel.func` is a valid function handle from a loaded module.
        // `stream.sys` is a valid CUDA stream. `bindings` contains valid device pointers
        // for all kernel arguments. The dispatch and cube dimensions are validated by
        // the caller.
        unsafe {
            cudarc::driver::result::function::set_function_attribute(
                kernel.func,
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                kernel.shared_mem_bytes as i32,
            )
            .map_err(|err| LaunchError::Unknown {
                reason: format!("{err:?}"),
                backtrace: BackTrace::capture(),
            })?;
            cudarc::driver::result::launch_kernel(
                kernel.func,
                dispatch_count,
                (cube_dim.x, cube_dim.y, cube_dim.z),
                // Shared memory is collected into a single buffer, with each shared memory being
                // an offset pointer
                kernel.shared_mem_bytes as u32,
                stream.sys,
                &mut bindings,
            )
            .map_err(|err| LaunchError::Unknown {
                reason: format!("{err:?}"),
                backtrace: BackTrace::capture(),
            })?;
        };

        Ok(())
    }

    fn validate_shared(&self, repr: &Option<CudaComputeKernel>) -> Result<(), LaunchError> {
        let requested = repr.as_ref().map(|repr| repr.shared_memory_size());
        let max = self.properties.hardware.max_shared_memory_size;
        if let Some(requested) = requested
            && requested > max
        {
            Err(ResourceLimitError::SharedMemory {
                requested,
                max,
                backtrace: BackTrace::capture(),
            }
            .into())
        } else {
            Ok(())
        }
    }
}
