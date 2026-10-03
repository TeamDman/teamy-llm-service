# Vendored CubeCL CUDA cache recovery

Source: `cubecl-cuda` 0.10.0, copied from the already resolved crates.io package.

- Upstream repository: https://github.com/tracel-ai/cubecl
- Upstream commit: `7cf203735e095e640a2c03b2400d0faa03196bb4`
- Upstream package directory: `crates/cubecl-cuda`
- Crates.io package checksum: `b6b0a69ff45688d322ad8e92c8bf645167b9ca490fa8fa087fc6adac8c5e46be`
- Original Apache-2.0 and MIT license files, normalized Cargo manifest and upstream package metadata are retained.

The patch treats a driver failure while loading a cached CUDA artifact as a cache miss and recompiles the kernel through the existing path. It rejects malformed PTX termination, structurally invalid ELF64 CUBIN images and invalid entry-point names before passing cache data to the driver. Eleven tests in the dependency-free `cache_artifact.rs` helper cover those checks, the separate artifact directories and checksum field binding; they can run through `rustc --test` without compiling generated CUDA tests or initializing a GPU.

The optional `teamy-cubin-cache` feature extracts native CUBIN bytes through the already resolved cudarc NVRTC interface. It uses the existing real-GPU-architecture compiler flag and stores a format-tagged cache record in a separate `cubin` directory. Without that feature, PTX remains the default. The existing CBOR `ptx` payload field name is retained for compatibility.

A versioned XXH3-128 checksum binds the expected kernel key, format, entry-point name, shared-memory size and payload bytes. This detects accidental bitrot; it does not authenticate a cache or protect against an adversary who can rewrite both data and checksum. Legacy entries with no checksum and entries with a mismatched checksum are misses. Invalid entries remain untouched, and fresh compilation can still proceed without persisting over the conflicting record.

This patch does not change numerical kernels, NVRTC compiler flags or automatic cache enablement. The application owns source/toolchain/device compatibility namespacing and cache write serialization. CUDA remains responsible for validating module semantics after the structural bounds check. Runtime and numerical qualification are recorded by the consuming application after its checks run.
