# Vendored CubeCL common cache support

Source: `cubecl-common` 0.10.0, copied from the already resolved crates.io package.

- Upstream repository: https://github.com/tracel-ai/cubecl
- Upstream commit: `7cf203735e095e640a2c03b2400d0faa03196bb4`
- Upstream package directory: `crates/cubecl-common`
- Crates.io package checksum: `cc956c2dcc993f16f748d03bfdbd900f75cab4d469f4e5d8924f8ee128e861ad`
- Original Apache-2.0 and MIT license files, normalized Cargo manifest and upstream package metadata are retained.

The patch makes optional compiled-kernel cache failures recoverable. Missing, unreadable or malformed CBOR chunks produce cache misses, and an invalid chunk is left untouched with persistence disabled. Failed appends retain the live in-memory entry and disable further writes. TOC read and serialization failures retain entries in memory with diagnostics, short writes are handled, and TOC paths cannot escape the cache root. Refreshes preserve the addresses of previously exposed cache values. The public cache construction and lookup signatures and the existing TOC error variants are unchanged.

The application remains responsible for a source/toolchain/device-specific cache namespace and exclusive ownership during writes. This patch does not change numerical kernels, compiler flags, automatic cache enablement or the existing PTX/CBOR format. Cache data remains disposable and does not authorize file cleanup.

The added tests cover a fresh-instance cache hit, malformed and unreadable chunks, a missing TOC target, path traversal, a failed append, reference lifetime stability and conflicting records. Qualification is recorded by the consuming application after its checks run.
