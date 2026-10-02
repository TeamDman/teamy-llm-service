# Makepad LLM source

Source: <https://github.com/makepad/makepad/tree/9e5e3d2b03214f8c2c37adffa0e308c815662060/libs/ai/llm>

Revision: `9e5e3d2b03214f8c2c37adffa0e308c815662060`.

The source, build script and original tests were copied from this pinned
revision. The upstream root MIT license is retained in `LICENSE`, including
its copyright and permission notice.
Text uses LF line endings, matching the upstream Git blobs rather than the
Windows checkout's CRLF conversion.

Local changes:

- `makepad-ai-loader` resolves from the sibling vendored loader. All other
  Makepad dependencies remain on the same exact upstream Git revision,
  retaining their existing features and platform conditions.
- `LlamaSession::from_model_with_progress` exposes the existing session
  construction progress callback for an already parsed `LlamaModel`.
  It delegates directly to the original construction method; phase labels
  and fractional progress retain their existing upstream meaning.
- `LlamaSession::from_model_with_mapped_weights_and_progress` accepts an
  already verified, immutable `Arc<MappedRegion>`. The weight context retains
  the same live view during session construction, graph-reserve retries and
  later runtime-state rebuilding instead of reopening and remapping the file.
  `GgufWeightLayout::load_from_mapped_region` checks its length against the
  parsed GGUF and preserves the existing tensor bounds validation. An explicit
  mapping mismatch returns an error rather than reopening a different view.
  The caller must bind the mapping to the parsed metadata and keep its backing
  bytes immutable; reuse does not add namespace or privileged-writer protection.
- The Windows initial CUDA arena upload uses two local, bounded 64 MiB
  CUDA-pinned staging slots. CPU byte copies can overlap queued H2D transfers
  on the existing stream; a slot's completion event is waited before reuse.
  Completed progress, exact logical destination offsets and tensor bytes
  are preserved. The Unix upload path is unchanged. The owner drains pending
  DMA on success and errors before freeing staging; if both stream and device
  drains fail, it retains the bounded allocation until CUDA teardown.
  Allocation, CPU staging, enqueue, completion wait and completed H2D event
  timings are emitted as native diagnostics. These intervals overlap and
  are not independent additive costs or a cold-start performance claim.
  CPU staging timing and byte counts distinguish the read-only weights from
  the dirty prefix. The process-scoped experiment
  `MAKEPAD_LLAMA_CUDA_UPLOAD_WRITE_COMBINED=1` adds CUDA's write-combined flag
  (4) to these CPU-write/GPU-read staging allocations; absence or `0` keeps
  the original default flag (0). Other values fail rather than select a mode
  implicitly. Real transport never reads back staging contents on the CPU.
  This option makes no performance claim and does not change the default.
  The separate process-scoped `MAKEPAD_LLAMA_CUDA_UPLOAD_COPY_THREADS`
  experiment accepts exactly `1`, `2` or `4`; absence keeps one CPU copy
  worker. Chunks smaller than 1 MiB always copy sequentially. Larger chunks
  use disjoint borrowed source and destination slices, with every started
  worker joined before DMA enqueue or an error return. Invalid values and
  non-UTF-8 values fail. This experiment makes no performance claim and does
  not change the one-worker default.

Model interpretation, graph construction and numerical operations are
unchanged. CUDA kernels remain in the pinned upstream CUDA crate; Metal and
Unix upload execution retain their original paths. The Windows change affects
initial byte transport and its bounded staging ownership; the explicit mapped
session API also preserves the caller's existing mapping lifetime. These local
changes make no startup-time or numerical-parity claim before qualification.

Cargo dependency activation and validation are managed by the service
workspace. No upstream checkout is modified by this source copy.
