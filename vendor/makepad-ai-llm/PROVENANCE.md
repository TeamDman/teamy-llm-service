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

Model interpretation, graph construction, numerical operations, CUDA/Metal
execution and existing session behavior are unchanged. CUDA kernels remain
in the pinned upstream CUDA crate. This hook makes no startup-time or
numerical-parity claim before qualification.

Cargo dependency activation and validation are managed by the service
workspace. No upstream checkout is modified by this source copy.
