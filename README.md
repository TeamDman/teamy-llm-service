# Teamy LLM Service

Persistent local LLM service for Teamy tools.

V1 exposes Facet-serializable request/response/event models, extracts the Teamy-Studio Burn Qwen3.5 path, and keeps the Vox-facing service boundary isolated from backend internals.

## Workspace

- `teamy_llm_core`: shared service types.
- `teamy_llm_burn_qwen35`: Teamy-Studio Burn Qwen3.5 backend extraction and Python reference/export harness.
- `teamy_llm_service`: in-process service facade, model registry/cache, FIFO per-model generation, and optional Vox trait/API.
- `teamy_llm_cli`: local commands for health, model inspection/preparation, and prompt generation.

The workspace pins `facet`, `figue`, and `vox` to the same TeamDman/facet git revision used by Cloud-Terrastodon.

## CLI cancellation

The local CLI accepts the same cancellation options as `teamy-rust-cli`:

```text
teamy-llm prompt --stop-after-duration 2m --max-new-tokens 200 "Why is the sky blue?"
teamy-llm prompt --stop-after "LLM model weights loaded" --max-new-tokens 200 "Why is the sky blue?"
```

`--stop-after-duration` accepts `humantime` durations such as `500ms`, `2s`, and `1m`.
`--stop-after` matches a tracing span or event message. Cancellation is cooperative: it is checked
between model-loading and generation operations, so it cannot preempt an already-running Burn/CUDA
kernel.

## CUDA runtime

CUDA inference needs the CUDA runtime compiler (`nvrtc`) to be discoverable through `PATH` at runtime.
Having an NVIDIA driver (and therefore `nvidia-smi`) is not sufficient by itself; install a compatible
CUDA Toolkit or otherwise provide its NVRTC runtime directory on `PATH`.
