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

## Thinking modes

The CLI defaults to direct answers: it uses the Qwopus tokenizer's native empty-thinking-block control
instead of spending the output budget on a reasoning trace. Use `--thinking` when deliberate reasoning
is useful; its default token budget is 1024 rather than the direct mode's 256. `--max-new-tokens` always
overrides those defaults.

```text
teamy-llm prompt --no-thinking --max-new-tokens 128 "Why is the sky blue?"
teamy-llm prompt --thinking --max-new-tokens 1024 "Compare two approaches and justify the choice."
```

A long-lived `TeamyLlmService` also retains the default hybrid CUDA runtime per model after its
first request. This reuses lazily populated GPU weight caches for later requests; preparing a model
invalidates its cached runtime so updated weights are never reused.

## Performance baseline

The benchmark command defaults to the fixed prompt `Why is the sky blue?`, a 200-token cap, and
both native prompt modes. It prints one machine-readable row per mode:

```text
cargo run --release -p teamy_llm_cli -- benchmark --timeout-ms 600000
```

`time_to_first_token_ms` starts when the request is submitted, so it includes model loading and
prompt processing. `time_to_completion_ms` covers the entire request. `tokens_per_second` is the
post-first-token decode throughput (`generated token count / (completion - first token)`). Run each
mode in a new process (`--no-thinking` or `--thinking`) for a process-cold comparison; run both in
one invocation to observe the long-lived CUDA runtime cache's warm follow-up request.

## CUDA runtime

This is a GPU-only program: it does not fall back to CPU inference. The workspace pins Cudarc to the
CUDA 13.3 ABI and checks the following before inference:

- NVIDIA driver series 580 or newer, with `nvidia-smi` reporting CUDA Version 13.3 or newer.
- `CUDA_PATH` set to the CUDA 13.3 Toolkit installation.
- `%CUDA_PATH%\bin\x64\nvrtc64_130_0.dll` and `nvrtc-builtins64_133.dll` present.

Install the CUDA 13.3 Toolkit with its `nvrtc_13.3` component, open a fresh terminal, then rebuild.
The program adds `%CUDA_PATH%\bin\x64` to its own DLL search path before it loads Cudarc, matching
the CUDA 13.3.1 Windows layout.
Because Burn compiles CUDA code to PTX with NVRTC, a driver which merely meets CUDA 13.x minor-version
compatibility but reports an older CUDA version cannot run it. Update the NVIDIA driver before running
this program if it reports less than 13.3.

The unit tests check the CUDA contract. The ignored hardware test validates the configured Toolkit and
driver on a compatible machine:

```text
cargo test --workspace cuda_13
cargo test -p teamy_llm_burn_qwen35 installed_cuda_runtime_matches_contract_when_cuda_path_is_configured -- --ignored
```
