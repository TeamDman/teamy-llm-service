# Native CUDA GGUF adapter

This crate keeps Q4_K_M weights in their GGUF encoding on the GPU. It uses
[Makepad's native LLM executor](https://github.com/makepad/makepad/tree/9e5e3d2b03214f8c2c37adffa0e308c815662060/libs/ai/llm)
at one immutable revision. It does not convert the model into a Burn FP16 bundle.

A dedicated thread owns the CUDA session. The public `GgufRuntime` facade is
`Send + Sync`; the session and its pointers stay on the owner thread. Jobs and
token events use bounded channels. The caller's event callback runs on the
caller thread. Split UTF-8 bytes are decoded with Makepad's incremental decoder.

The current slice accepts dense `qwen35`, GGUF file type 15 (`Q4_K_M`), text
prompts, and one generation at a time. It disables MTP. Mixed auxiliary F32/F16
tensors remain in their declared formats. Vision, projector loading and tool
execution are not implemented in this adapter.

## Build

Use an existing Windows CUDA toolkit and MSVC installation. These assignments
affect the current PowerShell process; they do not install or change a toolkit:

```powershell
$env:MAKEPAD_CUDA_ROOT = $env:CUDA_PATH
$env:MAKEPAD_GGML_CUDA_ARCH = '89'
$env:MAKEPAD_GGML_REQUIRE_CUDA = '1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
cargo check -p teamy_llm_makepad_gguf --locked -j2
```

`89` targets an RTX 4090. Makepad's CUDA build must succeed; the required-CUDA
setting rejects a build that would otherwise produce an unavailable-backend
stub. Device ordinal defaults to zero and can be set at process startup with
`MAKEPAD_LLAMA_CUDA_DEVICE`. `GgufLoadOptions.device_id` must match that setting.
The application initializes the cuBLAS setting before starting worker threads.

## Use

`GgufLoadOptions::orca_qwen_q4(path)` supplies the frozen OrcaRouter model hash.
The published filename is `Qwen3.8-27B-Uncensored-Q4_K_M.gguf`, revision
`fc437a3374c9977bdc339a8ec106dcd9a0357001`, 16,810,714,496 bytes, SHA-256
`3445102e9cde5d562508642c100a2f5ac3368a5a3f748442811d7a95daee3bec`.
Acquisition and licence acceptance happen outside this crate.

Pass a fully rendered prompt to `generate_with_events`; the adapter does not
guess a chat template. `GgufGenerationOptions::validate(prompt)` checks basic
bounds before loading artifacts. The caller must also tokenize independently
and check that prompt plus completion fits the chosen context.
Its report includes generated IDs/text and prefill/decode
wall times, plus the prompt token IDs. Supplying `expected_prompt_token_ids`
rejects a mismatch with an independent tokenizer before GPU prefill.
Its audit includes model hash, dimensions, actual device name and
the source revision. Event backpressure contributes to those wall times.
Callbacks must return promptly. One token event is buffered to preserve the
final decoder tail; callback latency is not a GPU first-token compute metric.

Cancellation is checked between hash blocks and before and after loading.
Generation cancellation and timeout are checked between prefill batches and
between decode tokens. A running GPU
kernel cannot be preempted. Dropping the last facade waits for its owner to
finish and release the CUDA session. Typed cancellation and timeout errors are
preserved across the owner-thread response channel.

Load diagnostics separate `gguf_verify_identity`, `gguf_load_model` and
`gguf_prepare_session`. These info spans contain byte/dimension counts and
exclude prompt text and paths.

## Validation boundary

Unit tests check argument bounds, cooperative cancellation/timeout, the facade's
thread traits and typed response transport. A successful build does not prove
that a particular model fits VRAM or produces correct output. Real model load,
tokenizer comparison, generation and phase timing are separate qualification
gates.
