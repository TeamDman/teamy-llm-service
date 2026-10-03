# Host launch diagnostics

This copy comes from Makepad revision `9e5e3d2b03214f8c2c37adffa0e308c815662060`, directory `libs/ai/cuda`.

`LICENSE` retains Makepad's original MIT copyright and permission notice. `NOTICE` is the unmodified upstream `libs/ai/NOTICE`, including the ggml/llama.cpp notices for these CUDA kernels; its paths and broader inventory refer to the original Makepad AI stack. The original `kernels/llm/fattn/LICENSE` is also retained unchanged.

The three MMVQ, MMQ and flash-attention host diagnostic sites previously wrote directly to stderr. They now pass bounded messages through `mkllm_set_launch_diagnostic_callback` to Rust tracing at DEBUG level, using target `teamy_llm_makepad_gguf::cuda::launch`. The application calls `install_tracing_callback()` before executor creation. No callback or no enabled subscriber means silence. Existing message limits and the MMVQ opt-in remain unchanged.

The C formatter owns a 1,024-byte stack buffer. It supplies at most 1,023 bytes, including when formatting truncates. The Rust callback borrows those bytes only for the call, requires valid UTF-8 and catches subscriber panics before returning through the C ABI. An exceptional panic payload is retained because a caller-defined destructor could itself panic outside the guard. The callback function has process lifetime, carries no user pointer and is stored atomically. Stub builds reference no new native symbol.

CUDA arithmetic, quantization, kernel dispatch decisions, launch arguments, model and stream pointers are unchanged. Build-time Cargo output, other-model cuDNN diagnostics and device/fatal assertion messages retain their upstream behavior. Model-free tests cover tracing filtering, malformed input bounds, UTF-8 and unwind containment; actual CUDA linkage and inference require separate qualification.
