# Teamy LLM Service

Persistent local LLM service for Teamy tools.

V1 exposes Facet-serializable request/response/event models, extracts the Teamy-Studio Burn Qwen3.5 path, and keeps the Vox-facing service boundary isolated from backend internals.

## Workspace

- `teamy_llm_core`: shared service types.
- `teamy_llm_burn_qwen35`: Teamy-Studio Burn Qwen3.5 backend extraction and Python reference/export harness.
- `teamy_llm_service`: in-process service facade, model registry/cache, FIFO per-model generation, and optional Vox trait/API.
- `teamy_llm_cli`: local commands for health, model inspection/preparation, and prompt generation.

The workspace pins `facet`, `figue`, and `vox` to the same TeamDman/facet git revision used by Cloud-Terrastodon.
