from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys
from typing import Any


DEFAULT_MODEL_ID = "Jackrong/Qwopus3.5-9B-Coder"
PACKED_TENSOR_FILE_NAME = "tensors.bin"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="teamy-llm-reference")
    parser.add_argument("--check-imports", action="store_true")
    parser.add_argument("--config-report", action="store_true")
    parser.add_argument("--prompt-report", action="store_true")
    parser.add_argument("--layer-report", action="store_true")
    parser.add_argument("--export-burn-text", action="store_true")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--system-prompt")
    parser.add_argument("--user-prompt")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir")
    parser.add_argument("--dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args(argv)

    try:
        if args.check_imports:
            print_json(check_imports(), args.indent)
            return 0
        if args.config_report:
            print_json(config_report(args), args.indent)
            return 0
        if args.prompt_report:
            if not args.user_prompt:
                raise ValueError("--prompt-report requires --user-prompt")
            print_json(prompt_report(args), args.indent)
            return 0
        if args.layer_report:
            if not args.user_prompt:
                raise ValueError("--layer-report requires --user-prompt")
            print_json(layer_report(args), args.indent)
            return 0
        if args.export_burn_text:
            if not args.output_dir:
                raise ValueError("--export-burn-text requires --output-dir")
            print_json(export_burn_text(args), args.indent)
            return 0
    except Exception as error:  # noqa: BLE001 - CLI should report any reference failure.
        print(json.dumps({"ok": False, "error": str(error)}, indent=2), file=sys.stderr)
        return 1

    parser.print_help(sys.stderr)
    return 2


def check_imports() -> dict[str, Any]:
    import tokenizers
    import torch
    import transformers

    return {
        "ok": True,
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "cuda_available": torch.cuda.is_available(),
    }


def prompt_report(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_source = args.tokenizer_path or args.model_id
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
        local_files_only=is_local_model_source(tokenizer_source),
    )
    rendered_prompt = render_qwen_single_turn_prompt(
        system_prompt=args.system_prompt,
        user_prompt=args.user_prompt,
    )
    encoded = tokenizer(rendered_prompt, return_tensors="pt")

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype="auto",
    ).to(device)
    encoded = {name: value.to(device) for name, value in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits[0, -1]
        top = torch.topk(logits, args.top_k)
        generated = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    input_ids = encoded["input_ids"][0].detach().cpu().tolist()
    top_token_ids = top.indices.detach().cpu().tolist()
    top_logits = [float(value) for value in top.values.detach().cpu().tolist()]
    top_token_text = [
        tokenizer.decode([int(token_id)], skip_special_tokens=False)
        for token_id in top_token_ids
    ]
    generated_suffix = generated[0][encoded["input_ids"].shape[-1] :]
    generated_text = tokenizer.decode(generated_suffix, skip_special_tokens=False)

    return {
        "ok": True,
        "model_id": args.model_id,
        "tokenizer_source": tokenizer_source,
        "device": str(device),
        "rendered_prompt": rendered_prompt,
        "input_token_count": len(input_ids),
        "input_token_ids": input_ids,
        "top_token_ids": top_token_ids,
        "top_token_text": top_token_text,
        "top_logits": top_logits,
        "generated_text": generated_text,
    }


def layer_report(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_source = args.tokenizer_path or args.model_id
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
        local_files_only=is_local_model_source(tokenizer_source),
    )
    rendered_prompt = render_qwen_single_turn_prompt(
        system_prompt=args.system_prompt,
        user_prompt=args.user_prompt,
    )
    encoded = tokenizer(rendered_prompt, return_tensors="pt")

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype="auto",
        local_files_only=is_local_model_source(args.model_id),
    ).to(device)
    encoded = {name: value.to(device) for name, value in encoded.items()}

    text_model = resolve_text_model(model)
    layer_outputs: list[list[float] | None] = [None] * len(text_model.layers)
    final_norm_last_hidden: list[float] | None = None
    hooks = []

    def capture_layer(index: int):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            layer_outputs[index] = tensor[0, -1].detach().float().cpu().tolist()

        return hook

    def capture_norm(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        nonlocal final_norm_last_hidden
        tensor = output[0] if isinstance(output, tuple) else output
        final_norm_last_hidden = tensor[0, -1].detach().float().cpu().tolist()

    for index, layer in enumerate(text_model.layers):
        hooks.append(layer.register_forward_hook(capture_layer(index)))
    hooks.append(text_model.norm.register_forward_hook(capture_norm))

    try:
        with torch.no_grad():
            outputs = model(**encoded)
            logits = outputs.logits[0, -1]
            top = torch.topk(logits, args.top_k)
    finally:
        for hook in hooks:
            hook.remove()

    if any(value is None for value in layer_outputs):
        raise RuntimeError("One or more layer hooks did not capture a hidden state")
    if final_norm_last_hidden is None:
        raise RuntimeError("Final norm hook did not capture a hidden state")

    input_ids = encoded["input_ids"][0].detach().cpu().tolist()
    top_token_ids = top.indices.detach().cpu().tolist()
    top_logits = [float(value) for value in top.values.detach().cpu().tolist()]
    top_token_text = [
        tokenizer.decode([int(token_id)], skip_special_tokens=False)
        for token_id in top_token_ids
    ]

    return {
        "ok": True,
        "model_id": args.model_id,
        "tokenizer_source": tokenizer_source,
        "device": str(device),
        "rendered_prompt": rendered_prompt,
        "input_token_count": len(input_ids),
        "input_token_ids": input_ids,
        "top_token_ids": top_token_ids,
        "top_token_text": top_token_text,
        "top_logits": top_logits,
        "layer_last_hidden_states": layer_outputs,
        "final_norm_last_hidden": final_norm_last_hidden,
    }


def config_report(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    text_config = getattr(config, "text_config", None)
    layer_types = list(getattr(text_config, "layer_types", []) or [])
    layer_histogram: dict[str, int] = {}
    for layer_type in layer_types:
        layer_histogram[layer_type] = layer_histogram.get(layer_type, 0) + 1

    return {
        "ok": True,
        "model_id": args.model_id,
        "config_class": type(config).__name__,
        "architectures": list(getattr(config, "architectures", []) or []),
        "model_name": getattr(config, "model_name", None),
        "model_type": getattr(config, "model_type", None),
        "text_model_type": getattr(text_config, "model_type", None) if text_config else None,
        "text_num_hidden_layers": getattr(text_config, "num_hidden_layers", None) if text_config else None,
        "text_hidden_size": getattr(text_config, "hidden_size", None) if text_config else None,
        "text_intermediate_size": getattr(text_config, "intermediate_size", None) if text_config else None,
        "text_num_attention_heads": getattr(text_config, "num_attention_heads", None) if text_config else None,
        "text_num_key_value_heads": getattr(text_config, "num_key_value_heads", None) if text_config else None,
        "text_head_dim": getattr(text_config, "head_dim", None) if text_config else None,
        "text_hidden_act": getattr(text_config, "hidden_act", None) if text_config else None,
        "text_partial_rotary_factor": getattr(text_config, "partial_rotary_factor", None) if text_config else None,
        "text_rope_theta": resolve_optional_rope_theta(text_config) if text_config else None,
        "text_full_attention_interval": getattr(text_config, "full_attention_interval", None) if text_config else None,
        "text_linear_num_key_heads": getattr(text_config, "linear_num_key_heads", None) if text_config else None,
        "text_linear_num_value_heads": getattr(text_config, "linear_num_value_heads", None) if text_config else None,
        "text_linear_key_head_dim": getattr(text_config, "linear_key_head_dim", None) if text_config else None,
        "text_linear_value_head_dim": getattr(text_config, "linear_value_head_dim", None) if text_config else None,
        "text_linear_conv_kernel_dim": getattr(text_config, "linear_conv_kernel_dim", None) if text_config else None,
        "text_layer_histogram": layer_histogram,
        "text_layer_types_preview": layer_types[:8],
    }


def export_burn_text(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM

    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "burn-text-manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise RuntimeError(
            f"Burn text manifest already exists at {manifest_path}; rerun with --overwrite to replace it"
        )
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    model.eval()
    config = getattr(model.config, "text_config", model.config)
    state_dict = model.state_dict()
    layer_types = list(getattr(config, "layer_types", []) or [])
    if "model.embed_tokens.weight" in state_dict:
        source_prefix = "model"
    elif "model.language_model.embed_tokens.weight" in state_dict:
        source_prefix = "model.language_model"
    else:
        raise RuntimeError(
            "state_dict did not include either `model.embed_tokens.weight` or "
            "`model.language_model.embed_tokens.weight`"
        )

    required_tensors: dict[str, str] = {
        "model.embed_tokens.weight": f"{source_prefix}.embed_tokens.weight",
        "model.norm.weight": f"{source_prefix}.norm.weight",
        "lm_head.weight": "lm_head.weight",
    }
    for layer_index, layer_type in enumerate(layer_types):
        canonical_prefix = f"model.layers.{layer_index}"
        source_layer_prefix = f"{source_prefix}.layers.{layer_index}"
        required_tensors.update(
            {
                f"{canonical_prefix}.input_layernorm.weight": f"{source_layer_prefix}.input_layernorm.weight",
                f"{canonical_prefix}.post_attention_layernorm.weight": f"{source_layer_prefix}.post_attention_layernorm.weight",
                f"{canonical_prefix}.mlp.gate_proj.weight": f"{source_layer_prefix}.mlp.gate_proj.weight",
                f"{canonical_prefix}.mlp.up_proj.weight": f"{source_layer_prefix}.mlp.up_proj.weight",
                f"{canonical_prefix}.mlp.down_proj.weight": f"{source_layer_prefix}.mlp.down_proj.weight",
            }
        )
        if layer_type == "full_attention":
            required_tensors.update(
                {
                    f"{canonical_prefix}.self_attn.q_proj.weight": f"{source_layer_prefix}.self_attn.q_proj.weight",
                    f"{canonical_prefix}.self_attn.k_proj.weight": f"{source_layer_prefix}.self_attn.k_proj.weight",
                    f"{canonical_prefix}.self_attn.v_proj.weight": f"{source_layer_prefix}.self_attn.v_proj.weight",
                    f"{canonical_prefix}.self_attn.o_proj.weight": f"{source_layer_prefix}.self_attn.o_proj.weight",
                    f"{canonical_prefix}.self_attn.q_norm.weight": f"{source_layer_prefix}.self_attn.q_norm.weight",
                    f"{canonical_prefix}.self_attn.k_norm.weight": f"{source_layer_prefix}.self_attn.k_norm.weight",
                }
            )
        elif layer_type == "linear_attention":
            required_tensors.update(
                {
                    f"{canonical_prefix}.linear_attn.conv1d.weight": f"{source_layer_prefix}.linear_attn.conv1d.weight",
                    f"{canonical_prefix}.linear_attn.dt_bias": f"{source_layer_prefix}.linear_attn.dt_bias",
                    f"{canonical_prefix}.linear_attn.A_log": f"{source_layer_prefix}.linear_attn.A_log",
                    f"{canonical_prefix}.linear_attn.norm.weight": f"{source_layer_prefix}.linear_attn.norm.weight",
                    f"{canonical_prefix}.linear_attn.out_proj.weight": f"{source_layer_prefix}.linear_attn.out_proj.weight",
                    f"{canonical_prefix}.linear_attn.in_proj_qkv.weight": f"{source_layer_prefix}.linear_attn.in_proj_qkv.weight",
                    f"{canonical_prefix}.linear_attn.in_proj_z.weight": f"{source_layer_prefix}.linear_attn.in_proj_z.weight",
                    f"{canonical_prefix}.linear_attn.in_proj_b.weight": f"{source_layer_prefix}.linear_attn.in_proj_b.weight",
                    f"{canonical_prefix}.linear_attn.in_proj_a.weight": f"{source_layer_prefix}.linear_attn.in_proj_a.weight",
                }
            )
        else:
            raise RuntimeError(f"unsupported Qwen3.5 layer type {layer_type!r}")

    if args.dtype == "float16":
        target_dtype = torch.float16
    elif args.dtype == "bfloat16":
        target_dtype = torch.bfloat16
    else:
        target_dtype = torch.float32

    manifest = {
        "format_version": 1,
        "architecture": require_string_config_value(config, "model_type"),
        "source_model_id": args.model_id,
        "text_model_type": require_string_config_value(config, "model_type"),
        "vocab_size": int(config.vocab_size),
        "hidden_size": int(config.hidden_size),
        "intermediate_size": int(config.intermediate_size),
        "num_hidden_layers": int(config.num_hidden_layers),
        "num_attention_heads": int(config.num_attention_heads),
        "num_key_value_heads": int(config.num_key_value_heads),
        "head_dim": int(config.head_dim),
        "rms_norm_eps": float(config.rms_norm_eps),
        "hidden_act": require_string_config_value(config, "hidden_act"),
        "partial_rotary_factor": require_float_config_value(config, "partial_rotary_factor"),
        "rope_theta": resolve_required_rope_theta(config),
        "linear_num_key_heads": int(config.linear_num_key_heads),
        "linear_num_value_heads": int(config.linear_num_value_heads),
        "linear_key_head_dim": int(config.linear_key_head_dim),
        "linear_value_head_dim": int(config.linear_value_head_dim),
        "linear_conv_kernel_dim": int(config.linear_conv_kernel_dim),
        "layer_types": layer_types,
        "tensors": {},
    }

    packed_tensor_path = output_dir / PACKED_TENSOR_FILE_NAME
    with packed_tensor_path.open("wb") as packed_tensor_file:
        for canonical_name, source_name in required_tensors.items():
            tensor = state_dict.get(source_name)
            if tensor is None:
                raise RuntimeError(
                    f"state_dict was missing required tensor {source_name!r} for canonical key {canonical_name!r}"
                )
            if not tensor.is_floating_point():
                raise RuntimeError(f"state_dict tensor {source_name!r} was not floating point")
            tensor = tensor.detach().cpu().to(target_dtype).contiguous()
            tensor_bytes = tensor.view(torch.uint8).numpy().tobytes()
            offset_bytes = packed_tensor_file.tell()
            packed_tensor_file.write(tensor_bytes)
            manifest["tensors"][canonical_name] = {
                "path": PACKED_TENSOR_FILE_NAME,
                "shape": [int(dim) for dim in tensor.shape],
                "dtype": args.dtype,
                "offset_bytes": int(offset_bytes),
                "byte_len": len(tensor_bytes),
            }

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "ok": True,
        "model_id": args.model_id,
        "output_dir": output_dir.as_posix(),
        "dtype": args.dtype,
        "tensor_count": len(manifest["tensors"]),
    }


def require_float_config_value(config: Any, name: str) -> float:
    value = getattr(config, name, None)
    if value is None:
        raise RuntimeError(
            f"model config did not define required numeric field {name!r} for Burn text export"
        )
    return float(value)


def require_string_config_value(config: Any, name: str) -> str:
    value = getattr(config, name, None)
    if value is None:
        raise RuntimeError(
            f"model config did not define required string field {name!r} for Burn text export"
        )
    return str(value)


def resolve_required_rope_theta(config: Any) -> float:
    rope_parameters = getattr(config, "rope_parameters", None)
    if rope_parameters is not None:
        rope_theta = rope_parameters.get("rope_theta")
        if rope_theta is not None:
            return float(rope_theta)
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is None:
        raise RuntimeError(
            "model config did not define rope_theta in text_config.rope_parameters or text_config.rope_theta"
        )
    return float(rope_theta)


def resolve_optional_rope_theta(config: Any) -> float | None:
    rope_parameters = getattr(config, "rope_parameters", None)
    if rope_parameters is not None:
        rope_theta = rope_parameters.get("rope_theta")
        if rope_theta is not None:
            return float(rope_theta)
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is None:
        return None
    return float(rope_theta)


def render_qwen_single_turn_prompt(
    system_prompt: str | None,
    user_prompt: str,
) -> str:
    parts: list[str] = []
    if system_prompt and system_prompt.strip():
        parts.append(f"<|im_start|>system\n{system_prompt.strip()}<|im_end|>\n")
    parts.append(f"<|im_start|>user\n{user_prompt.strip()}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def is_local_model_source(value: str) -> bool:
    return Path(value).exists()


def resolve_text_model(model: Any) -> Any:
    direct = getattr(model, "model", None)
    if direct is not None and hasattr(direct, "layers") and hasattr(direct, "norm"):
        return direct
    nested = getattr(direct, "language_model", None) if direct is not None else None
    if nested is not None and hasattr(nested, "layers") and hasattr(nested, "norm"):
        return nested
    raise RuntimeError("Could not resolve the Qwen3.5 text model for layer hooks")


def print_json(value: Any, indent: int) -> None:
    print(json.dumps(value, indent=indent, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
