"""Independent pinned publisher Julia reference; no Rust output is used as an expected value."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

NATIVE_REVISION = "a85b127321d580d65176c89ced8273f305745d85"
ONNX_REVISION = "82a2fadf8fccfccdc5fd4e1009ba8f1a265eb7a8"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def prepare_case(case, configuration):
    allowed = {"id", "state", "question", "options"}
    if set(case) - allowed or not isinstance(case.get("id"), str) or not case["id"]:
        raise ValueError("Case has unknown fields or a missing ID")
    if not isinstance(case.get("state"), str):
        raise ValueError("Case state must be text")
    options = case.get("options")
    if not isinstance(options, list) or not 2 <= len(options) <= 20:
        raise ValueError("Each case must contain 2–20 options")
    ids = []
    for option in options:
        if set(option) != {"id", "text"}:
            raise ValueError("Each option requires exactly id/text")
        if not all(isinstance(option[key], str) and option[key].strip() for key in ("id", "text")):
            raise ValueError("Option ID/text must be nonempty")
        ids.append(option["id"])
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate option IDs")
    question = case.get("question", configuration["default_question"])
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be nonempty text")
    row = {"state": case["state"], "question": question, "type": "choice",
           "options": [option["text"] for option in options]}
    payload = {"state": row["state"], "question": question, "type": "choice", "options": options,
               "max_tokens": configuration["max_tokens"], "head_tokens": configuration["head_tokens"]}
    canonical = canonical_json(payload)
    return row, ids, canonical


def verify_artifacts(root, manifest):
    native = manifest["native"]
    if native["revision"] != NATIVE_REVISION or manifest["onnx"]["revision"] != ONNX_REVISION:
        raise ValueError("Artifact manifest revisions differ from the frozen reference")
    for artifact in native["files"]:
        relative = Path(artifact["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe artifact path")
        path = root / relative
        if sha256(path) != artifact["sha256"]:
            raise ValueError(f"Artifact checksum differs: {relative}")
    vendor = Path(__file__).parent / "vendor"
    for artifact in native["files"]:
        if artifact["path"].startswith("julia/"):
            if sha256(vendor / artifact["path"]) != artifact["sha256"]:
                raise ValueError("Vendored publisher source differs from the pinned artifact")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("cases.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument("--image-id", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Reference outputs are never overwritten")
    configuration = json.loads(args.cases.read_text(encoding="utf-8"))
    if configuration["version"] != 1 or not configuration["cases"]:
        raise ValueError("Unsupported/empty case configuration")
    cases = [prepare_case(case, configuration) for case in configuration["cases"]]
    if len({case["id"] for case in configuration["cases"]}) != len(cases):
        raise ValueError("Duplicate case IDs")
    manifest = json.loads(args.artifact_manifest.read_text(encoding="utf-8"))
    verify_artifacts(args.model_dir, manifest)
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("Reference requires explicit offline settings")

    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AutoTokenizer

    sys.path.insert(0, str(Path(__file__).parent / "vendor"))
    from julia.data import Collator, sequence, validate_row
    from julia.model import JuliaDecisionModel

    expected = {"torch": "2.11.0+cu128", "transformers": "5.0.0", "tokenizers": "0.22.2",
                "safetensors": "0.7.0", "numpy": "2.4.6", "huggingface-hub": "1.16.1"}
    versions = {name: importlib.metadata.version(name) for name in expected}
    if versions != expected:
        raise RuntimeError(f"Dependency tuple differs from freeze: {versions}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable; CPU fallback is disabled")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(42)
        torch.cuda.reset_peak_memory_stats(device)
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir / "tokenizer", trust_remote_code=False,
                                              local_files_only=True)
    model = JuliaDecisionModel.from_pretrained(args.model_dir, memory_map=True).to(device).eval()
    model.marker_only_head = False
    collate = Collator(tokenizer, configuration["max_tokens"], configuration["head_tokens"])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_started
    results = []
    for case, (row, option_ids, canonical) in zip(configuration["cases"], cases):
        validate_row(row, 1)
        encoded = sequence(tokenizer, row, configuration["max_tokens"], configuration["head_tokens"], strict=True)
        row["_encoded"] = encoded
        batch = collate([row], include_targets=False)
        serialized = {"version": 1, "raw_input_ids": encoded["ids"], "input_ids": batch["input_ids"][0].tolist(),
                      "attention_mask": batch["attention_mask"][0].tolist(),
                      "marker_positions": batch["marker_pos"][0].tolist(),
                      "marker_mask": batch["marker_mask"][0].tolist(), "qtype": int(batch["qtype"][0]),
                      "option_ids": option_ids}
        moved = {key: value.to(device) for key, value in batch.items()}
        intermediates = {}

        def capture(name):
            def hook(module, inputs, output):
                hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output
                markers = moved["marker_pos"][:, :, None].expand(-1, -1, hidden.shape[-1])
                values = hidden.gather(1, markers).detach().float().cpu()
                intermediates[name] = {"shape": list(values.shape), "dtype": "float32", "values": values[0].tolist()}
            return hook

        hooks = [model.encoder.register_forward_hook(capture("encoder_marker_hidden"))]
        hooks.extend(layer.register_forward_hook(capture(f"head_{index}_marker_hidden"))
                     for index, layer in enumerate(model.head.layers))
        inference_started = time.perf_counter()
        try:
            with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                output = model(**moved)[0].float()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds = time.perf_counter() - inference_started
        finally:
            for hook in hooks:
                hook.remove()
        if not torch.isfinite(output).all():
            raise FloatingPointError("Publisher reference produced nonfinite logits")
        logits = output.cpu().tolist()
        probabilities = torch.softmax(output.double(), dim=-1).cpu().tolist()
        if len(logits) != len(option_ids) or abs(sum(probabilities) - 1.0) > 1e-12:
            raise RuntimeError("Invalid reference result shape/probabilities")
        result = {"id": case["id"], "state": row["state"], "question": row["question"], "type": "choice",
                  "max_tokens": configuration["max_tokens"], "head_tokens": configuration["head_tokens"],
                  "options": case["options"], "canonical_input": canonical,
                  "input_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                  "encoded": serialized,
                  "prompt_token_ids": tokenizer("choice question: " + row["question"], add_special_tokens=False)["input_ids"],
                  "state_token_ids": tokenizer(row["state"], add_special_tokens=False)["input_ids"],
                  "option_token_ids": [tokenizer(" " + option, add_special_tokens=False)["input_ids"] for option in row["options"]],
                  "logits": logits, "probabilities": probabilities,
                  "selected_option_id": option_ids[int(output.argmax())],
                  "intermediates": intermediates, "inference_seconds": inference_seconds}
        results.append(result)
        print(f"Reference {case['id']}: {len(option_ids)} options, {len(encoded['ids'])} tokens", file=sys.stderr)

    environment = {"python": platform.python_version(), "platform": platform.platform(), "packages": versions,
                   "device": str(device), "torch_cuda": torch.version.cuda, "torch_cudnn": torch.backends.cudnn.version(),
                   "dtype": "float32", "tf32": False, "autocast": False, "sdpa_backend": "math",
                   "seed": 42, "deterministic_algorithms": True, "image_id": args.image_id,
                   "reference_script_sha256": sha256(__file__), "cases_sha256": sha256(args.cases),
                   "uv_lock_sha256": sha256(Path(__file__).with_name("uv.lock")), "load_seconds": load_seconds}
    if device.type == "cuda":
        environment.update(gpu=torch.cuda.get_device_name(device), compute_capability=list(torch.cuda.get_device_capability(device)),
                           peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                           peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    document = {"version": 1, "status": "completed", "reference": "pinned-publisher-python-independent-of-rust",
                "artifact_manifest": manifest, "environment": environment,
                "comparison_policy": {"encoding": "exact", "logit_abs_tolerance": 1e-4, "logit_rel_tolerance": 1e-4,
                                      "probability_abs_tolerance": 1e-5, "probability_rel_tolerance": 1e-4},
                "cases": results}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": "completed", "cases": len(results), "output_sha256": sha256(args.output)}))


if __name__ == "__main__":
    main()
