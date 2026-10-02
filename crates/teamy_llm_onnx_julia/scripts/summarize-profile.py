"""Summarize recorded ORT placement, without claiming EP registration proves GPU execution."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


HEAVY_OPS = {"MatMul", "FusedMatMul", "Gemm", "Attention", "MultiHeadAttention", "MatMulNBits"}


def summarize(path):
    if path.stat().st_size > 128 * 1024**2:
        raise ValueError("Profile exceeds the bounded metadata size")
    data = path.read_bytes()
    events = json.loads(data)
    if not isinstance(events, list):
        raise ValueError("Expected an ORT Chrome-trace array")
    providers, nodes, heavy = defaultdict(Counter), Counter(), Counter()
    durations = Counter()
    cpu_types = {"input": Counter(), "output": Counter()}
    cpu_typed_events = 0
    for event in events:
        if not isinstance(event, dict) or event.get("cat") != "Node":
            continue
        args = event.get("args")
        if not isinstance(args, dict):
            continue
        provider, operation, name = args.get("provider"), args.get("op_name"), event.get("name")
        if not all(isinstance(value, str) and value for value in (provider, operation, name)):
            continue
        providers[provider][operation] += 1
        nodes[(provider, operation, name)] += 1
        duration = event.get("dur", 0)
        if isinstance(duration, (int, float)) and duration >= 0:
            durations[provider] += duration
        if operation in HEAVY_OPS:
            heavy[provider] += 1
        if provider == "CPUExecutionProvider":
            typed = True
            for direction in ("input", "output"):
                shapes = args.get(direction + "_type_shape")
                if not isinstance(shapes, list) or not shapes:
                    typed = False
                    continue
                for shape in shapes:
                    if isinstance(shape, dict):
                        cpu_types[direction].update(shape.keys())
                    else:
                        typed = False
            cpu_typed_events += typed
    cuda, cpu = heavy["CUDAExecutionProvider"], heavy["CPUExecutionProvider"]
    return {"version": 1, "profile_sha256": hashlib.sha256(data).hexdigest(),
            "profile_bytes": len(data), "source": "Recorded ORT node execution events",
            "heavy_operations_checked": sorted(HEAVY_OPS),
            "heavy_cuda_events": cuda, "heavy_cpu_events": cpu,
            "cpu_events_with_input_and_output_type_metadata": cpu_typed_events,
            "cpu_tensor_element_types": {direction: dict(sorted(types.items()))
                for direction, types in cpu_types.items()},
            "heavy_compute_placement_passed": cuda > 0 and cpu == 0,
            "providers": {provider: {"execution_events": sum(operations.values()),
                "recorded_duration_us": durations[provider], "operations": dict(sorted(operations.items()))}
                for provider, operations in sorted(providers.items())},
            "heavy_cuda_nodes": [{"provider": provider, "operation": operation,
                "node": name, "calls": calls}
                for (provider, operation, name), calls in sorted(nodes.items())
                if provider == "CUDAExecutionProvider" and operation in HEAVY_OPS],
            "cpu_nodes": [{"operation": operation, "node": name, "calls": calls}
                for (provider, operation, name), calls in sorted(nodes.items())
                if provider == "CPUExecutionProvider"],
            "limits": "This records this session and inputs only. Event duration totals are not wall time. CPU nodes are enumerated by operation, not automatically classified as harmless shape work."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--require-heavy-cuda", action="store_true")
    args = parser.parse_args()
    report = summarize(args.profile.resolve(strict=True))
    print(json.dumps(report, indent=2))
    if args.require_heavy_cuda and not report["heavy_compute_placement_passed"]:
        raise SystemExit("Recorded placement did not establish CUDA heavy-compute execution")
