"""Placement assertions must fail when no CUDA compute was actually recorded."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location("profile_summary", Path(__file__).with_name("summarize-profile.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PlacementTests(unittest.TestCase):
    def report(self, events):
        with tempfile.TemporaryDirectory(prefix="teamy-ort-profile-test-") as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps(events), encoding="utf-8")
            return module.summarize(path)

    def event(self, provider, operation):
        return {"cat": "Node", "name": "fixture_kernel_time", "dur": 1,
                "args": {"provider": provider, "op_name": operation}}

    def test_registration_without_execution_does_not_establish_gpu(self):
        self.assertFalse(self.report([{"cat": "Session", "name": "CUDAExecutionProvider"}])
                         ["heavy_compute_placement_passed"])

    def test_cpu_matmul_is_reported_even_when_cuda_matmul_also_exists(self):
        report = self.report([self.event("CPUExecutionProvider", "MatMul"),
                              self.event("CUDAExecutionProvider", "MatMul")])
        self.assertFalse(report["heavy_compute_placement_passed"])
        self.assertEqual(report["heavy_cpu_events"], 1)

    def test_mixed_shape_and_cuda_compute_keeps_cpu_assignments_visible(self):
        report = self.report([self.event("CPUExecutionProvider", "Shape"),
                              self.event("CUDAExecutionProvider", "FusedMatMul")])
        self.assertTrue(report["heavy_compute_placement_passed"])
        self.assertEqual(report["cpu_nodes"][0]["operation"], "Shape")
        self.assertEqual(report["heavy_cuda_nodes"], [{"provider": "CUDAExecutionProvider",
            "operation": "FusedMatMul", "node": "fixture_kernel_time", "calls": 1}])
        self.assertEqual(report["cpu_events_with_input_and_output_type_metadata"], 0)

    def test_cpu_integer_claim_requires_recorded_type_metadata(self):
        event = self.event("CPUExecutionProvider", "Shape")
        event["args"].update(input_type_shape=[{"int64": [1]}], output_type_shape=[{"int64": [2]}])
        report = self.report([event])
        self.assertEqual(report["cpu_events_with_input_and_output_type_metadata"], 1)
        self.assertEqual(report["cpu_tensor_element_types"], {"input": {"int64": 1}, "output": {"int64": 1}})


if __name__ == "__main__":
    unittest.main()
