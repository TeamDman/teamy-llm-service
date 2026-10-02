"""Contract checks without loading a model or using a network."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from reference import NATIVE_REVISION, ONNX_REVISION, prepare_case, verify_artifacts


class ReferenceContractTests(unittest.TestCase):
    def setUp(self):
        self.configuration = {
            "default_question": "Which option should be chosen?",
            "max_tokens": 1024,
            "head_tokens": 256,
        }
        self.case = {
            "id": "synthetic",
            "state": "A decision needs review.",
            "options": [
                {"id": "review", "text": "Review the proposal"},
                {"id": "wait", "text": "Wait for more information"},
            ],
        }

    def test_default_question_and_ordered_ids_are_explicit(self):
        row, ids, canonical = prepare_case(self.case, self.configuration)
        self.assertEqual(ids, ["review", "wait"])
        self.assertEqual(row["type"], "choice")
        self.assertEqual(row["question"], self.configuration["default_question"])
        payload = json.loads(canonical)
        self.assertEqual(payload["options"], self.case["options"])
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertEqual(payload["head_tokens"], 256)

    def test_question_override_is_in_input_digest(self):
        _, _, original = prepare_case(self.case, self.configuration)
        self.case["question"] = "Which action preserves evidence?"
        row, _, changed = prepare_case(self.case, self.configuration)
        self.assertEqual(row["question"], self.case["question"])
        self.assertNotEqual(original, changed)

    def test_duplicate_ids_are_rejected(self):
        self.case["options"][1]["id"] = "review"
        with self.assertRaisesRegex(ValueError, "Duplicate option IDs"):
            prepare_case(self.case, self.configuration)

    def test_option_count_bounds(self):
        for count in (0, 1, 21):
            with self.subTest(count=count):
                case = copy.deepcopy(self.case)
                case["options"] = [{"id": str(index), "text": "Synthetic option"}
                                   for index in range(count)]
                with self.assertRaisesRegex(ValueError, "2–20"):
                    prepare_case(case, self.configuration)

    def test_unknown_fields_cannot_silently_introduce_images(self):
        for field in ("image", "backend", "type"):
            with self.subTest(field=field):
                case = copy.deepcopy(self.case)
                case[field] = "unsupported"
                with self.assertRaises(ValueError):
                    prepare_case(case, self.configuration)
        self.case["options"][0]["image"] = "unsupported"
        with self.assertRaises(ValueError):
            prepare_case(self.case, self.configuration)

    def test_nontext_state_or_empty_values_are_rejected(self):
        self.case["state"] = {"structured": True}
        with self.assertRaisesRegex(ValueError, "state must be text"):
            prepare_case(self.case, self.configuration)
        self.case["state"] = "Synthetic state"
        self.case["options"][0]["text"] = "   "
        with self.assertRaises(ValueError):
            prepare_case(self.case, self.configuration)

    def test_same_description_with_distinct_ids_is_preserved(self):
        self.case["options"][1]["text"] = self.case["options"][0]["text"]
        row, ids, _ = prepare_case(self.case, self.configuration)
        self.assertEqual(row["options"][0], row["options"][1])
        self.assertEqual(ids, ["review", "wait"])

    def test_artifact_mismatch_and_path_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            artifact = root / "synthetic.json"
            artifact.write_bytes(b"synthetic fixture")
            manifest = {
                "native": {"revision": NATIVE_REVISION, "files": [{
                    "path": "synthetic.json",
                    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                }]},
                "onnx": {"revision": ONNX_REVISION},
            }
            verify_artifacts(root, manifest)
            artifact.write_bytes(b"changed fixture")
            with self.assertRaisesRegex(ValueError, "checksum differs"):
                verify_artifacts(root, manifest)
            manifest["native"]["files"][0]["path"] = "../outside.json"
            with self.assertRaisesRegex(ValueError, "Unsafe artifact path"):
                verify_artifacts(root, manifest)
            manifest["native"]["revision"] = "unsupported-revision"
            with self.assertRaisesRegex(ValueError, "revisions differ"):
                verify_artifacts(root, manifest)


if __name__ == "__main__":
    unittest.main()
