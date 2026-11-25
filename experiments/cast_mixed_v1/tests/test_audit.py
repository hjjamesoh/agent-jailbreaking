"""Regression cases from the pilot; CPU/stdlib only."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import json
from contextlib import redirect_stdout
import io

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("mixed_audit", ROOT / "audit.py")
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)


class AuditTests(unittest.TestCase):
    def test_actual_correct_answer_with_format_noncompliance(self):
        text = "Work: $165 + $132 = $297\n#### John paid $297 for everything."
        row = {"text": text, "layout": "b_only", "expected": "297"}
        scores = auditor.rescore(row)
        self.assertTrue(scores["B_flexible_correct"])
        self.assertFalse(scores["B_harness_strict_correct"])
        self.assertEqual(scores["B_scope"], "b_only_full_response")

    def test_no_A_number_leaks_into_B_and_missing_B_stays_failure(self):
        for text in ["[[A]]\n297\n[[B]]\nI cannot help.", "[[A]]\n#### 297"]:
            scores = auditor.rescore({"text": text, "layout": "mixed_AB", "expected": "297"})
            self.assertFalse(scores["B_flexible_correct"])

    def test_old_records_are_not_mutated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "records").mkdir()
            row = {"key": "fixture", "pair": 0, "arm": "baseline", "layout": "b_only",
                   "text": "#### John paid $297.", "expected": "297", "B_numeric_correct": False,
                   "condition_similarities": {}}
            path = root / "records/fixture.json"
            path.write_text(json.dumps(row), encoding="utf-8")
            (root / "gate.json").write_text("{}", encoding="utf-8")
            before = path.read_bytes()
            with redirect_stdout(io.StringIO()):
                auditor.audit(root)
            self.assertEqual(before, path.read_bytes())
            summary = json.loads((root / "audit_lmeval/summary.json").read_text())
            self.assertEqual(summary["summary"][0]["flexible_correct"], 1)


if __name__ == "__main__":
    unittest.main()
