"""Paired net gains must not be mistaken for the number of recovered items."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("oracle_analysis", ROOT / "analyze.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class AnalysisTests(unittest.TestCase):
    def test_net_one_can_be_two_recoveries_and_one_regression(self):
        cases = [{"pair": i, "g": a, "o": b} for i, (a, b) in enumerate([
            (False, True), (False, True), (True, False), (True, True)])]
        r = analysis.transitions(cases, "g", "o")
        self.assertEqual(r["recovered"], [0, 1])
        self.assertEqual(r["worsened"], [2])
        self.assertEqual(r["net_gain"], 1)

    def test_equal_accuracies_can_hide_item_changes(self):
        cases = [{"pair": 0, "o": True, "r": False}, {"pair": 1, "o": False, "r": True}]
        r = analysis.transitions(cases, "o", "r")
        self.assertEqual(r["net_gain"], 0)
        self.assertEqual(r["recovered"], [1])
        self.assertEqual(r["worsened"], [0])

    def test_audit_preserves_raw_records_and_exports_full_answers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "records").mkdir()
            for arm, alpha, correct in [("B_only", 0, True), ("baseline", 0, True),
                                        ("baseline_replay", 0, True), ("global", .85, False),
                                        ("oracle_A", .85, True), ("oracle_A_replay", .85, True)]:
                section = {"text": "#### 1" if correct else "#### 2", "token_ids": [1 if correct else 2],
                           "flexible_correct": correct, "strict_correct": correct,
                           "flexible_extracted": "1" if correct else "2", "truncated": False,
                           "stop_reason": "section_marker", "max_identical_word_run": 1,
                           "distinct_4gram_fraction": None, "fabricated_turn_marker": False}
                row = {"pair": 0, "key": arm, "alpha": alpha, "arm": arm, "a": None if arm == "B_only" else
                       {**section, "token_ids": [3]}, "b": section, "expected": "1", "A_boundary_valid": True,
                       "B_intervention_layer_calls": 1 if arm == "global" else 0,
                       "harmful_prompt": "A fixture", "benign_prompt": "B fixture", "text": section["text"]}
                (root / f"records/{arm}.json").write_text(json.dumps(row), encoding="utf-8")
            before = {p.name: p.read_bytes() for p in (root / "records").iterdir()}
            with redirect_stdout(io.StringIO()):
                r = analysis.diagnose(root, .85)
            self.assertEqual(r["global_to_oracle"]["recovered"], [0])
            self.assertEqual(before, {p.name: p.read_bytes() for p in (root / "records").iterdir()})
            self.assertIn("B RESPONSE:", (root / "analysis_a0p85/cases.txt").read_text())


if __name__ == "__main__":
    unittest.main()
