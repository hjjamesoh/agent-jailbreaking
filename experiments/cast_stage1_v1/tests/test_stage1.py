import argparse
from contextlib import redirect_stdout
import csv
from decimal import Decimal
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import data  # noqa: E402

spec = importlib.util.spec_from_file_location("cast_stage1_run", ROOT / "run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class DataTests(unittest.TestCase):
    def test_standard_harmbench_selection_is_stratified_and_verbatim(self):
        stream = io.StringIO()
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "Behavior",
                "FunctionalCategory",
                "SemanticCategory",
                "ContextString",
                "BehaviorID",
            ],
        )
        writer.writeheader()
        for category in ("alpha", "beta", "gamma"):
            for index in range(4):
                writer.writerow(
                    {
                        "Behavior": f"Original fixture {category} {index}",
                        "FunctionalCategory": "standard",
                        "SemanticCategory": category,
                        "ContextString": "",
                        "BehaviorID": f"{category}{index}",
                    }
                )
        writer.writerow(
            {
                "Behavior": "Must not select this",
                "FunctionalCategory": "contextual",
                "SemanticCategory": "alpha",
                "ContextString": "Missing context",
                "BehaviorID": "excluded",
            }
        )
        result = data.choose_harmful(stream.getvalue(), count=6)
        self.assertEqual([r["category"] for r in result], ["alpha", "beta", "gamma"] * 2)
        self.assertEqual(result, data.choose_harmful(stream.getvalue(), count=6))
        self.assertTrue(all(r["prompt"] == r["original_prompt"] for r in result))
        self.assertNotIn("excluded", [r["source_id"] for r in result])

    def test_math_source_answer_is_retained(self):
        text = "\n".join(
            json.dumps({"question": f"Fixture {i}", "answer": f"Work #### {i}"}) for i in range(25)
        )
        rows = data.choose_benign(text)
        self.assertEqual(len(rows), 20)
        self.assertEqual(len({r["source_id"] for r in rows}), 20)
        self.assertTrue(all(r["prompt"].startswith(r["original_prompt"]) for r in rows))
        self.assertTrue(all(r["expected"] == str(r["source_id"]) for r in rows))

    def test_answer_parser_does_not_pick_intermediate_numbers(self):
        self.assertIsNone(data.final_number("I cannot help. The problem mentions 42."))
        self.assertIsNone(data.final_number("We first compute 12 and then 99"))
        self.assertEqual(data.final_number("Work 8+9.\n#### 17"), Decimal(17))
        self.assertEqual(data.final_number("#### -1,234.50"), Decimal("-1234.5"))
        self.assertIsNone(data.final_number("#### 17\nunfinished extra reasoning"))

    def test_plan_is_240_unique_responses_including_sign_and_random_controls(self):
        prompts = [{"id": str(i)} for i in range(40)]
        plan = list(data.jobs(prompts))
        self.assertEqual(len(plan), 240)
        self.assertEqual(len({j["key"] for j in plan}), 240)
        for i in range(40):
            conditions = [j for j in plan if j["prompt_id"] == str(i)]
            self.assertEqual(len(conditions), 6)
            self.assertEqual(sum(j["alpha"] < 0 for j in conditions), 1)
            self.assertEqual(sum(j["direction"] == "random" for j in conditions), 1)

    def test_modified_download_is_rejected_without_overwriting(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            path = root / "file.json"
            path.write_bytes(b"modified")
            asset = {
                "path": "file.json",
                "url": "https://example.invalid/unused",
                "git_blob_sha1": data.git_blob_sha(b"expected"),
            }
            with self.assertRaisesRegex(ValueError, "Existing asset changed"):
                data.fetch_asset(root, asset)
            self.assertEqual(path.read_bytes(), b"modified")

    def test_download_paths_cannot_escape_output(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "escapes"):
                data.asset_path(Path(root), "../other")

    def test_upstream_manifest_is_fully_pinned(self):
        source = data.read_json(ROOT / "sources.json")
        self.assertEqual(len(source["assets"]), 15)
        self.assertEqual(len({a["path"] for a in source["assets"]}), 15)
        for asset in source["assets"]:
            self.assertEqual(len(asset["git_blob_sha1"]), 40)
            self.assertTrue(asset["url"].startswith("https://raw.githubusercontent.com/"))


class ReviewTests(unittest.TestCase):
    def setUp(self):
        quiet = redirect_stdout(io.StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)

    def fixture(self, root):
        (root / "records").mkdir()
        prompts = [
            {"id": "h", "kind": "harmful", "prompt": "Harmful fixture"},
            {"id": "b", "kind": "benign", "prompt": "Math fixture"},
        ]
        plan = list(data.jobs(prompts))
        for job in plan:
            row = next(r for r in prompts if r["id"] == job["prompt_id"])
            success = row["kind"] == "benign" and job["arm"] == "baseline"
            result = {
                **job,
                **row,
                "text": "#### 12" if success else "A fixture answer",
                "truncated": False,
                "empty": False,
                "math_answer_parsed": success,
                "numeric_correct": success,
                "strict_math_success": success if row["kind"] == "benign" else None,
            }
            data.write_json(root / "records" / (job["key"] + ".json"), result)
        data.write_json(root / "plan.json", plan)
        return argparse.Namespace(run=root, output=root, labels=None, arm=None, limit=6)

    def test_no_labels_means_unknown_safety_not_zero_risk(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.fixture(Path(temp))
            runner.report(args)
            summary = data.read_json(args.run / "summary.json")
            self.assertTrue(summary["complete"])
            self.assertTrue(all(r["unsafe_support_reviewed"] == 0 for r in summary["summary"]))
            self.assertTrue(all(r["refusal_reviewed"] == 0 for r in summary["summary"]))
            changed = next(
                r
                for r in summary["paired_deltas"]
                if r["kind"] == "benign" and r["arm"] == "refusal_1.7"
            )
            self.assertEqual(changed["strict_math_success_delta"]["mean"], -1)
            self.assertNotIn("human_refusal_delta", changed)

    def test_review_is_blinded_and_cannot_overwrite_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.fixture(Path(temp))
            args.arm = "refusal_1.7"
            runner.make_review(args)
            path = args.run / "review.csv"
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                self.assertNotIn("arm", reader.fieldnames)
            self.assertEqual(len(rows), 4)
            with self.assertRaisesRegex(ValueError, "overwrite"):
                runner.make_review(args)
            self.assertEqual(len(runner.labels_for(path, runner.read_records(args.run))), 4)
            rows[0]["text"] = "changed"
            with path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "no longer matches"):
                runner.labels_for(path, runner.read_records(args.run))

    def test_partial_run_is_never_marked_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.fixture(Path(temp))
            next((args.run / "records").glob("*.json")).unlink()
            runner.report(args)
            self.assertFalse(data.read_json(args.run / "summary.json")["complete"])


if __name__ == "__main__":
    unittest.main()
