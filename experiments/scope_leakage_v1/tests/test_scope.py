"""Meaningful CPU tests for data isolation, calibration gates and paired analysis."""

import argparse
from contextlib import redirect_stdout
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scope_data as data  # noqa: E402

spec = importlib.util.spec_from_file_location("scope_runner", ROOT / "run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class DataTests(unittest.TestCase):
    def test_disjoint_families_and_counts(self):
        rows = data.build_data()
        self.assertEqual(data.validate_data(rows)["counts"], {"train": 24, "cal": 12, "test": 12})
        self.assertEqual(len({r["family"] for r in rows}), 48)
        self.assertEqual(rows, data.build_data())
        broken = [dict(r) for r in rows]
        broken[0]["split"] = "test"
        with self.assertRaises(ValueError):
            data.validate_data(broken)

    def test_oracle_mask_is_exact_and_sham_keeps_position(self):
        row = data.build_data()[0]
        for position in range(3):
            for context in ("mixed", "benign_context"):
                parts = data.segments(row, position, context)
                active = [data.active_for("scoped", p["id"]) for p in parts]
                self.assertEqual(active, [i == position for i in range(3)])
                self.assertEqual(sum(p["kind"] == "harmful" for p in parts), context == "mixed")
                self.assertEqual(len([p for p in parts if p["expected"] is not None]), 2)
                self.assertNotIn("harmful", data.render_prompt(parts))

    def test_complete_plan_has_no_duplicates(self):
        jobs = list(
            data.plan(
                data.build_data(),
                "test",
                [19],
                [1],
                runner.ARMS,
                [0, 1, 2],
                ["mixed", "benign_context"],
            )
        )
        self.assertEqual(len(jobs), 504)
        self.assertEqual(len({j["key"] for j in jobs}), 504)
        self.assertEqual(sum(j["arm"] == "baseline" for j in jobs), 72)

    def test_exact_answer_not_a_keyword_or_refusal_judge(self):
        part = {"expected": "R31243"}
        self.assertTrue(data.task_success(part, "R31243"))
        self.assertFalse(data.task_success(part, "I cannot help. R31243"))
        self.assertFalse(data.task_success(part, "R31244"))
        self.assertIsNone(data.task_success({"expected": None}, "Anything"))

    def test_small_smoke_round_robins_categories(self):
        rows = runner.subset(data.build_data(), "cal", 6)
        self.assertEqual(len({r["category"] for r in rows}), 6)

    def test_checkpoint_corruption_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "broken.jsonl"
            path.write_text('{"id": 1}\n{"id":', encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                data.read_jsonl(path)


class ReviewTests(unittest.TestCase):
    def setUp(self):
        quiet = redirect_stdout(io.StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)

    def calibration(self, root):
        run = root / "cal"
        (run / "records").mkdir(parents=True)
        jobs, annotations = [], []
        for row in data.build_data():
            if row["split"] != "cal":
                continue
            for side in ("harmful", "benign"):
                for arm in ("baseline", "global"):
                    key = data.digest([row["id"], side, arm])[:20]
                    job = {
                        "key": key,
                        "scenario": row["id"],
                        "family": row["family"],
                        "side": side,
                        "arm": arm,
                        "layer": None if arm == "baseline" else 19,
                        "scale": 0 if arm == "baseline" else 1,
                    }
                    refuses = side == "harmful" or arm == "global"
                    part = {
                        "id": "target",
                        "kind": side,
                        "prompt": row[side],
                        "text": "I cannot help with that." if refuses else "A normal answer.",
                        "truncated": False,
                        "format_error": False,
                        "auto_task_success": None,
                        "refusal_hint": refuses,
                        "stop_reason": "end_marker",
                    }
                    data.write_json(run / "records" / (key + ".json"), {**job, "parts": [part]})
                    jobs.append(job)
                    annotations.append(
                        {
                            "record_key": key,
                            "part_id": "target",
                            "question": row[side],
                            "text": part["text"],
                            "human_refusal": int(refuses),
                            "human_unsafe_support": 0,
                            "human_degraded": 0,
                            "notes": "",
                        }
                    )
        data.write_json(run / "plan.json", jobs)
        data.write_json(
            run / "settings.json",
            {
                "stage": "calibrate",
                "limit": 0,
                "max_tokens": 128,
                "bank_sha256": "fake",
                "dataset_sha256": "fake",
            },
        )
        labels = root / "labels.csv"
        self.write_labels(labels, annotations)
        args = argparse.Namespace(
            run=run,
            labels=labels,
            review_layer=19,
            review_scale=1,
            selection=root / "selection.json",
        )
        return args, annotations

    def write_labels(self, path, rows):
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_no_harmful_headroom_is_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _ = self.calibration(Path(temp))
            runner.select(args)
            selected = json.loads(args.selection.read_text())
            self.assertFalse(selected["harmful_support_reduction_observed"])
            self.assertIn("screening only", selected["qualification"])

    def test_blank_labels_and_changed_response_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            args, annotations = self.calibration(Path(temp))
            annotations[0]["human_unsafe_support"] = ""
            self.write_labels(args.labels, annotations)
            with self.assertRaisesRegex(ValueError, "Label all"):
                runner.select(args)
            annotations[0]["human_unsafe_support"] = 0
            annotations[0]["text"] = "Modified answer"
            self.write_labels(args.labels, annotations)
            with self.assertRaisesRegex(ValueError, "no longer matches"):
                runner.select(args)

    def test_smoke_and_partial_runs_cannot_select(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _ = self.calibration(Path(temp))
            settings = json.loads((args.run / "settings.json").read_text())
            settings["limit"] = 2
            data.write_json(args.run / "settings.json", settings)
            with self.assertRaisesRegex(ValueError, "unlimited"):
                runner.select(args)
            next((args.run / "records").glob("*.json")).unlink()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                runner.assert_complete(args.run)

    def test_review_never_overwrites_annotations(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _ = self.calibration(Path(temp))
            with self.assertRaisesRegex(ValueError, "overwrite"):
                runner.review(args)

    def test_report_does_not_invent_unreviewed_safety(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _ = self.calibration(Path(temp))
            args.labels = None
            runner.report(args)
            result = json.loads((args.run / "summary.json").read_text())
            self.assertTrue(result["complete"])
            self.assertTrue(
                all(r["unsafe_support_rate_reviewed"] is None for r in result["summary"])
            )


class PairedTests(unittest.TestCase):
    def test_bootstrap_counts_families_not_repeated_orders(self):
        result = runner.bootstrap_family({"a": [-1, -1, -1], "b": [0, 0, 0]})
        self.assertEqual(result["families"], 2)
        self.assertEqual(result["mean"], -0.5)
        self.assertEqual(result, runner.bootstrap_family({"a": [-1], "b": [0]}))

    def test_paired_report_and_cache_prefix_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "records").mkdir()
            rows = [r for r in data.build_data() if r["split"] == "test"][:2]
            arms = ["baseline", "scoped", "scoped_replay", "baseline_replay"]
            jobs = list(data.plan(rows, "test", [19], [1], arms, [0, 1, 2], ["mixed"]))
            data.write_json(root / "plan.json", jobs)
            for job in jobs:
                row = next(r for r in rows if r["id"] == job["scenario"])
                parts = data.segments(row, job["order"])
                for position, part in enumerate(parts):
                    damaged = job["arm"] == "scoped" and position > job["order"]
                    text = (
                        "I cannot help." if part["id"] == "target" or damaged else part["expected"]
                    )
                    part.update(
                        {
                            "text": text,
                            "position": position,
                            "token_ids": [10 if damaged else 20],
                            "truncated": False,
                            "format_error": False,
                            "refusal_hint": "cannot" in text,
                            "auto_task_success": data.task_success(part, text),
                        }
                    )
                data.write_json(
                    root / "records" / (job["key"] + ".json"),
                    {**job, "family": row["family"], "prompt_token_ids": [1, 2], "parts": parts},
                )
            with redirect_stdout(io.StringIO()):
                runner.report(argparse.Namespace(run=root, labels=None))
            result = json.loads((root / "summary.json").read_text())
            delta = next(
                r
                for r in result["paired_vs_baseline"]
                if r["arm"] == "scoped" and r["region"] == "after"
            )
            self.assertEqual(delta["success_delta"]["mean"], -1)
            self.assertEqual(delta["success_delta"]["families"], 2)
            self.assertTrue(
                all(r["same_token_prefix_through_target"] for r in result["replay_prefix_checks"])
            )
            self.assertTrue(all(r.get("joint_reviewed", 0) == 0 for r in result["summary"]))


if __name__ == "__main__":
    unittest.main()
