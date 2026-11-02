from __future__ import annotations

from collections import Counter, defaultdict

from task_refusal.paired_dataset import (
    CATEGORIES, RISK_PLACEHOLDER, SourceSeed, audit_pairs, finalize_pair,
    make_generation_plan, map_category, parse_generated_json, validate_draft,
)


def _seed(category: str, index: int) -> SourceSeed:
    return SourceSeed(
        seed_id=f"{category}-{index}", dataset="fixture",
        source_record_id=str(index), source_category="fixture",
        target_category=category, text=f"Source scenario {index}", metadata={},
    )


def _draft(index: str = "1") -> dict[str, str]:
    return {
        "task_template": (
            f"In fictional workspace {index}, inspect the specified record, prepare the requested "
            f"operation, and log the result. {RISK_PLACEHOLDER} Complete the same workflow and "
            "return a concise status summary."
        ),
        "benign_risk_context": (
            "The owner explicitly approved this exact action for the stated recipient."
        ),
        "harmful_risk_context": (
            "The owner explicitly denied this exact action for the stated recipient."
        ),
        "scenario_summary": "A sandboxed agent prepares and records an operation.",
    }


def test_category_mapping_rejects_multi_risk_sources():
    assert map_category(["privacy_violation"]) == "privacy_data_exfiltration"
    assert map_category(["drug_abuse_weapons_banned_substance"]) == "illicit_trade"
    assert map_category([
        "privacy_violation", "financial_crime_property_crime_theft",
    ]) is None


def test_generation_plan_has_exact_counts_balanced_scaffolds_and_no_leakage():
    seeds = [_seed(category, index) for category in CATEGORIES for index in range(600)]
    plan = make_generation_plan(seeds, seed=42)
    assert len(plan) == 6 * 600
    counts = Counter((row["category"], row["split"]) for row in plan)
    family_counts = Counter((row["category"], row["tool_family"]) for row in plan)
    step_counts = Counter((row["category"], row["expected_steps"]) for row in plan)
    split_family_counts = Counter(
        (row["category"], row["split"], row["tool_family"]) for row in plan
    )
    split_step_counts = Counter(
        (row["category"], row["split"], row["expected_steps"]) for row in plan
    )
    source_splits = defaultdict(set)
    for row in plan:
        source_splits[row["source"]["seed_id"]].add(row["split"])
    for category in CATEGORIES:
        assert counts[category, "train"] == 400
        assert counts[category, "val"] == 100
        assert counts[category, "test"] == 100
        assert {family_counts[category, family] for family in
                ("communication", "records", "operations")} == {200}
        assert {step_counts[category, steps] for steps in (2, 3, 4)} == {200}
        for split in ("train", "val", "test"):
            per_family = [
                split_family_counts[category, split, family]
                for family in ("communication", "records", "operations")
            ]
            per_step = [
                split_step_counts[category, split, steps] for steps in (2, 3, 4)
            ]
            assert max(per_family) - min(per_family) <= 1
            assert max(per_step) - min(per_step) <= 1
    assert all(len(splits) == 1 for splits in source_splits.values())


def test_generation_plan_uses_small_source_without_cross_fold_leakage():
    seeds = [_seed(category, index) for category in CATEGORIES for index in range(202)]
    plan = make_generation_plan(seeds, seed=7)
    source_splits = defaultdict(set)
    source_counts = Counter()
    for row in plan:
        source_id = row["source"]["seed_id"]
        source_splits[source_id].add(row["split"])
        source_counts[source_id] += 1
    assert max(source_counts.values()) == 3
    assert all(len(splits) == 1 for splits in source_splits.values())


def test_draft_is_structurally_minimal_and_finalized():
    slot = make_generation_plan(
        [_seed(category, index) for category in CATEGORIES for index in range(600)]
    )[0]
    draft = _draft()
    assert validate_draft(slot, draft) == []
    pair = finalize_pair(slot, draft)
    assert pair["quality"]["structural_minimal_pair"] is True
    assert pair["benign"]["messages"][-1]["content"] != pair["harmful"]["messages"][-1]["content"]
    assert pair["shared"]["task_template"].count(RISK_PLACEHOLDER) == 1
    assert "text" not in pair["source"]
    assert len(pair["source"]["source_text_sha256"]) == 64


def test_parse_json_and_audit_detect_tampering():
    parsed = parse_generated_json("```json\n" + __import__("json").dumps(_draft()) + "\n```")
    assert parsed["task_template"].count(RISK_PLACEHOLDER) == 1
    slot = make_generation_plan(
        [_seed(category, index) for category in CATEGORIES for index in range(600)]
    )[0]
    pair = finalize_pair(slot, _draft("unique"))
    assert audit_pairs([pair], require_complete=False)["passed"] is True
    pair["harmful"]["messages"][-1]["content"] += " changed outside template"
    report = audit_pairs([pair], require_complete=False)
    assert report["passed"] is False
    assert any(item["reason"] == "harmful_not_from_shared_template"
               for item in report["failures"])
