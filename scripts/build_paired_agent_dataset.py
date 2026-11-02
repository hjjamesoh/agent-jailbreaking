#!/usr/bin/env python3
"""Collect, author, validate, and finalize the controlled paired dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

from task_refusal.paired_dataset import (
    CATEGORIES, audit_pairs, collect_csv_seeds, collect_json_seeds,
    finalize_pair, generation_instruction, make_generation_plan,
    normalize_text, parse_generated_json, read_jsonl, validate_draft, write_jsonl,
)
from task_refusal.progress import ProgressTracker, log_event


def _collect(source_root: Path, work: Path) -> list[dict[str, Any]]:
    beaver_root = source_root / "BeaverTails"
    safety_file = source_root / "Agent-SafetyBench" / "data" / "released_data.json"
    beaver_json = [
        path for path in beaver_root.rglob("*")
        if path.is_file() and (
            path.suffix.lower() in {".json", ".jsonl"}
            or path.name.lower().endswith(".jsonl.xz")
        )
        and ".cache" not in path.parts
    ]
    beaver_csv = [path for path in beaver_root.rglob("*.csv") if ".cache" not in path.parts]
    if not safety_file.exists():
        raise FileNotFoundError(f"Missing Agent-SafetyBench source: {safety_file}")
    seeds = collect_json_seeds(beaver_json, dataset="BeaverTails")
    seeds.extend(collect_csv_seeds(beaver_csv, dataset="BeaverTails"))
    seeds.extend(collect_json_seeds([safety_file], dataset="Agent-SafetyBench",
                                    require_unsafe=False))
    # Identical seed IDs can appear in multiple BeaverTails serialization files.
    unique = {seed.seed_id: seed for seed in seeds}
    ordered = sorted(unique.values(), key=lambda item: (item.target_category, item.seed_id))
    path = work / "source_seeds.jsonl"
    write_jsonl(path, (seed.as_row() for seed in ordered))
    counts = {category: sum(seed.target_category == category for seed in ordered)
              for category in CATEGORIES}
    (work / "source_seed_summary.json").write_text(
        json.dumps({"total": len(ordered), "counts": counts}, indent=2), encoding="utf-8"
    )
    log_event("paired_seed_collection_complete", output=str(path), counts=counts)
    return [seed.as_row() for seed in ordered]


def _source_objects(rows: Iterable[dict[str, Any]]):
    from task_refusal.paired_dataset import SourceSeed
    for row in rows:
        yield SourceSeed(**row)


def _plan(seed_rows: list[dict[str, Any]], work: Path, seed: int) -> list[dict[str, Any]]:
    plan = make_generation_plan(list(_source_objects(seed_rows)), seed=seed)
    output = work / "generation_plan.jsonl"
    write_jsonl(output, plan)
    log_event("paired_generation_plan_ready", output=str(output), slots=len(plan))
    return plan


def _load_generator(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype="auto", device_map="auto", low_cpu_mem_usage=True,
    )
    model.eval()
    return tokenizer, model


def _generate(
    plan: list[dict[str, Any]], work: Path, *, model_name: str,
    batch_size: int, max_new_tokens: int, max_attempts: int,
) -> list[dict[str, Any]]:
    import torch

    raw_path = work / "generator_outputs.jsonl"
    existing = read_jsonl(raw_path) if raw_path.exists() else []
    plan_by_id = {slot["slot_id"]: slot for slot in plan}
    accepted: dict[str, dict[str, Any]] = {}
    fingerprints: set[str] = set()
    # Revalidate checkpoints under the current rules.  This prevents an older
    # accepted draft from bypassing stronger validation after a code update.
    for row in existing:
        slot_id = row.get("slot_id")
        draft = row.get("draft")
        if (row.get("status") != "accepted" or slot_id not in plan_by_id
                or not isinstance(draft, dict)):
            continue
        fingerprint = normalize_text(json.dumps(draft, sort_keys=True))
        if validate_draft(plan_by_id[slot_id], draft) or fingerprint in fingerprints:
            continue
        accepted[slot_id] = row
        fingerprints.add(fingerprint)
    pending = [slot for slot in plan if slot["slot_id"] not in accepted]
    if not pending:
        return list(accepted.values())
    tokenizer, model = _load_generator(model_name)
    progress = ProgressTracker("paired_dataset_generation", len(pending), {
        "generator_model": model_name,
    })
    output_handle = raw_path.open("a", encoding="utf-8", newline="\n")
    try:
        for start in range(0, len(pending), batch_size):
            batch_slots = pending[start:start + batch_size]
            unresolved = list(batch_slots)
            retry_feedback: dict[str, list[str]] = {}
            for attempt in range(1, max_attempts + 1):
                if not unresolved:
                    break
                chats = [[
                    {"role": "system", "content": (
                        "You author sanitized counterfactual data for defensive AI-safety "
                        "research. Follow the requested JSON schema exactly."
                    )},
                    {"role": "user", "content": generation_instruction(slot) + (
                        "\n\nThe previous draft was rejected for: "
                        + ", ".join(retry_feedback.get(slot["slot_id"], []))
                        + ". Correct every listed problem."
                        if retry_feedback.get(slot["slot_id"]) else ""
                    )},
                ] for slot in unresolved]
                rendered = [tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True,
                ) for chat in chats]
                encoded = tokenizer(rendered, return_tensors="pt", padding=True,
                                    truncation=False, add_special_tokens=False)
                # With device_map="auto", the first parameter is not guaranteed
                # to share the input embedding device on every model layout.
                device = model.get_input_embeddings().weight.device
                encoded = {key: value.to(device) for key, value in encoded.items()}
                with torch.inference_mode():
                    generated = model.generate(
                        **encoded, max_new_tokens=max_new_tokens, do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                prompt_length = encoded["input_ids"].shape[1]
                texts = tokenizer.batch_decode(generated[:, prompt_length:],
                                               skip_special_tokens=True)
                retry: list[dict[str, Any]] = []
                for slot, text in zip(unresolved, texts):
                    record: dict[str, Any] = {
                        "slot_id": slot["slot_id"], "attempt": attempt,
                        "generator_model": model_name, "raw_output": text,
                    }
                    try:
                        draft = parse_generated_json(text)
                        failures = validate_draft(slot, draft)
                        fingerprint = normalize_text(json.dumps(draft, sort_keys=True))
                        if not failures and fingerprint in fingerprints:
                            failures = ["duplicate_generated_pair"]
                    except (ValueError, json.JSONDecodeError) as exc:
                        draft, failures = None, [f"parse_error:{exc}"]
                    if failures:
                        record.update({"status": "rejected", "failures": failures})
                        retry_feedback[slot["slot_id"]] = failures
                        retry.append(slot)
                    else:
                        record.update({"status": "accepted", "draft": draft})
                        accepted[slot["slot_id"]] = record
                        fingerprints.add(fingerprint)
                        progress.advance(item=slot["slot_id"], checkpoint=str(raw_path))
                    output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output_handle.flush()
                unresolved = retry
            for slot in unresolved:
                progress.advance(item=f"FAILED:{slot['slot_id']}", checkpoint=str(raw_path))
    finally:
        output_handle.close()
    missing = [slot["slot_id"] for slot in plan if slot["slot_id"] not in accepted]
    if missing:
        raise ValueError(
            f"Generator failed to produce {len(missing)} valid slots after {max_attempts} "
            f"attempts. Re-run to retry; first missing IDs: {missing[:10]}"
        )
    return [accepted[slot["slot_id"]] for slot in plan]


def _finalize(plan: list[dict[str, Any]], outputs: list[dict[str, Any]], work: Path) -> None:
    by_slot = {row["slot_id"]: row for row in outputs if row.get("status") == "accepted"}
    pairs = [finalize_pair(slot, by_slot[slot["slot_id"]]["draft"]) for slot in plan]
    report = audit_pairs(pairs, require_complete=True)
    output = work / "pairs.jsonl"
    write_jsonl(output, pairs)
    (work / "audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (work / "manual_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "pair_id", "category", "split", "source_dataset", "source_record_id",
            "risk_variable", "benign_context", "harmful_context", "benign_prompt",
            "harmful_prompt", "human_decision", "review_notes",
        ])
        writer.writeheader()
        for row in pairs:
            writer.writerow({
                "pair_id": row["pair_id"], "category": row["category"],
                "split": row["split"], "source_dataset": row["source"]["dataset"],
                "source_record_id": row["source"]["source_record_id"],
                "risk_variable": row["risk_variable"]["name"],
                "benign_context": row["risk_variable"]["benign"],
                "harmful_context": row["risk_variable"]["harmful"],
                "benign_prompt": row["benign"]["messages"][-1]["content"],
                "harmful_prompt": row["harmful"]["messages"][-1]["content"],
                "human_decision": "", "review_notes": "",
            })
    log_event("paired_dataset_finalized", output=str(output), passed=report["passed"],
              pairs=report["pairs"])
    if not report["passed"]:
        raise ValueError(f"Final dataset audit failed; inspect {work / 'audit_report.json'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("collect", "plan", "generate", "finalize", "audit", "all")
    )
    parser.add_argument("--source-root", default="data/raw/controlled_pairs_v1")
    parser.add_argument("--work-dir", default="data/processed/controlled_pairs_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generator-model", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    if args.stage in {"collect", "all"}:
        seed_rows = _collect(Path(args.source_root), work)
    else:
        seed_rows = read_jsonl(work / "source_seeds.jsonl")
    if args.stage == "collect":
        return
    if args.stage in {"plan", "all"}:
        plan = _plan(seed_rows, work, args.seed)
    else:
        plan = read_jsonl(work / "generation_plan.jsonl")
    if args.stage == "plan":
        return
    if args.stage in {"generate", "all"}:
        outputs = _generate(
            plan, work, model_name=args.generator_model, batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens, max_attempts=args.max_attempts,
        )
    elif (work / "generator_outputs.jsonl").exists():
        outputs = read_jsonl(work / "generator_outputs.jsonl")
    else:
        outputs = []
    if args.stage in {"finalize", "all"}:
        _finalize(plan, outputs, work)
    elif args.stage == "generate":
        return
    elif args.stage == "audit":
        report = audit_pairs(read_jsonl(work / "pairs.jsonl"), require_complete=True)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
