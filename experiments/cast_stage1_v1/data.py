"""Immutable public inputs for stage 1. No LLM-written scenario generation."""

from __future__ import annotations

from collections import defaultdict
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
from pathlib import Path
import random
import re
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
MODEL = "NousResearch/Hermes-2-Pro-Llama-3-8B"
MODEL_REVISION = "8ab73a6800796d84448bc936db9bac5ad9f984ae"
# CAST paper Table 4: Hermes 2 Pro add-refusal configuration.
LAYERS = [15, 17, 18, 19, 20, 21, 22, 23, 24]
ALPHAS = [-1.7, 0.85, 1.7, 3.4]


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_blob_sha(data):
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def asset_path(output, relative):
    root = Path(output).resolve()
    destination = (root / relative).resolve()
    if not destination.is_relative_to(root):
        raise ValueError("Asset path escapes experiment output")
    return destination


def fetch_asset(output, asset):
    destination = asset_path(output, asset["path"])
    if destination.exists():
        content = destination.read_bytes()
        if git_blob_sha(content) != asset["git_blob_sha1"]:
            raise ValueError(f"Existing asset changed: {destination}; use a new output folder")
        return
    error = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(
                asset["url"], headers={"User-Agent": "CAST-stage1-research"}
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                content = response.read()
            if git_blob_sha(content) != asset["git_blob_sha1"]:
                raise ValueError(f"Upstream checksum mismatch: {asset['path']}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_bytes(content)
            temporary.replace(destination)
            return
        except (urllib.error.URLError, TimeoutError) as exc:
            error = exc
            if attempt < 2:
                time.sleep(1)
    raise RuntimeError(f"Cannot download {asset['url']}: {error}")


def choose_harmful(csv_text, count=20, seed=20261125):
    groups = defaultdict(list)
    for row in csv.DictReader(io.StringIO(csv_text)):
        # Contextual/copyright/completion tasks require different rendering/evaluation.
        if row["FunctionalCategory"] == "standard" and not row["ContextString"].strip():
            groups[row["SemanticCategory"]].append(row)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    selected = []
    for index in range(max(map(len, groups.values()))):
        for category in sorted(groups):
            if index < len(groups[category]):
                row = groups[category][index]
                selected.append(
                    {
                        "id": "harmbench:" + row["BehaviorID"],
                        "kind": "harmful",
                        "source": "HarmBench",
                        "source_id": row["BehaviorID"],
                        "category": category,
                        "original_prompt": row["Behavior"],
                        "prompt": row["Behavior"],
                        "expected": None,
                    }
                )
    if len(selected) < count:
        raise ValueError("Not enough standard HarmBench items")
    return selected[:count]


def decimal_value(value):
    try:
        result = Decimal(value.strip().replace(",", ""))
        return result if result.is_finite() else None
    except InvalidOperation:
        return None


def final_number(text):
    # Require the requested explicit final-answer marker; never grab a random
    # intermediate number from a refusal or unfinished mathematical derivation.
    matches = re.findall(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\.?\s*$", text)
    return decimal_value(matches[-1]) if matches else None


def choose_benign(jsonl_text, count=20, seed=20261125):
    rows = [json.loads(line) for line in jsonl_text.splitlines() if line.strip()]
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    result = []
    for index in indices[:count]:
        row = rows[index]
        expected = row["answer"].split("####")[-1].strip()
        if decimal_value(expected) is None:
            raise ValueError(f"Invalid GSM8K final answer: {index}")
        result.append(
            {
                "id": f"gsm8k:train:{index}",
                "kind": "benign",
                "source": "GSM8K train",
                "source_id": index,
                "category": "math",
                "original_prompt": row["question"],
                "prompt": row["question"]
                + "\nGive a concise solution and end with #### followed by the final number.",
                "expected": expected,
            }
        )
    return result


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    sources = read_json(ROOT / "sources.json")
    for index, asset in enumerate(sources["assets"]):
        fetch_asset(output, asset)
        print(f"prepare {index + 1}/{len(sources['assets'])}: {asset['path']}", flush=True)
    raw = output / "raw"
    prompts = choose_harmful((raw / "harmbench.csv").read_text(encoding="utf-8"))
    prompts += choose_benign((raw / "gsm8k_train.jsonl").read_text(encoding="utf-8"))
    if len({r["prompt"] for r in prompts}) != 40:
        raise ValueError("Duplicate calibration prompts")
    path = output / "prompts.json"
    if path.exists() and read_json(path) != prompts:
        raise ValueError("Prepared data changed; choose a new output folder")
    write_json(path, prompts)
    prefixes = read_json(output / "vendor/docs/demo-data/behavior_refusal.json")
    alpaca = read_json(output / "vendor/docs/demo-data/alpaca.json")
    questions = [r["question"] for r in alpaca["train"][:100]]
    if len(questions) != 100 or any(
        len(prefixes[k]) < 100 for k in ("non_compliant_responses", "compliant_responses")
    ):
        raise ValueError("Expected official 100 questions x 100 suffix pairs")
    if set(questions) & {r["original_prompt"] for r in prompts}:
        raise ValueError("Extraction/calibration prompt overlap")
    write_json(
        output / "provenance.json",
        {
            "sources": sources,
            "prompts_sha256": digest(prompts),
            "extraction_pairs": 10000,
            "harmful": 20,
            "benign": 20,
            "synthetic_prompts_written_by_us": 0,
            "note": "CAST upstream demo data are machine-generated demo artifacts, not certified "
            "identical to paper experiment data. Evaluation source prompts are retained; "
            "GSM8K has one fixed answer-format instruction appended.",
            "reserved_future_test": "Do not reuse these source IDs as held-out scope-test items.",
        },
    )
    print(f"PREPARE_COMPLETE: {output}", flush=True)


def validate_assets(output):
    sources = read_json(ROOT / "sources.json")
    for asset in sources["assets"]:
        path = asset_path(output, asset["path"])
        if not path.exists() or git_blob_sha(path.read_bytes()) != asset["git_blob_sha1"]:
            raise ValueError(f"Missing or modified asset: {path}. Run prepare.")


def validate_prepared(output):
    validate_assets(output)
    provenance = read_json(output / "provenance.json")
    prompts = read_json(output / "prompts.json")
    expected = choose_harmful((output / "raw/harmbench.csv").read_text(encoding="utf-8"))
    expected += choose_benign((output / "raw/gsm8k_train.jsonl").read_text(encoding="utf-8"))
    if prompts != expected or provenance["prompts_sha256"] != digest(prompts):
        raise ValueError("Prepared prompts changed; do not mix modified data into this experiment")
    return prompts


def jobs(prompts):
    arms = [("baseline", 0.0, "none")] + [
        (f"refusal_{alpha:g}", alpha, "refusal") for alpha in ALPHAS
    ]
    arms.append(("random_1.7", 1.7, "random"))
    for row in prompts:
        for arm, alpha, direction in arms:
            item = {"prompt_id": row["id"], "arm": arm, "alpha": alpha, "direction": direction}
            yield {**item, "key": digest(item)[:20]}
