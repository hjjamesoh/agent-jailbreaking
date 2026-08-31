import argparse
import json
from pathlib import Path


DEFAULT_SPLITS = [
    "harmful_train",
    "harmful_val",
    "harmful_test",
    "harmless_train",
    "harmless_val",
    "harmless_test",
]


def _read_source_split(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a JSON list")

    prompts = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{idx}: expected object")
        instruction = row.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"{path}:{idx}: missing non-empty instruction")
        prompts.append({
            "prompt": instruction.strip(),
            "source_category": row.get("category"),
        })
    return prompts


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert andyrdt/refusal_direction dataset/splits JSON files to Experiment 0 JSONL."
    )
    parser.add_argument(
        "--upstream-dir",
        default="../upstream_refusal_direction",
        help="Path to a clone of https://github.com/andyrdt/refusal_direction.",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/exp0_llm_refusal_dir/data/paper_splits",
        help="Output directory for converted JSONL splits.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=DEFAULT_SPLITS,
        help="Split basenames to convert, e.g. harmful_train harmless_train.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    upstream_dir = Path(args.upstream_dir)
    if not upstream_dir.is_absolute():
        upstream_dir = (repo_root / upstream_dir).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir

    summaries = []
    for split in args.splits:
        source_path = upstream_dir / "dataset" / "splits" / f"{split}.json"
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        rows = _read_source_split(source_path)
        out_path = out_dir / f"{split}.jsonl"
        _write_jsonl(out_path, rows)
        summaries.append({
            "split": split,
            "source": str(source_path),
            "output": str(out_path),
            "num_prompts": len(rows),
            "num_unique_prompts": len({row["prompt"] for row in rows}),
        })

    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
