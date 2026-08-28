import argparse
import json
from pathlib import Path


def validate_jsonl(path):
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc

            if isinstance(obj, str):
                prompt = obj
            elif isinstance(obj, dict) and "prompt" in obj:
                prompt = obj["prompt"]
            else:
                raise ValueError(
                    f"{path}:{line_no}: expected a JSON string or object with 'prompt'"
                )

            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{path}:{line_no}: prompt must be a non-empty string")
            prompts.append(prompt.strip())

    if not prompts:
        raise ValueError(f"{path}: no prompts found")

    return {
        "path": str(path),
        "num_prompts": len(prompts),
        "num_unique_prompts": len(set(prompts)),
        "num_duplicates": len(prompts) - len(set(prompts)),
        "min_chars": min(len(p) for p in prompts),
        "max_chars": max(len(p) for p in prompts),
    }


def main():
    parser = argparse.ArgumentParser(description="Validate Experiment 0 JSONL prompt files.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=[
            "data/harmful_train.jsonl",
            "data/harmless_train.jsonl",
            "data/harmful_val.jsonl",
            "data/harmless_val.jsonl",
        ],
    )
    args = parser.parse_args()

    summaries = [validate_jsonl(Path(path)) for path in args.paths]
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
