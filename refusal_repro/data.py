import json
from pathlib import Path

def load_jsonl(path):
    items = []
    with open(Path(path), "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, str):
                prompt = obj
            elif isinstance(obj, dict) and "prompt" in obj:
                prompt = obj["prompt"]
            else:
                raise ValueError(f"{path}:{line_no}: expected a JSON string or object with 'prompt'")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{path}:{line_no}: prompt must be a non-empty string")
            items.append(prompt.strip())
    if not items:
        raise ValueError(f"No prompts found in {path}")
    return items
