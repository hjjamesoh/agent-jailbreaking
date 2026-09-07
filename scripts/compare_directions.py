import argparse
import csv
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from refusal_repro.analysis import (
    cosine_similarity,
    save_cosine_heatmap,
    top_cosine_matches,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare selected and candidate refusal-direction artifacts."
    )
    p.add_argument(
        "--reference-direction",
        required=True,
        help="Path to the reference direction.pt, usually an Exp0 selected direction.",
    )
    p.add_argument(
        "--target-direction",
        default=None,
        help="Optional path to another selected direction.pt, usually Exp1 agent_direction.pt.",
    )
    p.add_argument(
        "--target-candidates",
        default=None,
        help="Optional path to candidate directions pt, e.g. agent_candidate_directions.pt.",
    )
    p.add_argument("--out", default="runs/direction_comparisons")
    p.add_argument("--run-name", default=None)
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--no-heatmap", action="store_true")
    return p.parse_args()


def _repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_selected_direction(path):
    payload = _safe_torch_load(path)
    if "direction" not in payload:
        raise ValueError(f"{path} does not contain a selected 'direction' tensor.")
    metrics = payload.get("metrics", {})
    layer = payload.get("layer", metrics.get("layer"))
    position = payload.get("position", metrics.get("position"))
    model = payload.get("model")
    return {
        "path": str(path),
        "payload": payload,
        "direction": payload["direction"],
        "model": model,
        "layer": int(layer) if layer is not None else None,
        "position": int(position) if position is not None else None,
        "metrics": metrics,
    }


def _load_candidates(path):
    payload = _safe_torch_load(path)
    if "candidates" not in payload:
        raise ValueError(f"{path} does not contain a 'candidates' tensor.")
    if "positions" not in payload:
        raise ValueError(f"{path} does not contain a 'positions' list.")
    return {
        "path": str(path),
        "payload": payload,
        "candidates": payload["candidates"],
        "positions": payload["positions"],
        "model": payload.get("model"),
        "context_type": payload.get("context_type"),
        "effective_candidate_layers": payload.get("effective_candidate_layers"),
    }


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    out = _repo_path(args.out)
    if args.run_name:
        out = out / args.run_name
    out.mkdir(parents=True, exist_ok=True)

    reference = _load_selected_direction(_repo_path(args.reference_direction))
    summary = {
        "reference_direction": {
            "path": reference["path"],
            "model": reference["model"],
            "layer": reference["layer"],
            "position": reference["position"],
            "metrics": reference["metrics"],
        },
    }

    if args.target_direction is not None:
        target = _load_selected_direction(_repo_path(args.target_direction))
        selected_cosine = cosine_similarity(reference["direction"], target["direction"])
        summary["target_direction"] = {
            "path": target["path"],
            "model": target["model"],
            "layer": target["layer"],
            "position": target["position"],
            "metrics": target["metrics"],
            "cosine_with_reference": selected_cosine,
            "abs_cosine_with_reference": abs(selected_cosine),
        }

    if args.target_candidates is not None:
        target_candidates = _load_candidates(_repo_path(args.target_candidates))
        top_matches, grid = top_cosine_matches(
            reference["direction"],
            target_candidates["candidates"],
            target_candidates["positions"],
            top_k=args.top_k,
        )
        _write_csv(out / "candidate_cosine_top_matches.csv", top_matches)
        torch.save(grid, out / "candidate_cosine_grid.pt")
        if not args.no_heatmap:
            save_cosine_heatmap(
                grid,
                target_candidates["positions"],
                out / "candidate_cosine_heatmap.png",
                title="Reference direction vs target candidates",
            )
        summary["target_candidates"] = {
            "path": target_candidates["path"],
            "model": target_candidates["model"],
            "context_type": target_candidates["context_type"],
            "effective_candidate_layers": target_candidates["effective_candidate_layers"],
            "positions": target_candidates["positions"],
            "top_matches": top_matches,
            "cosine_grid_shape": list(grid.shape),
        }

    _write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved: {out / 'summary.json'}")
    if args.target_candidates is not None:
        print(f"Saved: {out / 'candidate_cosine_top_matches.csv'}")
        print(f"Saved: {out / 'candidate_cosine_grid.pt'}")
        if not args.no_heatmap:
            print(f"Saved: {out / 'candidate_cosine_heatmap.png'}")


if __name__ == "__main__":
    main()
