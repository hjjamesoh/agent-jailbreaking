import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from refusal_repro.analysis import cosine_similarity, top_cosine_matches


def test_cosine_similarity_same_and_opposite():
    a = torch.tensor([1.0, 0.0, 0.0])
    assert cosine_similarity(a, a) == 1.0
    assert cosine_similarity(a, -a) == -1.0


def test_top_cosine_matches_uses_absolute_similarity():
    reference = torch.tensor([1.0, 0.0])
    candidates = torch.tensor([
        [[0.0, 1.0], [-1.0, 0.0]],
        [[1.0, 0.0], [0.5, 0.5]],
    ])
    rows, grid = top_cosine_matches(reference, candidates, positions=[-1, -2], top_k=2)
    assert list(grid.shape) == [2, 2]
    assert rows[0]["abs_cosine_similarity"] == 1.0
    assert rows[1]["abs_cosine_similarity"] == 1.0
