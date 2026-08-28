import torch
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from refusal_repro.directions import _position_indices_from_attention_mask


def test_position_indices_left_padding():
    mask = torch.tensor([
        [0, 0, 1, 1, 1],
        [0, 1, 1, 1, 1],
    ])
    got = _position_indices_from_attention_mask(mask, positions=(-1, -2, 0))
    expected = torch.tensor([
        [4, 3, 2],
        [4, 3, 1],
    ])
    assert torch.equal(got, expected)


def test_position_indices_no_padding():
    mask = torch.tensor([
        [1, 1, 1],
        [1, 1, 1],
    ])
    got = _position_indices_from_attention_mask(mask, positions=(-1, -3, 0))
    expected = torch.tensor([
        [2, 0, 0],
        [2, 0, 0],
    ])
    assert torch.equal(got, expected)
