import torch

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
