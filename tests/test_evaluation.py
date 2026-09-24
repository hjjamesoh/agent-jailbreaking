from __future__ import annotations

import csv
from pathlib import Path

from task_refusal.evaluation import write_matrix_csv


def test_write_matrix_csv_supports_different_row_and_column_labels(
    tmp_path: Path,
) -> None:
    matrix = {
        "global": {"System_RCE": 1.0, "safe": 2.0},
        "System_RCE": {"System_RCE": 3.0, "safe": 4.0},
    }
    output = tmp_path / "rectangular.csv"

    write_matrix_csv(output, matrix)

    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows == [
        ["source", "System_RCE", "safe"],
        ["global", "1.0", "2.0"],
        ["System_RCE", "3.0", "4.0"],
    ]
