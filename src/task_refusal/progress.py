from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def log_event(event: str, **fields: Any) -> None:
    payload = {
        "time": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "event": event,
        **fields,
    }
    print(json.dumps(payload, ensure_ascii=False, default=str), flush=True)


def gpu_memory() -> dict[str, str | int]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"gpu": "unavailable"}
        device = torch.cuda.current_device()
        gib = 1024**3
        return {
            "gpu": torch.cuda.get_device_name(device),
            "gpu_index": device,
            "allocated_gib": f"{torch.cuda.memory_allocated(device) / gib:.2f}",
            "reserved_gib": f"{torch.cuda.memory_reserved(device) / gib:.2f}",
            "peak_allocated_gib": f"{torch.cuda.max_memory_allocated(device) / gib:.2f}",
        }
    except Exception as exc:  # progress reporting must never stop an experiment
        return {"gpu": "error", "gpu_error": repr(exc)}


@dataclass
class ProgressTracker:
    stage: str
    total: int
    context: dict[str, Any] = field(default_factory=dict)
    completed: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def advance(self, *, item: str | None = None, checkpoint: str | None = None) -> None:
        self.completed += 1
        elapsed = time.monotonic() - self.started_at
        rate = self.completed / elapsed if elapsed > 0 else 0.0
        remaining = self.total - self.completed
        eta = remaining / rate if rate > 0 else None
        log_event(
            "progress",
            stage=self.stage,
            completed=self.completed,
            total=self.total,
            percent=round(100.0 * self.completed / max(self.total, 1), 2),
            elapsed=_duration(elapsed),
            eta=_duration(eta),
            item=item,
            checkpoint=checkpoint,
            **self.context,
            **gpu_memory(),
        )
