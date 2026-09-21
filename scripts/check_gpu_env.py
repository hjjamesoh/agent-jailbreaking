from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys


def command_output(command: list[str]) -> dict:
    if shutil.which(command[0]) is None:
        return {"available": False, "error": f"{command[0]} not found"}
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "available": True,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def main() -> None:
    fatal_error: str | None = None
    report = {
        "python": sys.version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hf_token_present": bool(
            os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        ),
        "tmux": command_output(["tmux", "-V"]),
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader",
            ]
        ),
    }
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        report["torch"] = {
            "version": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_version": torch.version.cuda,
            "devices": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "compute_capability": ".".join(
                        str(value) for value in torch.cuda.get_device_capability(index)
                    ),
                    "memory_gib": round(
                        torch.cuda.get_device_properties(index).total_memory / 1024**3, 2
                    ),
                }
                for index in range(torch.cuda.device_count())
            ],
        }
        if not cuda_available:
            fatal_error = "PyTorch cannot access CUDA. Install a Blackwell-compatible CUDA build."
    except Exception as exc:
        report["torch"] = {"error": repr(exc)}
        fatal_error = "PyTorch import failed. Install the CUDA-enabled PyTorch build first."
    if fatal_error is not None:
        report["fatal_error"] = fatal_error
    print(json.dumps(report, indent=2), flush=True)
    if fatal_error is not None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
