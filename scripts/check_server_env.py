import json
import os
import shutil
import subprocess
import sys


def _run(command):
    if shutil.which(command[0]) is None:
        return {
            "available": False,
            "command": command,
            "stdout": "",
            "stderr": f"{command[0]} not found",
            "returncode": None,
        }
    result = subprocess.run(command, capture_output=True, text=True)
    return {
        "available": True,
        "command": command,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "returncode": result.returncode,
    }


def main():
    report = {
        "python": sys.version,
        "executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "huggingface_token_present": bool(
            os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        ),
        "nvidia_smi": _run([
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader",
        ]),
    }

    try:
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "visible_device_count": torch.cuda.device_count()
            if torch.cuda.is_available()
            else 0,
            "visible_device_names": [
                torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())
            ] if torch.cuda.is_available() else [],
        }
    except Exception as exc:
        report["torch"] = {"error": repr(exc)}

    try:
        import transformers

        report["transformers"] = {"version": transformers.__version__}
    except Exception as exc:
        report["transformers"] = {"error": repr(exc)}

    try:
        import huggingface_hub

        report["huggingface_hub"] = {"version": huggingface_hub.__version__}
    except Exception as exc:
        report["huggingface_hub"] = {"error": repr(exc)}

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
