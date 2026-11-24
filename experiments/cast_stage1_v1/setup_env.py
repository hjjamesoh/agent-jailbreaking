"""Create an isolated environment that can reuse the existing CUDA torch install.

Run with the server's working GPU Python. Only a new environment is written.
The old environment is read via a .pth search path, not modified or re-installed.
"""

import json
from pathlib import Path
import site
import subprocess
import sys
import venv


def main():
    root = Path(__file__).resolve().parents[2]
    target = root / "runs/cast_stage1_v1/env"
    import torch
    import transformers

    if not torch.cuda.is_available():
        raise RuntimeError("Run setup with the existing working CUDA Python environment")
    inherited = [str(Path(p).resolve()) for p in site.getsitepackages() if Path(p).is_dir()]
    identity = {
        "source_python": sys.executable,
        "source_sites": inherited,
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
    }
    manifest = target / "inherited_environment.json"
    if target.exists():
        if not manifest.exists() or json.loads(manifest.read_text()) != identity:
            raise ValueError(
                "Existing stage-1 environment has another origin; no overwrite performed"
            )
    else:
        venv.EnvBuilder(with_pip=True).create(target)
        python = target / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        sites = json.loads(
            subprocess.check_output(
                [str(python), "-c", "import site,json;print(json.dumps(site.getsitepackages()))"],
                text=True,
            )
        )
        destination = Path(sites[0]).resolve()
        if not destination.is_relative_to(target.resolve()):
            raise ValueError("New environment's site-packages escaped the intended directory")
        (destination / "cast_existing_gpu_environment.pth").write_text(
            "\n".join(inherited) + "\n", encoding="utf-8"
        )
        manifest.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    print(f"CREATED_OR_VERIFIED={target}\nInherited GPU packages from {sys.executable}")


if __name__ == "__main__":
    main()
