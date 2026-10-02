"""Fetch the pinned EO-Gym checkout into third_party/ and build its tool environment.

Earth-Bench and GeoBenchX are part of this package. Their bulk files are
downloaded by ``scripts/setup_benchmark.py``. This module remains for EO-Gym,
whose tool server is still the upstream project.

    uv run python scripts/fetch_upstream.py eo-gym
    uv run python scripts/fetch_upstream.py eo-gym --with-data
    uv run python scripts/fetch_upstream.py eo-gym --build-venv
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentic_evals.paths import EO_GYM_ROOT, EO_GYM_TOOLS_JSON, EO_GYM_VENV, venv_python  # noqa: E402

LOCK = tomllib.loads((REPO_ROOT / "upstream.lock.toml").read_text())


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True, env={**os.environ, **(env or {})})


def fetch_eo_gym(with_data: bool, build_venv: bool) -> None:
    from huggingface_hub import snapshot_download

    spec = LOCK["eo_gym"]
    patterns = list(spec["code_patterns"]) + (list(spec["data_patterns"]) if with_data else [])
    EO_GYM_ROOT.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=spec["repo_id"],
        repo_type=spec["repo_type"],
        revision=spec["revision"],
        local_dir=EO_GYM_ROOT,
        allow_patterns=patterns,
    )
    if build_venv:
        # EO-Gym's own uv project (torch, rasterio, fastapi, ...). It needs
        # Python >= 3.12 and numpy 2, which the harness environment does not share.
        if not venv_python(EO_GYM_VENV).exists():
            run(["uv", "venv", "--python", "3.12", str(EO_GYM_VENV)])
        run(["uv", "pip", "install", "--python", str(venv_python(EO_GYM_VENV)), "-e", str(EO_GYM_ROOT)])
    if venv_python(EO_GYM_VENV).exists():
        run(
            [
                str(venv_python(EO_GYM_VENV)),
                str(REPO_ROOT / "scripts" / "export_eo_gym_tools.py"),
                str(EO_GYM_TOOLS_JSON),
            ],
            env={"EO_GYM_ROOT": str(EO_GYM_ROOT)},
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--with-data", action="store_true", help="Also fetch EO_GYM_DATA.zip and image_cache.zip (~47 GB)")
    parser.add_argument("--build-venv", action="store_true", help="Build the EO-Gym tool-server venv")
    # Kept so `fetch_upstream.py eo-gym` still works; setup_benchmark.py is the command to use.
    parser.add_argument("target", nargs="?", default="eo-gym", choices=["eo-gym"])
    args = parser.parse_args()
    del args.target
    fetch_eo_gym(args.with_data, args.build_venv)


if __name__ == "__main__":
    main()
