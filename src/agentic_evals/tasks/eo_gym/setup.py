"""Download EO-Gym imagery and install the copied tool server.

The server source, the test split, and the tool schemas are already in this
package. This module fetches the image archives and the lookup tables from the
Hugging Face dataset pinned in ``upstream.lock.toml``, then installs
``server/`` into ``data/venvs/eo-gym``. ``--skip-data`` skips the archives and
the lookup tables and still installs the virtualenv.
"""

from __future__ import annotations

import subprocess
import tomllib
import zipfile
from pathlib import Path

from agentic_evals.paths import EO_GYM_ROOT, EO_GYM_SERVER, EO_GYM_VENV, REPO_ROOT, venv_python


def unpack_archives(root: Path = EO_GYM_ROOT) -> None:
    """Unpack EO-Gym's data zips when the matching directory is absent.

    Entries already under ``EO_GYM_DATA/`` or ``image_cache/`` extract into
    ``root``. A zip of bare files extracts into that directory.
    """
    for dirname in ("EO_GYM_DATA", "image_cache"):
        archive = root / f"{dirname}.zip"
        dest = root / dirname
        if dest.is_dir() or not archive.is_file():
            continue
        print(f"Unpacking {archive.name}", flush=True)
        with zipfile.ZipFile(archive) as zipped:
            prefix = dirname + "/"
            if any(name.startswith(prefix) or name.startswith("./" + prefix) for name in zipped.namelist()):
                zipped.extractall(root)
            else:
                dest.mkdir(parents=True)
                zipped.extractall(dest)


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def install_server() -> None:
    """Install the copied server into its own virtualenv. It depends on torch."""
    if not venv_python(EO_GYM_VENV).exists():
        _run(["uv", "venv", "--python", "3.12", str(EO_GYM_VENV)])
    _run(["uv", "pip", "install", "--python", str(venv_python(EO_GYM_VENV)), "-e", str(EO_GYM_SERVER)])


def install(args) -> None:
    install_server()
    if args.skip_data:
        print("EO-Gym images and lookup tables skipped (--skip-data).", flush=True)
        return
    from huggingface_hub import snapshot_download

    lock = tomllib.loads((REPO_ROOT / "upstream.lock.toml").read_text())
    spec = lock["eo_gym"]
    EO_GYM_ROOT.mkdir(parents=True, exist_ok=True)
    print("Downloading EO-Gym image archives and lookup tables.", flush=True)
    snapshot_download(
        repo_id=spec["repo_id"],
        repo_type=spec["repo_type"],
        revision=spec["revision"],
        local_dir=EO_GYM_ROOT,
        allow_patterns=list(spec["data_patterns"]),
    )
    unpack_archives()
    python = venv_python(EO_GYM_VENV)
    print(
        "EO-Gym tool server is not started. From "
        f"{EO_GYM_SERVER} run `{python} -m eo_gym.server`.",
        flush=True,
    )
