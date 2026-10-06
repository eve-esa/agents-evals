"""Download the GeoBenchX dataset archive.

Task JSON and the 24 tools are already in this package. This module fetches
the Google Drive archive pinned in ``upstream.lock.toml`` and records where
``StatData`` and ``Geodata`` landed.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

from agentic_evals.paths import GEOBENCHX_DATA, REPO_ROOT


def _find_dir(root: Path, name: str) -> Path | None:
    target = name.lower()
    for path in root.rglob("*"):
        if path.is_dir() and path.name.lower() == target:
            return path
    return None


def install(args) -> None:
    if args.skip_data:
        print("GeoBenchX Drive archive skipped (--skip-data). Tool code is already in the package.", flush=True)
        return
    lock = tomllib.loads((REPO_ROOT / "upstream.lock.toml").read_text())
    spec = lock["geobenchx"]
    GEOBENCHX_DATA.mkdir(parents=True, exist_ok=True)
    dest = GEOBENCHX_DATA / "geobenchx_data.zip"
    if not dest.exists():
        subprocess.run(
            ["uvx", "gdown", f"https://drive.google.com/uc?id={spec['drive_file_id']}", "-O", str(dest)],
            check=True,
        )
    bundle = GEOBENCHX_DATA / "data_bundle"
    if not bundle.exists():
        with zipfile.ZipFile(dest) as archive:
            bundle.mkdir(parents=True)
            archive.extractall(bundle)
    stat = _find_dir(bundle, "StatData")
    geo = _find_dir(bundle, "Geodata")
    if stat is None or geo is None:
        sys.exit(f"Unpacked {dest} into {bundle}, but StatData and Geodata were not both found.")
    pointer = GEOBENCHX_DATA / "data_paths.json"
    pointer.write_text(json.dumps({"stat": str(stat.resolve()), "geo": str(geo.resolve())}, indent=2) + "\n")
    print(f"StatData: {stat}", flush=True)
    print(f"Geodata: {geo}", flush=True)
