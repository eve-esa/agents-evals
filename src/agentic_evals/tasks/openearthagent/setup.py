"""Download OpenEarthAgent test images and the gpkg cache, and clone the tool server.

The test split and the prompts are already in this package. Images and
``gpkgs.zip`` come from the Hugging Face dataset pinned in
``upstream.lock.toml`` (about 0.74 GB of images and 0.66 GB of cached
GeoPackages). ``--tasks`` limits which images are fetched. The gpkg archive
is one file, so it is still downloaded whole.

The GitHub repository has no license, so the tool server is not copied into
this package. This module clones the pinned revision into
``data/openearthagent/upstream`` (gitignored). ``--skip-data`` skips the
Hugging Face download and still clones that revision.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

from agentic_evals.paths import OPENEARTHAGENT_DATA, OPENEARTHAGENT_UPSTREAM, REPO_ROOT
from agentic_evals.tasks.openearthagent.dataset import load_records


def task_ids(spec: str) -> list[str]:
    """``'0-2,5'`` to ``['0', '1', '2', '5']``."""
    chosen: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            chosen.extend(str(number) for number in range(int(start), int(end) + 1))
        else:
            chosen.append(part)
    return chosen


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def clone_upstream(dest: Path, rev: str) -> None:
    """Clone ``rev`` into ``dest`` if that checkout is not already there."""
    if (dest / ".git").is_dir():
        current = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
        if current == rev:
            print(f"OpenEarthAgent tool server already at {rev}.", flush=True)
            return
        _run(["git", "-C", str(dest), "fetch", "--depth", "1", "origin", rev])
        _run(["git", "-C", str(dest), "checkout", rev])
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--filter=blob:none", "https://github.com/mbzuai-oryx/OpenEarthAgent.git", str(dest)])
    _run(["git", "-C", str(dest), "checkout", rev])


def _image_patterns(tasks: str | None) -> list[str]:
    if not tasks:
        return ["test/*", "gpkgs.zip"]
    wanted = set(task_ids(tasks))
    patterns = ["gpkgs.zip"]
    for record in load_records():
        if str(record.get("idx")) not in wanted:
            continue
        for name in record.get("images") or []:
            patterns.append(f"test/{name}")
    return patterns


def _unpack_gpkgs(root: Path) -> Path:
    archive = root / "gpkgs.zip"
    dest = root / "gpkgs"
    if dest.is_dir() and any(dest.glob("*.gpkg")):
        return dest
    if not archive.is_file():
        sys.exit(f"{archive} was not downloaded.")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(dest)
    nested = [path for path in dest.iterdir() if path.is_dir()]
    gpks = list(dest.glob("*.gpkg"))
    if not gpks and len(nested) == 1:
        dest = nested[0]
    pointer = root / "data_paths.json"
    pointer.write_text(json.dumps({"gpkgs": str(dest.resolve())}, indent=2) + "\n")
    print(f"GeoPackages: {dest}", flush=True)
    return dest


def install(args) -> None:
    lock = tomllib.loads((REPO_ROOT / "upstream.lock.toml").read_text())
    spec = lock["openearthagent"]
    clone_upstream(OPENEARTHAGENT_UPSTREAM, spec["rev"])
    if args.skip_data:
        print("OpenEarthAgent images and gpkg cache skipped (--skip-data).", flush=True)
        return
    from huggingface_hub import snapshot_download

    OPENEARTHAGENT_DATA.mkdir(parents=True, exist_ok=True)
    patterns = _image_patterns(getattr(args, "tasks", None))
    print(f"Downloading OpenEarthAgent files ({len(patterns)} pattern(s)).", flush=True)
    snapshot_download(
        repo_id=spec["repo_id"],
        repo_type=spec["repo_type"],
        revision=spec["revision"],
        local_dir=OPENEARTHAGENT_DATA,
        allow_patterns=patterns,
    )
    _unpack_gpkgs(OPENEARTHAGENT_DATA)
    print(
        "The tool server is not started. Launch it from "
        f"{OPENEARTHAGENT_UPSTREAM} on a GPU host, then point "
        "OPENEARTHAGENT_CONTROLLER_URL at its controller.",
        flush=True,
    )
