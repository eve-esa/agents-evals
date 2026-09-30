"""Fetch pinned upstream benchmarks into third_party/ and build their tool environments.

Revisions come from ``upstream.lock.toml``. Nothing fetched here is committed or
imported by ``agentic_evals``: tool servers run in their own venvs and are
reached over MCP.

Examples::

    uv run python scripts/fetch_upstream.py earth-agent                 # code + tools venv
    uv run python scripts/fetch_upstream.py earth-agent --questions 1-5 # + rasters for 5 tasks
    uv run python scripts/fetch_upstream.py earth-agent --modality spectrum
    uv run python scripts/fetch_upstream.py eo-gym                      # code, test set, indexes
    uv run python scripts/fetch_upstream.py eo-gym --with-data          # + ~47 GB of data
    uv run python scripts/fetch_upstream.py geobenchx --build-venv      # code, tools venv, judge prompt
    uv run python scripts/fetch_upstream.py geobenchx --with-data       # + the Drive dataset archive
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentic_evals.paths import (  # noqa: E402
    EARTH_AGENT_ROOT,
    EARTH_AGENT_VENV,
    EO_GYM_ROOT,
    EO_GYM_TOOLS_JSON,
    EO_GYM_VENV,
    GEOBENCHX_JUDGE_PROMPT,
    GEOBENCHX_ROOT,
    GEOBENCHX_VENV,
    venv_python,
)

LOCK = tomllib.loads((REPO_ROOT / "upstream.lock.toml").read_text())

MODALITY_QUESTIONS = {
    "spectrum": range(1, 101),
    "products": range(101, 189),
    "rgb": range(189, 249),
    "all": range(1, 249),
}


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True, env={**os.environ, **(env or {})})


def parse_questions(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def fetch_earth_agent_code() -> None:
    spec = LOCK["earth_agent"]
    legacy = REPO_ROOT / "Earth-Agent"
    if not EARTH_AGENT_ROOT.exists() and legacy.is_dir() and (legacy / ".git").exists():
        EARTH_AGENT_ROOT.parent.mkdir(parents=True, exist_ok=True)
        print(f"Moving existing clone {legacy} -> {EARTH_AGENT_ROOT}")
        shutil.move(str(legacy), EARTH_AGENT_ROOT)
    if not EARTH_AGENT_ROOT.exists():
        EARTH_AGENT_ROOT.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", spec["repo"], str(EARTH_AGENT_ROOT)])
    run(["git", "fetch", "--quiet", "origin"], cwd=EARTH_AGENT_ROOT)
    run(["git", "checkout", "--quiet", spec["rev"]], cwd=EARTH_AGENT_ROOT)


def build_earth_agent_venv() -> None:
    env_spec = LOCK["earth_agent"]["tools_env"]
    gdal_config = shutil.which("gdal-config")
    if gdal_config is None:
        sys.exit(
            "gdal-config not found. The Earth-Agent kits import osgeo.gdal, which is built "
            "against a system GDAL (macOS: `brew install gdal`; Debian/Ubuntu: "
            "`apt-get install gdal-bin libgdal-dev`)."
        )
    gdal_version = subprocess.check_output([gdal_config, "--version"], text=True).strip()
    python = venv_python(EARTH_AGENT_VENV)
    if not python.exists():
        run(["uv", "venv", "--python", env_spec["python"], str(EARTH_AGENT_VENV)])
    run(["uv", "pip", "install", "--python", str(python), *env_spec["packages"], "setuptools", "wheel"])
    # GDAL's setup.py passes -std=c++11 to its C extension too; Apple clang rejects
    # that for C sources unless the C++ driver compiles them.
    run(
        [
            "uv", "pip", "install", "--python", str(python), "--no-build-isolation",
            f"gdal=={gdal_version}",
        ],
        env={"CC": "c++"} if sys.platform == "darwin" else None,
    )
    run([str(python), "-c", "from osgeo import gdal; import fastmcp, rasterio; print('gdal', gdal.__version__)"])


def fetch_earth_bench_data(questions: list[int]) -> None:
    from huggingface_hub import snapshot_download

    spec = LOCK["earth_bench_data"]
    target = EARTH_AGENT_ROOT / "benchmark" / "data"
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=spec["repo_id"],
        repo_type=spec["repo_type"],
        revision=spec["revision"],
        local_dir=target,
        allow_patterns=[f"question{q}/*" for q in questions],
    )
    print(f"Rasters for {len(questions)} tasks in {target}")


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
        # EO-Gym's own uv project (torch, rasterio, fastapi, ...); used to run its
        # /execute tool server. It needs Python >= 3.12 and numpy 2.
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


def fetch_geobenchx_code() -> None:
    spec = LOCK["geobenchx"]
    if not GEOBENCHX_ROOT.exists():
        GEOBENCHX_ROOT.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", spec["repo"], str(GEOBENCHX_ROOT)])
    run(["git", "fetch", "--quiet", "origin"], cwd=GEOBENCHX_ROOT)
    run(["git", "checkout", "--quiet", spec["rev"]], cwd=GEOBENCHX_ROOT)


def build_geobenchx_venv() -> None:
    env_spec = LOCK["geobenchx"]["tools_env"]
    gdal_config = shutil.which("gdal-config")
    if gdal_config is None:
        sys.exit(
            "gdal-config not found. GeoBenchX tools import osgeo.gdal, which is built "
            "against a system GDAL (macOS: `brew install gdal`; Debian/Ubuntu: "
            "`apt-get install gdal-bin libgdal-dev`)."
        )
    gdal_version = subprocess.check_output([gdal_config, "--version"], text=True).strip()
    python = venv_python(GEOBENCHX_VENV)
    if not python.exists():
        run(["uv", "venv", "--python", env_spec["python"], str(GEOBENCHX_VENV)])
    run(["uv", "pip", "install", "--python", str(python), *env_spec["packages"], "setuptools", "wheel"])
    # Same constraint as the Earth-Agent venv: GDAL's setup.py passes -std=c++11
    # to its C extension, which Apple clang rejects unless the C++ driver compiles it.
    run(
        [
            "uv", "pip", "install", "--python", str(python), "--no-build-isolation",
            f"gdal=={gdal_version}",
        ],
        env={"CC": "c++"} if sys.platform == "darwin" else None,
    )
    run(
        [
            str(python), "-c",
            "from osgeo import gdal; import geopandas, rasterio, mcp; print('gdal', gdal.__version__)",
        ]
    )
    run(
        [str(python), str(REPO_ROOT / "scripts" / "export_geobenchx_judge_prompt.py"), str(GEOBENCHX_JUDGE_PROMPT)],
        env={"GEOBENCHX_ROOT": str(GEOBENCHX_ROOT), "MPLBACKEND": "Agg"},
    )


def _find_dir(root: Path, name: str) -> Path | None:
    target = name.lower()
    for path in root.rglob("*"):
        if path.is_dir() and path.name.lower() == target:
            return path
    return None


def fetch_geobenchx_data() -> None:
    """Download the Drive archive linked from upstream ``data/README.md`` and unpack it.

    The archive is not in the git repo. It contains ``StatData`` and ``Geodata``;
    their paths are written to ``data_paths.json`` for the MCP server.
    """
    spec = LOCK["geobenchx"]
    dest = GEOBENCHX_ROOT / "geobenchx_data.zip"
    if not dest.exists():
        # Current gdown takes a URL or a bare id as a positional argument.
        # The old ``--id`` flag was removed, so a Drive URL works on both.
        run(
            ["uvx", "gdown", f"https://drive.google.com/uc?id={spec['drive_file_id']}", "-O", str(dest)],
        )
    bundle = GEOBENCHX_ROOT / "data_bundle"
    if not bundle.exists():
        with zipfile.ZipFile(dest) as archive:
            bundle.mkdir(parents=True)
            archive.extractall(bundle)
    stat = _find_dir(bundle, "StatData")
    geo = _find_dir(bundle, "Geodata")
    if stat is None or geo is None:
        sys.exit(f"Unpacked {dest} into {bundle}, but StatData and Geodata were not both found.")
    pointer = GEOBENCHX_ROOT / "data_paths.json"
    pointer.write_text(json.dumps({"stat": str(stat.resolve()), "geo": str(geo.resolve())}, indent=2) + "\n")
    print(f"StatData: {stat}")
    print(f"Geodata: {geo}")


def fetch_geobenchx(build_venv: bool, with_data: bool) -> None:
    fetch_geobenchx_code()
    if build_venv:
        build_geobenchx_venv()
    elif venv_python(GEOBENCHX_VENV).exists():
        run(
            [str(venv_python(GEOBENCHX_VENV)), str(REPO_ROOT / "scripts" / "export_geobenchx_judge_prompt.py"), str(GEOBENCHX_JUDGE_PROMPT)],
            env={"GEOBENCHX_ROOT": str(GEOBENCHX_ROOT), "MPLBACKEND": "Agg"},
        )
    if with_data:
        fetch_geobenchx_data()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="target", required=True)

    ea = sub.add_parser("earth-agent", help="Earth-Agent code, tools venv, and Earth-Bench rasters")
    ea.add_argument("--questions", help="Task numbers to fetch rasters for, e.g. '1-5,101'")
    ea.add_argument("--modality", choices=sorted(MODALITY_QUESTIONS), help="Fetch rasters for a whole modality")
    ea.add_argument("--skip-venv", action="store_true", help="Do not build the tools venv")

    eg = sub.add_parser("eo-gym", help="EO-Gym code, test set and indexes")
    eg.add_argument("--with-data", action="store_true", help="Also fetch EO_GYM_DATA.zip and image_cache.zip (~47 GB)")
    eg.add_argument("--build-venv", action="store_true", help="Build the EO-Gym tool-server venv")

    gx = sub.add_parser("geobenchx", help="GeoBenchX code, tools venv, and the Drive dataset archive")
    gx.add_argument("--build-venv", action="store_true", help="Build the GeoBenchX tools venv and export the judge prompt")
    gx.add_argument("--with-data", action="store_true", help="Download and unpack the dataset archive from Google Drive")

    args = parser.parse_args()
    if args.target == "earth-agent":
        fetch_earth_agent_code()
        if not args.skip_venv:
            build_earth_agent_venv()
        questions: list[int] = []
        if args.modality:
            questions.extend(MODALITY_QUESTIONS[args.modality])
        if args.questions:
            questions.extend(parse_questions(args.questions))
        if questions:
            fetch_earth_bench_data(sorted(set(questions)))
    elif args.target == "eo-gym":
        fetch_eo_gym(args.with_data, args.build_venv)
    else:
        fetch_geobenchx(args.build_venv, args.with_data)


if __name__ == "__main__":
    main()
