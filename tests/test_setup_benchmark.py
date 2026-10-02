"""setup_benchmark installs one named benchmark and leaves the others alone."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import setup_benchmark as setup  # noqa: E402


def test_earth_bench_defaults_to_every_task():
    args = setup.build_parser().parse_args(["earth-bench"])
    assert setup.earth_bench_questions(args) == list(range(1, 249))


def test_modality_limits_earth_bench_rasters():
    args = setup.build_parser().parse_args(["earth-bench", "--modality", "spectrum"])
    assert setup.earth_bench_questions(args) == list(range(1, 101))


def test_skip_data_downloads_no_rasters():
    args = setup.build_parser().parse_args(["geobenchx", "--skip-data"])
    assert setup.canonical_benchmark(args.benchmark) == "geobenchx"
    assert setup.earth_bench_questions(args) == []


def test_earth_agent_is_an_alias_for_earth_bench():
    args = setup.build_parser().parse_args(["earth-agent", "--questions", "4"])
    assert setup.canonical_benchmark(args.benchmark) == "earth-bench"
    assert setup.earth_bench_questions(args) == [4]


def test_skip_data_rejects_a_raster_subset():
    try:
        setup.main(["earth-bench", "--skip-data", "--modality", "spectrum"])
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("expected SystemExit")


def test_modality_is_rejected_for_other_benchmarks():
    try:
        setup.main(["eo-gym", "--modality", "spectrum"])
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("expected SystemExit")


def test_unpack_bare_zip_into_its_directory(tmp_path: Path):
    archive = tmp_path / "EO_GYM_DATA.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("tile.png", b"png")
    setup.unpack_eo_gym_archives(tmp_path)
    assert (tmp_path / "EO_GYM_DATA" / "tile.png").read_bytes() == b"png"


def test_unpack_prefixed_zip_without_nesting(tmp_path: Path):
    archive = tmp_path / "image_cache.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("image_cache/a.txt", b"a")
    setup.unpack_eo_gym_archives(tmp_path)
    assert (tmp_path / "image_cache" / "a.txt").read_bytes() == b"a"
    assert not (tmp_path / "image_cache" / "image_cache").exists()
