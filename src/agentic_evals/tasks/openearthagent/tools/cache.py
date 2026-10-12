"""Replay of the OpenEarthAgent gpkg cache.

``CacheManager.get_response`` (``tool_workers/cache_manager/base_cache_manager.py``)
serves ``GetAreaBoundary``, ``AddPoisLayer``, ``ComputeDistance``,
``AddIndexLayer``, and ``ComputeIndexChange`` from ``{sample_num}.gpkg``
instead of calling OSM or Earth Engine. This is that lookup, reimplemented
here because the upstream file is not vendored (the GitHub repo has no
license; see ``PORTING.md``).

A matching column is a hit only after the QGIS raster copy succeeds.
A failed copy, including ``qgis.core`` missing, is a miss: upstream's
``get_response`` raises and returns ``error_code`` 1, and the inferencer
then calls the Earth Engine worker. Boundary, POI, and distance hits
need geopandas only.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from agentic_evals.tasks.openearthagent.metrics import CACHE_TOOLS
from agentic_evals.tasks.openearthagent.tools.session import ToolSession

_MISS = "Failed to get cached results for {tool}"


def normalize_name(value: str) -> str:
    """``CacheManager.normalize_name``: lowercase, drop non-alphanumerics."""
    return re.sub(r"[\W_]+", "", str(value).lower())


def _miss(tool: str) -> dict[str, Any]:
    return {"text": _MISS.format(tool=tool), "error_code": 1}


def _boundary_paths(cache_dir: Path, sample_num: str) -> list[Path]:
    paths = list(cache_dir.glob(f"{sample_num}.gpkg"))
    paths.extend(cache_dir.glob(f"{sample_num}_*.gpkg"))
    return paths


def _copy_raster(source_gpkg: str, source_layer: str, dest_gpkg: str, dest_layer: str) -> None:
    """QGIS raster copy used for index layers. Raises when QGIS is absent or the layer is invalid."""
    from qgis.core import QgsRasterFileWriter, QgsRasterLayer, QgsRasterPipe

    layer = QgsRasterLayer(f"{source_gpkg}|layername={source_layer}", source_layer)
    if not layer.isValid():
        raise RuntimeError(f"Invalid cached raster layer: {source_layer}")
    pipe = QgsRasterPipe()
    pipe.set(layer.dataProvider().clone())
    writer = QgsRasterFileWriter(f"{dest_gpkg}|layername={dest_layer}")
    writer.writeRaster(pipe, layer.width(), layer.height(), layer.extent(), layer.crs())


def cache_response(
    session: ToolSession,
    tool: str,
    params: dict[str, Any],
    *,
    cache_dir: Path,
    save_dir: Path,
) -> dict[str, Any] | None:
    """A cache hit or miss, or ``None`` when ``tool`` is not cache-served.

    ``None`` means the tool is not cache-served. A dict with ``error_code``
    1 is a miss. The caller uses a hit and sends every other call to the
    live worker, matching ``BaseToolInferencer``.
    """
    if tool not in CACHE_TOOLS:
        return None
    try:
        import geopandas as gpd
    except ImportError:
        return _miss(tool)
    try:
        return _lookup(session, tool, params, cache_dir=cache_dir, save_dir=save_dir, gpd=gpd)
    except Exception:
        return _miss(tool)


def _lookup(session: ToolSession, tool: str, params: dict[str, Any], *, cache_dir: Path, save_dir: Path, gpd) -> dict[str, Any]:
    if tool == "GetAreaBoundary":
        return _boundary(session, params, cache_dir=cache_dir, save_dir=save_dir, gpd=gpd)
    if tool == "AddPoisLayer":
        return _pois(session, params, gpd=gpd)
    if tool == "ComputeDistance":
        return _distance(session, params, gpd=gpd)
    if tool == "AddIndexLayer":
        return _index(session, params, gpd=gpd, change=False)
    if tool == "ComputeIndexChange":
        return _index(session, params, gpd=gpd, change=True)
    return _miss(tool)


def _boundary(session: ToolSession, params: dict[str, Any], *, cache_dir: Path, save_dir: Path, gpd) -> dict[str, Any]:
    session.cached_layers = {}
    session.cached_gpkg = None
    area = params["area"]
    buffer_m = params.get("buffer_m")
    expected = f"{area}_buffer{buffer_m}m" if buffer_m and buffer_m > 0 else f"{area}"
    for path in _boundary_paths(cache_dir, str(session.sample_num)):
        try:
            frame = gpd.read_file(path, layer="area_boundary")
        except Exception:
            continue
        if frame.empty or "name" not in frame.columns:
            continue
        if normalize_name(frame.iloc[0]["name"]) != normalize_name(expected):
            continue
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = f"aoi_{datetime.now():%Y%m%d_%H%M%S}.gpkg"
        dest = save_dir / filename
        frame.to_file(dest, layer="area_boundary", driver="GPKG")
        session.cached_gpkg = str(path)
        return {"text": f"Saved boundary to {filename}", "gpkg": str(dest), "error_code": 0}
    return _miss("GetAreaBoundary")


def _pois(session: ToolSession, params: dict[str, Any], *, gpd) -> dict[str, Any]:
    if not session.cached_gpkg:
        return _miss("AddPoisLayer")
    query = params["query"]
    layer_name = params["layer_name"]
    dest = params["gpkg"]
    frame = gpd.read_file(session.cached_gpkg, layer="area_boundary")
    key = json.dumps(query)
    if key not in frame.columns:
        return _miss("AddPoisLayer")
    cached_layer = frame[key].iloc[0]
    points = gpd.read_file(session.cached_gpkg, layer=cached_layer)
    points.to_file(dest, layer=layer_name, driver="GPKG")
    session.cached_layers[layer_name] = cached_layer
    text = f"Saved {len(points)} POIs to layer '{layer_name}' in {os.path.basename(dest)}"
    return {"text": text, "error_code": 0}


def _distance(session: ToolSession, params: dict[str, Any], *, gpd) -> dict[str, Any]:
    src = params["src_layer"]
    tar = params["tar_layer"]
    top = params.get("top")
    if not (session.cached_gpkg and src in session.cached_layers and tar in session.cached_layers):
        return _miss("ComputeDistance")
    src_cached = session.cached_layers[src]
    tar_cached = session.cached_layers[tar]
    column = f"{src_cached}_to_{tar_cached}_distances"
    boundary = gpd.read_file(session.cached_gpkg, layer="area_boundary")
    if column not in boundary.columns:
        return _miss("ComputeDistance")
    cached_top = json.loads(boundary[column].iloc[0])["top"]
    if cached_top != top:
        return _miss("ComputeDistance")
    layer_name = f"{src}_to_{tar}_distances"
    frame = gpd.read_file(session.cached_gpkg, layer=column)
    frame.to_file(params["gpkg"], layer=layer_name, driver="GPKG")
    lines = [f"Distances (in meters) saved to line layer: '{layer_name}': "]
    distances: list[float] = []
    travel: list[float] = []
    has_travel = "travel_time_s" in frame.columns and not frame["travel_time_s"].isna().all()
    for _, row in frame.iterrows():
        line = f"{row[src]} , {row[tar]}, distance={row['distance_m']:.2f} m"
        distances.append(round(float(row["distance_m"]), 2))
        if has_travel:
            line += f", travel_time={row['travel_time_s']:.1f} s"
            travel.append(round(float(row["travel_time_s"]), 1))
        lines.append(line)
    lines.append(f"distances = {distances}")
    if travel:
        lines.append(f"travel_times = {travel}")
    session.cached_layers[layer_name] = column
    return {"text": "\n".join(lines), "error_code": 0}


def _index(session: ToolSession, params: dict[str, Any], *, gpd, change: bool) -> dict[str, Any]:
    tool = "ComputeIndexChange" if change else "AddIndexLayer"
    if not session.cached_gpkg:
        return _miss(tool)
    index_type = params["index_type"]
    dest = params["gpkg"]
    boundary = gpd.read_file(session.cached_gpkg, layer="area_boundary")
    if change:
        left = params["layer1_name"]
        right = params["layer2_name"]
        if left not in session.cached_layers or right not in session.cached_layers:
            return _miss(tool)
        column = f"{index_type}-{session.cached_layers[left]}-{session.cached_layers[right]}"
        written = params.get("diff_layer_name", f"{index_type}_Change")
        message = f"delta-{index_type} layer saved to {os.path.basename(dest)} as '{written}'\n {index_type}"
    else:
        year = params["year"]
        month = params.get("month")
        column = f"{index_type}_{year}_{month}" if month is not None else f"{index_type}_{year}"
        written = params["layer_name"]
        message = f"{index_type} raster for {year} saved to {os.path.basename(dest)} as layer '{written}'\n{index_type} "
    if column not in boundary.columns:
        return _miss(tool)
    meta = json.loads(boundary[column].iloc[0])
    _copy_raster(session.cached_gpkg, meta["layer_name"], dest, written)
    session.cached_layers[written] = meta["layer_name"]
    return {"text": message + meta["layer_summary"], "error_code": 0}
