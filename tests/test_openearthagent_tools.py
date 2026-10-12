"""Per-sample tool state and the gpkg cache, without a controller."""

from __future__ import annotations

import json

import geopandas as gpd
from shapely.geometry import Point

from agentic_evals.tasks.openearthagent.tools.cache import cache_response, normalize_name
from agentic_evals.tasks.openearthagent.tools.session import ToolSession, observe, prepare_call


def test_normalize_name_drops_punctuation():
    assert normalize_name("Banff National Park, Alberta") == "banffnationalparkalberta"


def test_prepare_call_injects_paths_and_strips_placeholders():
    session = ToolSession(images=["/data/pre.jpg", "/data/post.jpg"], sample_num="5")
    params, error = prepare_call(
        session,
        "ChangeDetection",
        {"text": "buildings", "pre_image": "img_1", "post_image": "img_2"},
    )
    assert error is None
    assert params["pre_image"] == "/data/pre.jpg"
    assert params["post_image"] == "/data/post.jpg"
    assert params["image"] == "/data/pre.jpg"
    assert "img_1" not in params.values()

    missing, error = prepare_call(session, "AddPoisLayer", {"gpkg": "gpkg_1", "query": {}, "layer_name": "h"})
    assert missing is None
    assert error["error_code"] == 1

    session.gpkg = "/tmp/aoi.gpkg"
    params, error = prepare_call(session, "AddPoisLayer", {"gpkg": "gpkg_1", "query": {"amenity": "hospital"}, "layer_name": "h"})
    assert error is None
    assert params["gpkg"] == "/tmp/aoi.gpkg"
    assert params["query"] == {"amenity": "hospital"}


def test_observe_keeps_paths_off_the_text_and_truncates():
    session = ToolSession(images=[], sample_num="1")
    text = observe(session, "GetAreaBoundary", {"text": "Saved boundary to aoi.gpkg", "gpkg": "/tmp/aoi.gpkg", "error_code": 0})
    assert text == "Saved boundary to aoi.gpkg"
    assert session.gpkg == "/tmp/aoi.gpkg"
    long = observe(session, "OCR", {"text": "x" * 4000, "error_code": 0})
    assert long.endswith(" ... ")
    assert len(long) == 3000 + len(" ... ")


def test_boundary_cache_hit_copies_the_layer(tmp_path):
    cache = tmp_path / "cache"
    save = tmp_path / "save"
    cache.mkdir()
    area = "Banff National Park"
    frame = gpd.GeoDataFrame(
        {"name": [f"{area}_buffer3000m"]},
        geometry=[Point(0, 0)],
        crs="EPSG:4326",
    )
    frame.to_file(cache / "0.gpkg", layer="area_boundary", driver="GPKG")
    session = ToolSession(images=[], sample_num="0")
    response = cache_response(
        session,
        "GetAreaBoundary",
        {"area": area, "buffer_m": 3000},
        cache_dir=cache,
        save_dir=save,
    )
    assert response["error_code"] == 0
    assert response["text"].startswith("Saved boundary to aoi_")
    assert session.cached_gpkg.endswith("0.gpkg")
    copied = gpd.read_file(response["gpkg"], layer="area_boundary")
    assert normalize_name(copied.iloc[0]["name"]) == normalize_name(f"{area}_buffer3000m")


def test_boundary_cache_miss_is_an_error(tmp_path):
    session = ToolSession(images=[], sample_num="9")
    response = cache_response(
        session,
        "GetAreaBoundary",
        {"area": "Nowhere", "buffer_m": 10},
        cache_dir=tmp_path,
        save_dir=tmp_path / "out",
    )
    assert response["error_code"] == 1
    assert session.cached_gpkg is None


def test_poi_cache_uses_the_query_column(tmp_path):
    cache = tmp_path / "cache"
    save = tmp_path / "save"
    cache.mkdir()
    save.mkdir()
    query = {"amenity": "fire_station"}
    key = json.dumps(query)
    boundary = gpd.GeoDataFrame(
        {"name": ["Banff_buffer0m"], key: ["fires"]},
        geometry=[Point(0, 0)],
        crs="EPSG:4326",
    )
    # buffer 0 is stored without the suffix. This hit only needs the source file.
    boundary.to_file(cache / "3.gpkg", layer="area_boundary", driver="GPKG")
    points = gpd.GeoDataFrame({"poi": ["a", "b"]}, geometry=[Point(1, 1), Point(2, 2)], crs="EPSG:4326")
    points.to_file(cache / "3.gpkg", layer="fires", driver="GPKG", mode="a")
    dest = save / "aoi.gpkg"
    boundary.to_file(dest, layer="area_boundary", driver="GPKG")
    session = ToolSession(images=[], sample_num="3", gpkg=str(dest), cached_gpkg=str(cache / "3.gpkg"))
    response = cache_response(
        session,
        "AddPoisLayer",
        {"gpkg": str(dest), "query": query, "layer_name": "fire_stations"},
        cache_dir=cache,
        save_dir=save,
    )
    assert response["error_code"] == 0
    assert "Saved 2 POIs" in response["text"]
    assert session.cached_layers["fire_stations"] == "fires"


def test_schemas_match_the_tool_list():
    from agentic_evals.tasks.openearthagent.metrics import TOOLS
    from agentic_evals.tasks.openearthagent.tools.catalog import schemas_for

    names = {schema["name"] for schema in schemas_for()}
    assert names == set(TOOLS) - {"Terminate"}


def test_factory_passes_the_controller(tmp_path):
    from inspect_ai.model import ChatMessageUser
    from inspect_ai.solver import TaskState

    from agentic_evals.tasks.openearthagent.tools import sample_servers_factory

    del tmp_path
    state = TaskState(
        model="test",
        sample_id="0",
        epoch=1,
        input="q",
        messages=[ChatMessageUser(content="q")],
        metadata={"idx": "0", "images": []},
    )
    args = sample_servers_factory("http://127.0.0.1:20001")(state)["openearthagent"]["args"]
    assert args[args.index("--controller-url") + 1] == "http://127.0.0.1:20001"
    assert "--cached-only" not in args


def test_distance_cache_copies_the_ranked_lines(tmp_path):
    cache = tmp_path / "cache"
    save = tmp_path / "save"
    cache.mkdir()
    save.mkdir()
    column = "fires_to_hospitals_distances"
    boundary = gpd.GeoDataFrame(
        {"name": ["Banff"], column: [json.dumps({"top": 2})]},
        geometry=[Point(0, 0)],
        crs="EPSG:4326",
    )
    boundary.to_file(cache / "7.gpkg", layer="area_boundary", driver="GPKG")
    lines = gpd.GeoDataFrame(
        {
            "stations": ["a", "b"],
            "parks": ["c", "d"],
            "distance_m": [10.0, 20.5],
            "travel_time_s": [3.0, 4.0],
        },
        geometry=[Point(0, 0), Point(1, 1)],
        crs="EPSG:4326",
    )
    lines.to_file(cache / "7.gpkg", layer=column, driver="GPKG", mode="a")
    session = ToolSession(
        images=[],
        sample_num="7",
        gpkg=str(save / "aoi.gpkg"),
        cached_gpkg=str(cache / "7.gpkg"),
        cached_layers={"stations": "fires", "parks": "hospitals"},
    )
    response = cache_response(
        session,
        "ComputeDistance",
        {"gpkg": str(save / "aoi.gpkg"), "src_layer": "stations", "tar_layer": "parks", "top": 2},
        cache_dir=cache,
        save_dir=save,
    )
    assert response["error_code"] == 0
    assert "Distances" in response["text"]
    assert "distances = [10.0, 20.5]" in response["text"]
    assert "travel_times = [3.0, 4.0]" in response["text"]
    assert session.cached_layers["stations_to_parks_distances"] == column


def test_index_change_without_qgis_is_a_miss(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    column = "NDVI-early-late"
    meta = json.dumps({"layer_name": "ndvi_delta", "layer_summary": "change 4%"})
    frame = gpd.GeoDataFrame({"name": ["x"], column: [meta]}, geometry=[Point(0, 0)], crs="EPSG:4326")
    frame.to_file(cache / "8.gpkg", layer="area_boundary", driver="GPKG")
    session = ToolSession(
        images=[],
        sample_num="8",
        gpkg=str(cache / "8.gpkg"),
        cached_gpkg=str(cache / "8.gpkg"),
        cached_layers={"before": "early", "after": "late"},
    )
    response = cache_response(
        session,
        "ComputeIndexChange",
        {
            "gpkg": str(tmp_path / "out.gpkg"),
            "index_type": "NDVI",
            "layer1_name": "before",
            "layer2_name": "after",
            "diff_layer_name": "delta",
        },
        cache_dir=cache,
        save_dir=tmp_path / "out",
    )
    assert response["error_code"] == 1
    assert "Failed to get cached results for ComputeIndexChange" in response["text"]
    assert "delta" not in session.cached_layers


def test_index_cache_without_qgis_is_a_miss(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    column = "NDVI_2023"
    meta = json.dumps({"layer_name": "ndvi_cached", "layer_summary": "vegetation 10%"})
    frame = gpd.GeoDataFrame({"name": ["x"], column: [meta]}, geometry=[Point(0, 0)], crs="EPSG:4326")
    frame.to_file(cache / "4.gpkg", layer="area_boundary", driver="GPKG")
    session = ToolSession(images=[], sample_num="4", gpkg=str(cache / "4.gpkg"), cached_gpkg=str(cache / "4.gpkg"))
    response = cache_response(
        session,
        "AddIndexLayer",
        {"gpkg": str(cache / "4.gpkg"), "index_type": "NDVI", "layer_name": "ndvi", "year": 2023},
        cache_dir=cache,
        save_dir=tmp_path / "out",
    )
    assert response["error_code"] == 1
    assert "Failed to get cached results for AddIndexLayer" in response["text"]
    assert "ndvi" not in session.cached_layers
    wrong_year = cache_response(
        session,
        "AddIndexLayer",
        {"gpkg": str(cache / "4.gpkg"), "index_type": "NDVI", "layer_name": "ndvi", "year": 1999},
        cache_dir=cache,
        save_dir=tmp_path / "out",
    )
    assert wrong_year["error_code"] == 1
    assert "Failed to get cached results for AddIndexLayer" in wrong_year["text"]
