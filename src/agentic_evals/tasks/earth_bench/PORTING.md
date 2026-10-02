# Earth-Bench tool port

Source: [opendatalab/Earth-Agent](https://github.com/opendatalab/Earth-Agent) at `ca1dfaef750654331308a434966d307f1728fd90` (MIT, copyright OpenDataLab 2026; see `assets/LICENSE-Earth-Agent`).

The five kits and `utils.py` in `tools/` are that revision. Task files in `assets/` are `benchmark/question.json`, `extracted_tool_calls_GT.json`, and `benchmark/model_results.csv` from the same revision.

## Modifications

- `tools/utils.py`: `osgeo.gdal` replaced with rasterio, including `write_like` for GeoTIFF output. One band stays a 2-D array; several bands come back as height, width, band, matching GDAL `ReadAsArray`. `get_geotransform` returns rasterio's `(c, a, b, f, d, e)`, which is GDAL's geotransform order. Inline `from osgeo import gdal` in `Statistics.py`, `Analysis.py`, and `Inversion.py` (average, difference, hotspot mask, colormap, Gi*, and ATI resampling) now use `read_image`, `write_like`, or `rasterio.warp.reproject`. This drops the system-GDAL requirement.
- `tools/Perception.py`: the lookup CSV is `assets/model_results.csv`, resolved from this file. The bare `except` and the `'Failed to call model'` string are unchanged.
- `tools/__init__.py`: the kits run under this interpreter (`sys.executable`) with `cwd` set to `data/earth_bench`, so `benchmark/data/questionN` in the prompts still resolves. The `LD_LIBRARY_PATH` workaround for `libgdal` is gone.

Everything else in the kits (names, signatures, descriptions) is the upstream source. Scores from these kits are for this port and the library versions in the `earth-bench` extra, not for Earth-Agent's numpy 1.24 pins.

## Check

`tests/test_earth_bench_tools.py` lists the 104 tools when the extra is installed. On the eight local GeoTIFFs under the old Earth-Agent checkout (`question1`–`question5`), `read_image` sums and shapes from this rasterio port matched the GDAL `utils.py` in `third_party/venvs/earth-agent` exactly, including the uint8 scaling. Products and RGB rasters were not on disk, so those kits were checked by schema only.
