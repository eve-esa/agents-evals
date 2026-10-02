# GeoBenchX tool port

Source: [Solirinai/GeoBenchX](https://github.com/Solirinai/GeoBenchX) at `0edf6109ea5dca3e70fdfe081d39c9ccd0f832fb` (MIT, copyright Varvara Krechetova 2025; see `assets/LICENSE-GeoBenchX`).

`tools/geobenchx_tools.py` is `geobenchx/tools.py` from that revision. `assets/` holds `tasks_and_reference_solutions.json`, `evaluator_tuning_set.json`, and the judge prompt exported once from `geobenchx.evaluation.RESULT_CHECKING_PROMPT` at that revision. A verbatim copy of the modules our solution-code check executes is in `tests/fixtures/geobenchx/upstream`.

## Modifications

- Renamed `geobenchx/tools.py` to `geobenchx_tools.py` so it can live in the `tools/` package.
- `from geobenchx.utils import get_dataframe_info` now imports `geobenchx_utils.py`, which is only that function. `get_solution_code` stays in the fixture copy, because our metrics have their own renderer.
- Dropped `python-dotenv`. `STATDATAPATH` and `GEODATAPATH` still come from the environment; the MCP server sets them from `data/geobenchx`.
- `osgeo` is imported inside `generate_contours_display` only. The other 23 tools run without a system GDAL. That one tool still needs it (`brew install gdal` or `apt-get install gdal-bin libgdal-dev`).
- `tools/mcp_server.py` replaces `scripts/geobenchx_mcp.py` and imports the ported tools. It runs as `python -m agentic_evals.tasks.geobenchx.tools.mcp_server`.

Scores from these tools are for this port and the library versions in the `geobenchx` extra.
