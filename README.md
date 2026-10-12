# agentic-evals

Evaluation harness for Earth Observation agents, built for the EVE project (Pi School and ESA Φ-lab).

It runs EVE and baseline agents on public EO agent benchmarks under the same conditions, and records both the final answer and the tool-call trajectory. The harness is [Inspect AI](https://inspect.aisi.org.uk/). Agents are LangGraph graphs. Tools are reached over MCP.

Python 3.12 or newer. Dependencies are managed with [uv](https://docs.astral.sh/uv/). Results are Inspect `.eval` logs under `logs/`.

## Benchmarks

| Benchmark | Task name | Code and tasks | Libraries | Bulk data |
| --- | --- | --- | --- | --- |
| Earth-Bench | `agentic_evals/earth_bench` | In `src/agentic_evals/tasks/earth_bench/` | `uv sync --extra earth-bench` | Rasters, up to 16.6 GB, in `data/earth_bench` |
| GeoBenchX | `agentic_evals/geobenchx` | In `src/agentic_evals/tasks/geobenchx/` | `uv sync --extra geobenchx` | Google Drive archive in `data/geobenchx` |
| EO-Gym | `agentic_evals/eo_gym` | In `src/agentic_evals/tasks/eo_gym/` | Its own virtualenv, installed by setup | About 47 GB, plus a tool server you start yourself |
| OpenEarthAgent | `agentic_evals/openearthagent` | In `src/agentic_evals/tasks/openearthagent/` | No extra. The tool controller is external | Test images and the gpkg cache, about 1.4 GB, in `data/openearthagent` |

Earth-Bench and GeoBenchX were ported from pinned upstream revisions. The changes, and the MIT notices, are in each task's `PORTING.md`. Scores from those tools are for this port and the library versions in the extra, not for the original dependency pins.

EO-Gym's tool server is copied into `tasks/eo_gym/server`. It imports torch, so setup installs it in its own virtualenv.

OpenEarthAgent's GitHub repository has no license file. Copying its evaluation prompts into this package, and cloning the tool server for a run, is authorized. The clone stays in `data/openearthagent/upstream` and is not committed. The test split in the package is the Apache-2.0 Hugging Face dataset. See `tasks/openearthagent/PORTING.md`.

`uv sync` alone installs the harness. It does not install Earth-Bench or GeoBenchX libraries, and it does not download any rasters, imagery, or the Drive archive. Install one benchmark at a time: the two extras pull different stacks (rasterio, geopandas, and so on) into the same environment.

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if it is not already on the machine. uv downloads Python 3.12 itself; the system Python can be older.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

The repository is private. `git clone` asks for a GitHub username and a personal access token unless this machine is already authenticated (`gh auth login`, or a credential helper).

```bash
git clone https://github.com/eve-esa/agents-evals.git
cd agents-evals
uv sync --extra earth-bench
cp .env.example .env
```

Put API keys in `.env`. Inspect reads that file from the working directory. `.env` is gitignored.

| Variable | Used for |
| --- | --- |
| `OPENROUTER_API_KEY` | `--model openai-api/openrouter/<provider>/<model>` |
| `MISTRAL_API_KEY`, `MISTRAL_BASE_URL` | `--model openai-api/mistral/<model>` |
| `EVE_BASE_URL`, `EVE_API_KEY` | `--model openai-api/eve/<model>` |
| `WANDB_API_KEY`, `WANDB_ENTITY` | Optional run logging. See [Weights & Biases](#weights--biases). |

For any other OpenAI-compatible host, the provider name in the model id is the env prefix: `openai-api/<name>/<model>` reads `<NAME>_API_KEY` and `<NAME>_BASE_URL`.

### Earth-Bench

```bash
uv sync --extra earth-bench
uv run python scripts/setup_benchmark.py earth-bench --modality spectrum
```

`--modality` is `spectrum` (questions 1–100), `products` (101–188), `rgb` (189–248), or `all`. With no modality and no `--questions`, the script downloads every task raster (16.6 GB). `--questions 1-5` downloads only those task directories.

Task text, gold tool calls, and the perception lookup table are already in the package. You need the rasters only when a tool has to open a GeoTIFF.

### GeoBenchX

```bash
uv sync --extra geobenchx
uv run python scripts/setup_benchmark.py geobenchx
```

This downloads the dataset archive linked from the GeoBenchX README (Google Drive) into `data/geobenchx` and records the `StatData` and `Geodata` paths. Task JSON and the 24 tools are already in the package.

`generate_contours_display` is the one tool that still imports GDAL. The other 23 run without it. If you need that tool: `brew install gdal` on macOS, or `apt-get install gdal-bin libgdal-dev` on Debian or Ubuntu.

### EO-Gym

```bash
uv run python scripts/setup_benchmark.py eo-gym
```

This downloads the image archives and lookup tables into `data/eo_gym` (about 47 GB) and installs the copied tool server into `data/venvs/eo-gym`. Then start the server, with `src/agentic_evals/tasks/eo_gym/server` as the working directory:

```bash
data/venvs/eo-gym/bin/python -m eo_gym.server
```

`--skip-data` installs the virtualenv and leaves the archives alone. EO-Gym questions carry images, so a text-only model cannot read them. Pass `-T vision=false` to record those tasks as not applicable instead of scoring them.

### OpenEarthAgent

```bash
uv run python scripts/setup_benchmark.py openearthagent
```

This downloads the test images and `gpkgs.zip` into `data/openearthagent` (about 1.4 GB) and clones the pinned tool server there. It does not install the authors' Conda environments or download model weights. `--tasks 0-10` downloads images for those dataset ids only. `--skip-data` clones the tool server and skips the Hugging Face download.

Step mode needs none of that. It is teacher-forced JSON, with no tools.

An end-to-end run starts the tool server when `http://127.0.0.1:20001` is not already listing tools, and stops that process when the eval exits. A controller that is already up is left running. Five GIS tools return a stored GeoPackage result when the arguments match; every miss is sent to the live worker. The host must already have:

- Conda environments `tool-server-e1` and `tool-server-e2`
- the model weights named by the upstream launch config, under the clone
- `OPENEARTHAGENT_SERPER_API_KEY` for GoogleSearch
- `OPENEARTHAGENT_GEE_KEY`, a path to an Earth Engine service-account JSON (`client_email` in that file is the AddIndexLayer account)

`OPENEARTHAGENT_CUDA_DEVICES` defaults to `0`. `GetBboxFromGeotiff` and `DisplayOnGeotiff` are not in the upstream launch file, so this start does not register them. A remote `OPENEARTHAGENT_CONTROLLER_URL` is not started by the eval; it has to be up already.

## Run

```bash
uv run inspect eval agentic_evals/earth_bench \
  -T regime=ap -T modality=spectrum \
  --model openai-api/openrouter/openai/gpt-4o-mini \
  --model-base-url https://openrouter.ai/api/v1 \
  -M strict_tools=false

uv run inspect eval agentic_evals/geobenchx -T group=spatial \
  --model openai-api/openrouter/openai/gpt-4o-mini \
  --model-base-url https://openrouter.ai/api/v1 \
  -M strict_tools=false \
  --model-role grader=openai/gpt-4.1-2025-04-14

uv run inspect eval agentic_evals/eo_gym \
  --model openai-api/openrouter/<provider>/<vision-model> \
  --model-base-url https://openrouter.ai/api/v1 \
  -M strict_tools=false \
  --model-role grader=openai/gpt-4.1-mini

uv run inspect eval agentic_evals/openearthagent -T regime=step \
  --model openai-api/openrouter/openai/gpt-4o-mini \
  --model-base-url https://openrouter.ai/api/v1 \
  --model-role grader=openai/gpt-4o-mini

uv run inspect eval agentic_evals/openearthagent -T regime=e2e \
  --model openai-api/openrouter/<provider>/<vision-model> \
  --model-base-url https://openrouter.ai/api/v1 \
  -M strict_tools=false \
  --model-role grader=openai/gpt-4o-mini
```

`-M strict_tools=false` is required. Inspect's OpenAI-compatible provider marks tools strict by default, and these tool schemas contain untyped arguments that strict mode rejects.

Earth-Bench regimes are `ap` (the agent plans its own steps) and `if` (the question lists the steps). Report them separately. `--sample-id 1` runs one task. `--limit 10` runs the first ten.

GeoBenchX `group` is `merge`, `process`, `spatial`, `heatmaps`, or `all`. The official score is an LLM judge of the whole solution. The judge model is the Inspect `grader` role. Record it with the result: scores are not comparable across judges. Step metrics (Tool-Any-Order, Tool-In-Order, Tool-Exact-Match, parameter accuracy) are computed as well and labeled as our addition.

OpenEarthAgent regimes are `step` (teacher-forced, no tools) and `e2e` (live tools, 15 model turns). Report them separately. `e2e` starts the local tool server on a prepared GPU host and stops it when the process exits. A cache hit for the five gpkg tools is used; a miss calls the worker. `vision=false` records the 533 image tasks as not applicable. The official judge is `gpt-4o-mini` on the `grader` role. Logged rates are fractions; the paper's tables are those rates times 100. Step metrics in the Earth-Bench sense are computed as well and labeled as our addition.

```bash
uv run inspect view
uv run inspect score logs/<run>.eval --scorer agentic_evals/earth_bench_scorer
```

`inspect view` shows transcripts, the tool trace, and per-step scores. `inspect score` recomputes metrics from a saved log without calling the model again.

## Tests

```bash
uv sync --extra earth-bench --extra geobenchx --group dev
uv run pytest
```

Tests use the files committed in this repo. The EO-Gym comparison against `agent_passk.py` skips until the server virtualenv exists. The OpenEarthAgent comparison against `evaluate_function` skips until setup has cloned the tool server.

## Weights & Biases

Model logging is enabled in `pyproject.toml` and targets the Forge project `agentic_evals` (entity `LLM4EO`). A run is uploaded when `WANDB_API_KEY` and `WANDB_ENTITY` are set. Weave is off. The `.eval` log remains the source of truth.

On the pod, `WANDB_BASE_URL` is `https://forge.coreweave.com/api/wandb`. The UI is `https://forge.coreweave.com/wandb`.

To keep a local run off W&B: `WANDB_MODE=disabled`.

## Layout

```
src/agentic_evals/tasks/<benchmark>/
  setup.py       that benchmark's download (EO-Gym also installs its server virtualenv; OpenEarthAgent also clones its tool server into data/)
  task.py        Inspect task
  dataset.py     prompts and samples
  data/          committed questions, gold, and the test split
  prompts/       judge templates, when the benchmark has a judge
  tools/         tool code and the MCP config that launches it
  server/        EO-Gym only: the copied FastAPI tool server
  metrics.py     metric functions
  scorers.py     Inspect scorers
  PORTING.md     what changed relative to upstream
  LICENSE
scripts/setup_benchmark.py    dispatch to one task's setup.py
data/                         gitignored rasters, the Drive archive, EO-Gym imagery, and the OpenEarthAgent download
logs/                         gitignored Inspect logs
```

## Licenses

Earth-Bench tool code is MIT, copyright OpenDataLab 2026 (`tasks/earth_bench/LICENSE`).

GeoBenchX tool code and the copied task files are MIT, copyright Varvara Krechetova 2025 (`tasks/geobenchx/LICENSE`).

EO-Gym code copied into this repository is CC-BY-4.0 (`tasks/eo_gym/LICENSE`). The image archives stay a download.

OpenEarthAgent's test split is Apache-2.0 (`tasks/openearthagent/LICENSE`). The evaluation prompts are copied from the GitHub repository, which has no license file; that copy is authorized. The tool server is cloned into `data/` and is not committed.
