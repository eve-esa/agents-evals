# agentic-evals

Evaluation harness for Earth Observation agents, built for the EVE project (Pi School and ESA Φ-lab).

It runs agents on public EO agent benchmarks under the same conditions, records the final answer and the tool-call trajectory, and scores each run with that benchmark's own published metrics. The harness is [Inspect AI](https://inspect.aisi.org.uk/). Agents are LangGraph graphs, and tools are reached over MCP.

## Benchmarks

| Benchmark | Task name | Metrics | Install |
| --- | --- | --- | --- |
| Earth-Bench ([Feng et al., ICLR 2026](https://arxiv.org/abs/2509.23141)) | `agentic_evals/earth_bench` | Accuracy; TAO, TIO, TEM, parameter accuracy, efficiency | `uv sync --extra earth-bench` |
| GeoBenchX ([Krechetova & Kochedykov, SIGSPATIAL 2025](https://arxiv.org/abs/2503.18129)) | `agentic_evals/geobenchx` | LLM-judge match / partial / no match; length difference | `uv sync --extra geobenchx` |
| EO-Gym ([Ma et al., 2026](https://arxiv.org/abs/2605.01250)) | `agentic_evals/eo_gym` | Pass@k with an LLM judge; tool metrics; parameters score | its own virtualenv, created by setup |

Where we fix a known problem in an original benchmark, the change and its reason are in [docs/benchmark_changes.md](docs/benchmark_changes.md). Install one benchmark at a time: the extras pull different geospatial stacks into the same environment.

## Install

Python 3.12 or newer, managed with [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/eve-esa/agents-evals.git
cd agents-evals
uv sync --extra earth-bench
cp .env.example .env        # then add your API keys
uv run python scripts/setup_benchmark.py earth-bench --modality spectrum
```

`setup_benchmark.py` downloads each benchmark's bulk data into `data/` (gitignored):

- `earth-bench`: task rasters. `--modality` is `spectrum`, `products`, `rgb` or `all` (16.6 GB). `--questions 1-5` fetches only those tasks.
- `geobenchx`: the dataset archive from Google Drive. One tool, `generate_contours_display`, needs a system GDAL.
- `eo-gym`: about 47 GB of imagery and the tool server's virtualenv. Start the server from `src/agentic_evals/tasks/eo_gym/server` with `data/venvs/eo-gym/bin/python -m eo_gym.server`.

Model ids of the form `openai-api/<name>/<model>` read `<NAME>_API_KEY` and `<NAME>_BASE_URL` from `.env`, for example `OPENROUTER_API_KEY` or `EVE_API_KEY`.

## Run

A run is a YAML file in `configs/<benchmark>/`: model, endpoint, task arguments, generation settings and samples. For example, [configs/earth_bench/qwen3_32b.yaml](configs/earth_bench/qwen3_32b.yaml) runs Qwen3-32B on five Earth-Bench Spectrum questions in the auto-planning regime:

```bash
uv run inspect eval --run-config configs/earth_bench/qwen3_32b.yaml
```

Write a new file for a new run rather than adding command-line flags. Then browse the log, or rescore it without calling the model again:

```bash
uv run inspect view
uv run inspect score logs/<run>.eval --scorer agentic_evals/earth_bench_scorer
```

## Tests

```bash
uv sync --extra earth-bench --extra geobenchx --group dev
uv run pytest
```

Tests need no API key. The metric tests check our scorers against each benchmark's published results or reference code.

## Layout

```
configs/<benchmark>/*.yaml          run configurations
src/agentic_evals/agents/           LangGraph agents and the trace recorder
src/agentic_evals/tasks/<benchmark>/
  task.py, dataset.py               the Inspect task and its samples
  metrics.py, scorers.py            the paper's metrics
  tools/                            the benchmark's tools, served over MCP
  PORTING.md                        file-level changes from upstream
docs/benchmark_changes.md           scoring changes from each original benchmark
```

Optional run logging to Weights & Biases is configured in `pyproject.toml`. It uploads only when `WANDB_API_KEY` and `WANDB_ENTITY` are set; `WANDB_MODE=disabled` keeps a run local.

## Citations

If you use these benchmarks, cite the original papers:

```bibtex
@inproceedings{feng2026earthagent,
  title     = {Earth-Agent: Unlocking the Full Landscape of Earth Observation with Agents},
  author    = {Feng, Peilin and Lv, Zhutao and Ye, Junyan and Wang, Xiaolei and Huo, Xinjie and Yu, Jinhua and Xu, Wanghan and Zhang, Wenlong and Bai, Lei and He, Conghui and Li, Weijia},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  eprint    = {2509.23141},
  archivePrefix = {arXiv}
}

@inproceedings{krechetova2025geobenchx,
  title     = {GeoBenchX: Benchmarking LLMs in Agent Solving Multistep Geospatial Tasks},
  author    = {Krechetova, Varvara and Kochedykov, Denis},
  booktitle = {ACM SIGSPATIAL},
  year      = {2025},
  eprint    = {2503.18129},
  archivePrefix = {arXiv}
}

@misc{ma2026eogym,
  title     = {EO-Gym: A Multimodal, Interactive Environment for Earth Observation Agents},
  author    = {Ma, Sai and Li, Zhuang and Li, Sichao and Xu, Xinyue and Zhu, Ruibiao and Boston, Tony and Taylor, John A.},
  year      = {2026},
  eprint    = {2605.01250},
  archivePrefix = {arXiv}
}
```

## Licenses

Code copied from the benchmarks keeps its licence, in each task folder: Earth-Bench tools are MIT (OpenDataLab 2026), GeoBenchX tools and tasks are MIT (Varvara Krechetova 2025), and the EO-Gym server is CC-BY-4.0. Imagery and rasters stay a download.
