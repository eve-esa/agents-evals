# SageMaker hands-on

Closed environment for `eval_agents_handson.ipynb`. It has its own virtualenv and its own `.env`. It does not use the repository `.venv`.

The notebook surveys EO-agent benchmarks, compares answer-only and trajectory metrics across Earth-Bench, GeoBenchX and EO-Gym, and walks through how the harness runs an eval. The hands-on part scores Earth-Bench with the package metrics (Feng et al., ICLR 2026, [arXiv:2509.23141](https://arxiv.org/abs/2509.23141), Appendix B.2). The ReAct graph is written in the notebook and passed to `langgraph_agent`. The ten Spectrum questions are `WORKSHOP_QUESTIONS` in `workshop.py`. `judge()` rescores a saved log with the trajectory judge, and `leaderboard()` builds one row per finished run in a log directory.

The same baseline as a run config: `configs/earth_bench/workshop_eve.yaml`.

## Setup

From this folder, on the SageMaker notebook instance:

```bash
bash setup.sh
```

`setup.sh` creates `.venv`, installs this repo with the Earth-Bench extra, registers the Jupyter kernel `agentic-evals workshop`, and copies `.env.example` to `.env` when `.env` is missing.

Fill in `.env`:

- `EVE_API_KEY`
- `EVE_BASE_URL`
- `MODEL_NAME` (the id the endpoint lists on `GET /models`)

Then start Jupyter with this folder as the working directory and select that kernel:

```bash
.venv/bin/jupyter notebook eval_agents_handson.ipynb
```

Rasters download into `data/` the first time the notebook reaches that cell. Logs go to `logs/`, tool output to `runs/`. Those directories stay in this folder.

`WANDB_MODE=disabled` is set in `.env.example` so a workshop run is not uploaded.
