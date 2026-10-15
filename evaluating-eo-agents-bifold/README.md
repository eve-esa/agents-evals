# Evaluating EO agents, BIFOLD hands-on

Closed environment for `eval_agents_handson.ipynb`. It has its own virtualenv and its own `.env`. It does not share the agentic-evals checkout's `.venv`.

The notebook surveys EO-agent benchmarks, compares answer-only and trajectory metrics across Earth-Bench, GeoBenchX and EO-Gym, and walks through how the harness runs an eval. The hands-on part scores Earth-Bench with the package metrics (Feng et al., ICLR 2026, [arXiv:2509.23141](https://arxiv.org/abs/2509.23141), Appendix B.2). The ReAct graph is written in the notebook and passed to `langgraph_agent`. The ten Spectrum questions are `WORKSHOP_QUESTIONS` in `workshop.py`. `judge()` rescores a saved log with the trajectory judge, and `leaderboard()` builds one row per finished run in a log directory.

The same baseline as a run config: `configs/earth_bench/workshop_eve.yaml` in the agentic-evals repository.

## Setup

From this folder, on the SageMaker notebook instance:

```bash
bash setup.sh
```

`setup.sh` creates `.venv` with uv, installs `requirements.txt` (which includes `agentic-evals` with its Earth-Bench extra, from GitHub), registers the Jupyter kernel `agentic-evals workshop`, and copies `.env.example` to `.env` when `.env` is missing.

The library is pinned in `requirements.txt` to the `feat/workshop-notebook` branch of `https://github.com/eve-esa/agents-evals`. The notebook's install cell is commented out, because the lab already provides the environment.

Fill in `.env`:

- `EVE_API_KEY`
- `EVE_BASE_URL`
- `MODEL_NAME` (the id the endpoint lists on `GET /models`)

### On a SageMaker notebook instance

Keep this folder under `~/SageMaker`. That directory is the instance's persistent volume, so `.venv`, `.env`, `data/` and `runs/` survive a restart.

To build the environment automatically when the instance starts, attach `sagemaker_on_start.sh` as an OnStart lifecycle configuration. From a machine with the AWS CLI (`base64 -w0` on Linux, `base64 -i` on macOS, both without a line break):

```bash
aws sagemaker create-notebook-instance-lifecycle-config \
  --notebook-instance-lifecycle-config-name bifold-setup \
  --on-start Content=$(base64 -w0 sagemaker_on_start.sh)
aws sagemaker update-notebook-instance \
  --notebook-instance-name <your-instance> \
  --lifecycle-config-name bifold-setup    # the instance must be stopped
```

The hook starts `setup.sh` in the background and logs to `~/bifold-setup.log`. The first build takes several minutes and writes `.env` early. The kernel `agentic-evals workshop` appears in the launcher when `setup.sh` finishes.

Then start Jupyter with this folder as the working directory and select that kernel:

```bash
.venv/bin/jupyter notebook eval_agents_handson.ipynb
```

Rasters download into `data/` the first time the notebook reaches that cell. Logs go to `logs/`, tool output to `runs/`. Those directories stay in this folder.

`WANDB_MODE=disabled` is set in `.env.example` so a workshop run is not uploaded.
