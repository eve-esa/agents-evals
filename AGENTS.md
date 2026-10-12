# agentic-evals

Evaluation harness for Earth Observation agents. It runs agents on public benchmarks and scores each one with that benchmark's own published metrics.

## Metrics

When you add a benchmark, implement the metrics that benchmark's paper reports. Each benchmark keeps its own set. Do not invent metrics, variants, or a shared score that mixes them.

If a published definition looks wrong, stop and ask. Cite the paper and section in the metric module's docstring, and check the implementation against the paper's published numbers or its reference scorer before trusting it.

A judge that is not part of the paper's protocol is an extra scorer. Keep each judge's verdict separate. Do not majority-vote extras into the benchmark's own score.

## Configuration

A run is a YAML file in `configs/`:

```bash
uv run inspect eval --run-config configs/<name>.yaml
```

Model, endpoint, task arguments, generation settings, sample selection, and epochs belong in that file. Add a new file for a new run. Do not hardcode those choices, and do not start an eval from one-off CLI flags.

## Where evals run

Evals run on a Runpod pod. If the user has not given you a pod, ask for one before starting. Do not run an eval on the local machine.

The pod checkout is a separate copy. Before any command on the pod, sync this tree to that checkout and confirm the remote files match the code you intend to run. A pod that is behind is not a valid run. Sync again after every local change that the run depends on.

## Boundaries

- Python is managed with uv (`uv sync`, `uv run`). Do not use pip, poetry, or conda.
- Inspect AI is the harness. Do not rebuild model selection, tool connections, logs, epochs, or offline rescoring.
- Agents are LangGraph graphs run through `agent_bridge()`. Inspect's built-in `react()` is not the agent.
- `tasks/*` may import `agents/`, `scorers/`, and shared helpers. `agents/` and `scorers/` must not import `tasks/*`. Benchmark-specific logic stays in `src/agentic_evals/tasks/<benchmark>/`.
- This repo consumes benchmarks. Do not author new tasks here.
- Verify LangChain, LangGraph, and Inspect calls against the installed versions. These packages are 1.x; pre-1.0 examples do not apply.
- Code that calls a model or the network needs an offline fixture so tests run without API keys.
