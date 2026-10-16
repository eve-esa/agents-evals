# Evaluating EO agents, BIFOLD hands-on

Hands-on session of the [BIFOLD Agentic AI for Earth Observation workshop](https://www.bifold.berlin/news-events/events/view/event-details/agentic-ai-for-earth-observation), 19–21 October 2026 in Berlin. Pi School and ESA Φ-lab, EVE project.

In this folder we find the material for **"How do you know your agent works?"**, the hands-on notebook `eval_agents_handson.ipynb`.

## Session

Building an agent is straightforward. Showing that it works reliably is harder. In this session you evaluate an Earth Observation agent from end to end. You learn the step-by-step and end-to-end metrics, survey the EO agent benchmarks, and run a baseline agent on Earth-Bench with EVE. Then you read its execution traces, diagnose its failures, and try a change of your own.

The notebook scores Earth-Bench with its published metrics, through the `agentic_evals` harness. The model under test is EVE, served through an OpenAI-compatible endpoint.

## What You Learn

- Why an agent needs step-by-step metrics as well as end-to-end metrics
- How Tool-Any-Order, Tool-In-Order, Tool-Exact-Match and Parameter Accuracy are computed
- Which EO agent benchmarks exist, and what each one scores
- How to read an execution trace and diagnose a failed run
- How an LLM judge's verdict is reported together with the score
- How to write a LangGraph agent and run it through the same harness



## Resources

- `eval_agents_handson.ipynb`: the hands-on notebook
- [agentic-evals](https://github.com/eve-esa/agents-evals): the evaluation harness the notebook uses
- [Earth-Bench](https://arxiv.org/abs/2509.23141) (Feng et al., ICLR 2026): the benchmark scored in Part 3



## Folder Contents

```
.
├── eval_agents_handson.ipynb   # hands-on notebook
├── workshop.py                 # helpers: run, results_df, judge, leaderboard
├── trace_view.py               # execution trace views (show_trace)
├── figures/                    # figures used in the notebook
├── requirements.txt            # agentic-evals, pinned to a git branch
├── setup.sh                    # creates .venv, registers the kernel, copies .env
├── .env.example                # credentials template, copied to .env
├── .env                        # your credentials (not committed)
├── data/                       # benchmark rasters, downloaded on first run
├── logs/                       # Inspect eval logs
└── runs/                       # tool output
```

## Access the Lab

During the workshop you receive a username and a password. Log in at [https://eo-agents-eval.lab.eve-chat.chat](https://eo-agents-eval.lab.eve-chat.chat) with those credentials. The lab already provides the environment and the kernel `agentic-evals workshop`, so you can go straight to [Run the Notebook](#run-the-notebook).

## Run the Notebook

Open `eval_agents_handson.ipynb` and select the kernel `agentic-evals workshop`:

1. Run the cells in order. If you restart the kernel, run the setup cell again.
2. The notebook has three parts:
  - Part 1. Metrics for agents
  - Part 2. EO benchmarks for agents
  - Part 3. The evaluation framework, and a real Earth-Bench run
3. The same baseline as a run config: `configs/earth_bench/workshop_eve.yaml` in the agentic-evals repository.

## Troubleshooting

- `401` **or** `403` **from the model**: check `EVE_API_KEY` and `EVE_BASE_URL` in `.env`.
- `404 model not found`: check `MODEL_NAME`.
- `BudgetExceeded`: the key's token budget is spent. Ask the organisers for a key with budget. A full pass of the notebook uses about 1.8M tokens.
- `RateLimitError` **or HTTP 429**: the endpoint is busy. `run()` retries; if it persists, lower `max_samples`.
- **Wrong kernel or** `ModuleNotFoundError`: select `agentic-evals workshop`, restart the kernel, and run the setup cell again.

## Funding

This project is supported by the European Space Agency (ESA) Φ-lab through the Large Language Model for Earth Observation and Earth Science project, as part of the Foresight Element within FutureEO Block 4 programme.

## Citation

If you use this material in academic or research settings, please cite:

```bibtex
@misc{atrio2026evedomainspecificllmframework,
      title={{EVE}: A Domain-Specific {LLM} Framework for Earth Intelligence}, 
      author={Àlex R. Atrio and Antonio Lopez and Jino Rohit and Yassine El Ouahidi and Marcello Politi and Vijayasri Iyer and Umar Jamil and Sébastien Bratières and Nicolas Longépé},
      year={2026},
      eprint={2604.13071},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.13071}, 
}
```

