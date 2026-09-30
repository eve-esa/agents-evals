# agentic-evals

Evaluation harness for Earth Observation agents, built for the EVE project (Pi School × ESA Φ-lab).

Its job: run EVE and baseline agents against public EO agent benchmarks under identical conditions, and report both end-to-end answer quality and trajectory quality.

## Why this repo exists

Two BIFOLD deliverables depend on numbers this harness produces:

- **Oral / abstract** — "From Tools to Ecosystem: EVE's Community-Driven MCP Registry for an Agentic Platform for Earth Intelligence". Claims an initial benchmark of agentic EVE against public EO sets, reporting end-to-end accuracy *and* trajectory metrics, comparing single-agent vs multi-agent on identical tasks.
- **Hands-on session (1h30–2h)** — "How Do You Know Your Agent Works? Evaluating Earth Observation Agents with Existing Benchmarks". Participants assemble an agent from boilerplate, run it against a benchmark, and compute metrics on a shared leaderboard.

Everything reported in those two places must be reproducible from this repo.

## Scope

**In scope.** The harness: agent runners, benchmark adapters, trace capture, metric computation, result serialization.

**Out of scope** (separate repos; do not add them here):

- Demo material for the live EVE tour.
- Hackathon MCP starter template and registry CI.
- The EVE platform itself (backend, frontend, tool registry).
- Authoring new benchmark tasks. This repo *consumes* benchmarks; it does not create them. (ESA-EO-Bench and any manually-created EO benchmark live elsewhere and will be plugged in as adapters.)

If a task looks like it belongs to one of the above, stop and confirm before adding it.

**Deferred, not cancelled.** These are decided and should not be built yet:

- *Multi-agent configurations.* v1 evaluates a **single agent** only. How the multi-agent setup is defined will be resolved later, so do not design around a specific orchestration scheme — just avoid baking single-agent assumptions into the core (a run produces a trace; how many agents produced it is the runner's business). Note the BIFOLD oral abstract promises a single-vs-multi comparison on identical tasks, so that claim is not yet backed by this harness.
- *The shared leaderboard.* Publishing and its format come later. Until then, serialize results to local files in a form that is straightforward to aggregate later; do not invent a submission protocol now.

## Stack

- Python, managed with **uv** (`uv sync`, `uv run`). Do not use pip/poetry/conda in instructions or CI.
- **[Inspect AI](https://inspect.aisi.org.uk/)** is the harness. Each benchmark is an Inspect task (`@task` = dataset + solver + scorers) under `src/agentic_evals/tasks/<benchmark>/`, registered through the `inspect_ai` entry point in `pyproject.toml` (`src/agentic_evals/_registry.py`). Inspect provides model selection (`--model openai-api/<name>/<model>` with `<NAME>_BASE_URL` / `<NAME>_API_KEY`), MCP tool connections, `.eval` logs, epochs/reducers, model roles for judges, and offline re-scoring (`inspect score`). Do not rebuild any of that.
- **LangGraph** for agent construction (latest is `langgraph` 1.2.x, requires Python >= 3.10; `langchain` 1.4.x). This matches the agent stack already used in EVE. Our agents are ordinary LangGraph graphs that run as Inspect solvers through `agent_bridge()` (`ChatOpenAI(model="inspect")` is routed to whatever `--model` is), selected with `--solver`. Inspect's built-in `react()` is not our reference agent.
- Tools are reached over **MCP** wherever the benchmark permits, since EVE's production path is MCP-based and we want the evaluated configuration to resemble the deployed one.
- Upstream benchmark repos are **fetched, never vendored or imported**: `scripts/fetch_upstream.py` downloads them at the revisions pinned in `upstream.lock.toml` into a gitignored `third_party/`, and each upstream tool server runs in its own virtual environment (Earth-Agent pins numpy 1.24 / Python ≤ 3.11, EO-Gym needs numpy 2 / Python ≥ 3.12 — they cannot share an environment with each other or with us).

Before using a LangGraph/LangChain/Inspect API, verify it against the installed version's documentation or source. These packages moved to 1.x recently and many pre-1.0 patterns found in blog posts no longer apply.

Two properties of the bridge matter for scoring. Inspect rebuilds `state.messages` from the model calls passing through the bridge, so it only sees tool results as the model saw them and records no tool events of its own; and it tracks a single "main thread", so sub-agent calls do not appear in `state.messages`. Our agents therefore write their own trace (tool, arguments, output, error flag, timing) into the sample `store()` via a LangGraph callback, and scorers read that trace. The bridge also drops client-side generation parameters (temperature etc.) by default; set them through Inspect (`--temperature`).

## System under test

The EVE endpoint is deployed on demand, for the duration of a benchmark run. There is no long-lived URL to hardcode and no guarantee that two runs hit the same deployment.

Consequences for the harness: the endpoint is run configuration, supplied at invocation and never committed. Whatever identifies the deployment at run time — URL, model name, version, and any deployment identifier the endpoint reports — is captured into the run record when the run starts, not assumed from a config default. If the endpoint cannot state its model version, treat that as a defect worth reporting rather than filling in a plausible value, because results from an unidentified deployment are not comparable to anything.

### Baselines

EVE-Instruct is a **dense 24B** model fine-tuned from `mistralai/Mistral-Small-3.2-24B-Instruct-2506`. Baselines are chosen to match that scale, so that differences are attributable to the model rather than to its budget. Pin exact checkpoints; "the latest Qwen" is not a baseline.

| Role | Checkpoint | Size |
| --- | --- | --- |
| Base-model ablation | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | 24B dense |
| Baseline | `Qwen/Qwen3.8-27B` | 27B dense |
| Baseline | `google/gemma-4-31B-it` | 31B dense |
| Baseline | `openai/gpt-oss-20b` | 20.9B total, **3.6B active** (MoE) |

Two things about this table are deliberate and should survive contact with someone trying to tidy it up:

- **The Mistral baseline is 3.2, not Mistral Small 4.** Mistral Small 3.2 is the exact checkpoint EVE-Instruct was fine-tuned from, which is what makes it an ablation: it measures what domain adaptation bought. It is deprecated on Mistral's API (since 2026-04-30) but the Apache 2.0 weights remain on Hugging Face, and self-hosting it is the point. Mistral Small 4 is not a successor at this scale — it is 119B total / 6.5B active MoE — so substituting it would conflate domain adaptation with a change of architecture and a 5× scale jump.
- **GPT-OSS-20B is not size-matched and must be reported as such.** Its 3.6B active parameters per token mean it does roughly a seventh of EVE's per-token compute despite the similar-looking label. It is included as a deliberately cheap point of comparison. Always report it with active parameters alongside total; never let it sit in a column headed only "size".

Prefer dense models when extending this set. Several open-weight models in the 20–31B range are MoE with 3.6–4B active parameters (Gemma 4 26B-A4B is 25.2B total / 3.8B active, and is *not* the Gemma baseline above), so matching on the headline parameter count silently mismatches compute.

**Vision requirements differ per benchmark.** EO-Gym requires a vision-capable model: its task prompts carry the imagery as image parts of the user message, and its zoom tools return new crops into the conversation (`eo_gym_evaluation/driver.py`, `build_initial_messages` / `run_row`). A text-only model cannot even read an EO-Gym question, so the EO-Gym task declares a vision requirement and a text-only model is reported as *not applicable* there, not scored. GPT-OSS-20B is text-only, and whether EVE-Instruct qualifies depends on the deployment. The rest of this paragraph is about Earth-Bench only.

**For Earth-Bench, a vision-capable backbone is not required.** Earth-Agent delegates perception to expert models in the Perception kit; the LLM only orchestrates tool calls and never receives raw imagery. The evidence is in the paper's own results: DeepSeek-V3.1 and Kimi-K2 are text-only and post the strongest trajectory scores on the RGB subset (DeepSeek-V3.1 reaches 88.98 Tool-Any-Order there, above every VLM tested). The authors additionally released the RGB expert-model weights for online invocation (`online_infer` branch, 2026-03-31). So GPT-OSS-20B being text-only does not disqualify it, and baselines need not all be VLMs. What this does mean is that RGB-subset *accuracy* is capped by the expert models rather than the backbone — the paper says outputs diverge from ground truth even when the agent picks the correct tools — so read RGB accuracy as a statement about the toolkit, not about the model under test.

## Benchmark: Earth-Bench (v1 target)

v1 targets **Earth-Bench**, **EO-Gym**, and **GeoBenchX** (see the sections below). Earth-Bench comes from Earth-Agent (ICLR 2026, arXiv:2509.23141). It was chosen because it is the only EO agent benchmark that ships expert-annotated *reasoning trajectories*, which is what the trajectory metrics in both BIFOLD submissions require. It is also MCP-based and covers 248 expert-curated tasks over Spectrum, Products, and RGB modalities with 104 domain tools, averaging ~5.4 reasoning steps per task.

Earth-Bench defines two regimes; support both, and report them separately:

- **Auto-Planning (AP)** — step-implicit. The agent plans its own trajectory.
- **Instruction-Following (IF)** — step-explicit. The query carries the steps; the agent translates them into actions.

Other benchmarks (ThinkGeo, ESA-EO-Bench, and answer-only sets like UnivEARTH and MapEval-API) are deliberately deferred. Adding one means a new folder under `src/agentic_evals/tasks/` plus one import in `_registry.py`; nothing in the agents or scorers of other tasks should change. Do not implement them yet.

### Data footprint

The Earth-Bench assets live on Hugging Face at [`Sssunset/Earth-Bench`](https://huggingface.co/datasets/Sssunset/Earth-Bench) and are **downloaded locally, not streamed**: 13,657 files totalling **16.6 GB**, mostly GeoTIFF rasters (per-date LST/NDVI and similar), laid out one directory per task (`question1/` … `question248/`).

Two consequences:

- Partial download is the normal mode. Because the layout is one directory per task, fetch only the tasks in scope via `allow_patterns` rather than pulling all 16.6 GB. The upstream evaluation scripts slice the task list by modality using fixed index ranges — Spectrum `[0:100]`, Products `[100:188]`, RGB `[188:]` — so a modality subset is a contiguous range of task directories. Select modalities by **question number** (Spectrum 1–100, Products 101–188, RGB 189–248), not by list position: `extracted_tool_calls_GT.json` has 247 entries because `question43` is missing, so index slices over it are off by one from `question43` onwards.
- 16.6 GB is not downloadable inside a 1h30 hands-on session on participant laptops. The data must be pre-staged on the hosted infrastructure the organizers provide, or the session must run on a small pinned subset. Decide this before the session, not on the day.

Ground truth does **not** require the rasters. The upstream repo ships `benchmark/question.json` and `extracted_tool_calls_GT.json` (reference tool calls and answers) directly in git, so metric code can be developed and tested against real ground truth with no bulk download at all. Use this for fixtures.

Note the upstream repo pins `langchain` 0.3.27 / `langgraph` 0.6.6 — pre-1.0. Treat it as a source of data, ground truth, and metric semantics, not as code to vendor into our 1.x stack.

### Bringing the metrics up without running an agent

Upstream publishes, in git, the **predicted trajectories of 13 models × 2 regimes** under `evaluate_langchain/<model>_<AP|IF>/`. Each directory contains `extracted_tool_calls.json` (what that model actually called, for all 248 tasks, including the exact `query` sent to the model) alongside `step_by_step_evaluation_results.json` and `end_to_end_evaluation_results.json` (the scores the authors computed from it).

**The published score files cover only 59 RGB tasks, not the whole of Table 1.** Both `evaluate/step_by_step.py` and `evaluate/end_to_end.py` score `list(gt)[188:]`. Because `question43` is missing from the ground truth, that slice starts at `question190`, so `question189` is silently skipped and every published file covers `question190`–`question248`. Consequences: the exact reproduction check validates our metric code on RGB only; for Spectrum and Products we compute scores from the published predictions and compare aggregates against the paper's Table 1, which is a weaker check. Upstream's answer-level accuracy is also plain string matching after regex extraction (`<Answer>X<Answer>`, `<Answer>X</Answer>`, else the last standalone letter A–F; any text containing `FAIL` is a failure) — there is no LLM judge in Earth-Bench's published protocol.

This is the correctness oracle for our metric implementation. Feed their predictions and their ground truth into our TAO / TIO / TEM / Parameter Accuracy / Efficiency code and reproduce their published numbers. If we cannot reproduce them, our implementation is wrong — a formula transcribed correctly from Appendix B.2 can still disagree on tie-breaking, normalisation, or how a failed call is counted, and this is the only way to find that out.

Do this first. It costs no API calls, needs none of the 16.6 GB, and validates the metrics against the reference implementation rather than against our reading of the paper. This is the offline fixture the Conventions section asks for, using real published data.

The 13 models (Table 7 of the paper), each in both AP and IF: GPT-5, GPT-4o, Gemini-2.5-Flash, Mistral-Large, Qwen3-Max-Preview, Qwen3-32B, Qwen-Plus, Kimi-K2, DeepSeek-V3.1, Seed-1.6, LLaMA-4-Maverick, GLM-4.5v, InternVL-3.5.

**Do not read these as baselines for our models.** Two reasons. Their `Mistral` run is **Mistral-Large**, not Mistral Small 3.2 24B, so it says nothing about our base-model ablation; `qwen3-32b` is the only published run anywhere near our 24–31B band. And every model in Table 7 was called over a hosted API, inside *their* agent scaffold — different from a self-hosted checkpoint inside ours. Use these directories to prove our metric code is correct, and generate our own numbers for comparison.

### How reproducible is their scaffold

Mixed, and the split matters for which subsets we can meaningfully run.

**Reproducible.** The orchestration layer is fully committed: `scripts/langchain_<model>.py` (IF) and `scripts/langchain_<model>_autoplan.py` (AP) for each of the 13 models, each carrying its system prompt inline, `temperature=0.1` and `recursion_limit=50`, built on `create_react_agent` from `langgraph.prebuilt`. Per-model `agent/config_*.json` launches the five tool kits as MCP servers over `fastmcp`. All five kits are present under `agent/tools/` (Analysis, Index, Inversion, Perception, Statistics).

**Not reproducible as shipped.** On `main`, the Perception kit does not run models at all — `get_model_output` looks answers up in a precomputed CSV, keyed on exact image path, and the path is hardcoded to the authors' machine (`/root/autodl-tmp/Earth-Agent/benchmark/model_results.csv`). The authors confirm this in the `online_infer` README: the lookup table existed "to enable evaluation on Earth-Bench in GPU-free environments". Live perception means the `online_infer` branch: seven CV models (MSCN, RemoteCLIP, RemoteSAM, InstructSAM, SM3Det, Strip-R-CNN, ChangeOS) from seven upstream repos, served as separate FastAPI processes, with three mutually incompatible requirements files, needing a GPU — and weights distributed only via Baidu Pan.

Consequences to design around:

- **Spectrum and Products are the safe subsets.** Their tools are plain numpy/rasterio/scipy and reproduce cleanly. Only RGB (tasks 189–248) depends on the perception lookup or the expert-model services.
- **Treat `'Failed to call model'` as a tool failure.** The lookup is wrapped in a bare `except: pass` that returns that string as a normal observation, so a misconfigured setup degrades silently and the agent reasons over the failure text. Detect it explicitly; otherwise broken perception looks like poor model performance.
- Do not trust the README's file map: it references a `main.py` and a top-level `tools/` that are not in the repo, and the MCP configs point at `tools/...` while the code lives at `agent/tools/...`.
- Their Mistral run is ambiguous: the paper's Table 7 says Mistral-Large, but `agent/config_Mistral.json` specifies `mistralai/mistral-medium-3`. Another reason not to read it as a base-model reference.

None of this blocks the metric-validation step above, which is the point of doing that first: it consumes their published trajectories as data and touches none of this machinery.

Upstream quirks that the published numbers depend on (the `[188:]` slice, TEM scoring 0 when the gold trajectory is empty while TAO/TIO/Parameter Accuracy score 1.0, the data path glued onto the question with no separator, the case-sensitive regime lookup that picks the right regime only by luck) are reproduced behind explicit flags so the published files can be matched, and switched off for our own runs where they are not part of the prompt. The prompt concatenation quirk is kept in our runs because it is what the model sees in the published protocol.

## Benchmark: EO-Gym (v1 target)

EO-Gym (arXiv:2605.01250) is a Gymnasium-style EO workspace with 35 tools and a 1,436-task manually verified test split. Code and data are both in the Hugging Face dataset repo [`paperuploadacount/EO-Gym`](https://huggingface.co/datasets/paperuploadacount/EO-Gym) (CC-BY-4.0, ~47 GB, mostly `EO_GYM_DATA.zip` and `image_cache.zip`); there is no GitHub repo. Its tools run behind a FastAPI `/execute` server, not MCP, so we put a thin MCP proxy in front of it.

Its evaluation protocol, and where the released code disagrees with the paper:

- **Pass@k.** The paper (Section 5) describes the unbiased estimator of Chen et al. (2021). The released scorer `eo_gym_evaluation/agent_passk.py` computes an **ordered early-stop** instead: it judges `traj0`, `traj1`, `traj2` in order, stops at the first correct one, and sets Pass@k = 1 iff `first_correct_traj < k`. The two agree on Pass@3 with n = 3 but differ per task on Pass@1 and Pass@2. We reproduce the released code with a custom reducer and report Inspect's unbiased `pass_at_k` alongside under a separate name.
- **Consumed trajectories.** Tool metrics are computed only over trajectories up to the first correct one, and *pooled* across tasks: Tool-any / Tool-in-order / Tool-exact-match (each binary per trajectory) and the parameter score (share of gold steps whose normalised arguments and observation both match) are averaged over all consumed trajectories, and `efficiency_consumed` is total consumed calls over total reference steps. These are not per-task means, so they are computed by a whole-run metric, not by averaging per-task scores.
- **Judge.** Final answers are judged by GPT-4.1-mini with `SEMANTIC_FINAL_ANSWER_JUDGE_SYSTEM_PROMPT`. Record the judge model with every result.

Reproducibility gaps in the release, analogous to Earth-Agent's:

- **Multiple trajectories are not generated by the released code.** The driver runs each row once and skips any `uid` already in its output file (rows without a `uid` fall back to `row{N}`); only the scorers know about trajectories, which they group by a `:trajN` suffix on the `uid`. The step that creates those suffixed records is not released. Inspect epochs fill this gap: the epoch number is the trajectory ID.
- **The released driver's output is not the released scorer's input.** The driver writes `steps` / `tool_schema_mode`; `agent_passk.py` reads ShareGPT-style `conversations` and needs a `tool_call_mode` to assign the run to a "ground". The conversion is not released. The scorers' default paths also point at unreleased files (`eo_gym_test_full.jsonl`, `EO_GYM_datasets/eo_gym_test_dataset_traj.csv`, `Agent_TEST_outputs/`).
- **The released scorer does not import as shipped.** `agent_passk.py` imports `SEMANTIC_FINAL_ANSWER_JUDGE_SYSTEM_PROMPT` / `TOOL_ARGUMENT_SEMANTIC_JUDGE_SYSTEM_PROMPT` from `eo_gym_data_synthesis.prompt_manager`, which defines them in `evaluation_prompts.py` but does not re-export them. Our differential test injects the aliases before importing; upstream files are not edited.
- **Only the Verified tool mode exists.** Perception tools hard-code `OPTICAL_BBOX_BACKEND = "ground_truth"`, so the paper's Unverified setting cannot be run from the release. Our task rejects `tool_mode="unverified"` rather than silently running Verified.
- The driver's own inline judge (`semantic.py`, used with `--semantic-review`) uses a different prompt from the released scorer; reported numbers come from the scorer's `SEMANTIC_FINAL_ANSWER_JUDGE_SYSTEM_PROMPT`, which is what we use. The scorer's tool-argument judge is only used in its counting-question analysis; `parameters_score` itself uses exact match of normalised arguments and observations.
- For OpenAI models the driver's default image mode pastes base64 data URLs into the prompt text as "image references"; we list the relative `EO_GYM_DATA/...` paths instead (its `file_url`/`path` modes, used for vLLM), which is what the gold trajectories and the tool server use.
- There are no published per-model outputs, so there is no oracle like Earth-Bench's. Our metrics are validated by a differential test against `agent_passk.py` (run in EO-Gym's own venv) on synthetic trajectories built from real test records, which is weaker.

## Benchmark: GeoBenchX (v1 target)

GeoBenchX (Krechetova & Kochedykov, SIGSPATIAL 2025, [arXiv:2503.18129](https://arxiv.org/abs/2503.18129)) is 202 geospatial tasks with ordered reference solutions (`benchmark_set/tasks_and_reference_solutions.json`) and 24 tools in `geobenchx/agent.py` (`plot_contour_lines` is defined and not given to the agent). The datasets are a Google Drive archive linked from `data/README.md`, fetched with `scripts/fetch_upstream.py geobenchx --with-data`. The task JSON and the metric tests do not need it.

Its published score is a whole-solution LLM judgment, not step metrics. Section 5 and `geobenchx/evaluation.py` (`score_task_solution`, `generate_eval_stats`):

- Render the candidate and every reference with `get_solution_code(..., add_num=False)` and ask one judge for 0 (no match), 1 (partial), or 2 (match). The released code records a single `evaluator_model`. The paper's panel (Claude Sonnet 3.5, GPT-4.1, Gemini 2.5 Pro Preview) is three such runs; we store each judge separately and do not majority-vote. The tuning file was judged by `gpt-4.1-2025-04-14`.
- `length_diff` is `(n_pred - mean reference length) / mean reference length`, or `n_pred` when that mean is 0. Report the mean and the median. It is a signed ratio. The paper's sentence that the ratio is relative to the candidate length does not match the code; the tables follow the code.

Seventeen tasks have more than one reference solution. 79 are a single `reject_task` (unsolvable). 3 have an empty reference (`Control question`). The paper's 123 "solvable" tasks are our `solvable` (120) plus `control` (3).

`geobenchx_steps` applies Earth-Bench TAO / TIO / TEM / parameter accuracy, max over references, and is our addition. An empty reference scores 1 only when the candidate is empty. A candidate that calls other tools and then `reject_task` is a full match under their judge instructions and a broken prefix under those formulas, which is why the step scores stay out of the official scorer. `evaluator_tuning_set.json` (50 tasks with both a human score and a GPT-4.1 score) is the judge-calibration fixture; there are no published per-model trajectories to reproduce a table from.

The agent upstream runs is temperature 0 and `recursion_limit=25` (LangGraph supersteps; the paper calls the same parameter 25 iterations). The system prompt is `SYSTEM_PROMPT + RULES_PROMPT`, including today's date. Tool text is what the model sees; maps stay in their `image_store` and are not sent back.

### Benchmark landscape

The WAE literature review (`Literature review` sheet, `Datasets` tab) classifies EO agent benchmarks into **Agent traces** and **Answer-only**. That distinction is the spine of the hands-on session, and it is also our adapter roadmap: only trace benchmarks can exercise the trajectory metrics.

Trace-bearing candidates, smallest first, for when adapters are added:

| Benchmark | Tasks | Trajectory detail |
| --- | --- | --- |
| GeoAgentBench | 57 | Golden executable toolchains, length 3–17 (mean ~7), 6 domains |
| GeoNatureAgent | 93 | 18 categories, 16 tools |
| GeoBenchX | 202 | 197 with reference steps, mean 3.89, max 14 (**v1 target**; official score is the LLM match, not step metrics) |
| DisasterBench | 233 | Expert-verified plans, 1–9 steps |
| Earth-Bench | 248 | ~5.4 steps, 104 tools (**v1 target**) |
| TerraBench | 403 | 24,500 verified execution steps, 3 tracks, 8 domains |
| ThinkGeo | 486 | 1,778 expert-verified steps, 14 tools |
| DORA | 515 | 3,500 gold tool-call steps, 45 real events |

Answer-only sets (UnivEARTH, OpenEarth-Bench, MapEval-API, GeoCode, ZephyrusBench, …) can only exercise end-to-end Accuracy. Useful for the session's contrast between answer-only and trajectory evaluation; not useful for our headline trajectory numbers.

**Having gold traces is not the same as shipping trace metrics.** GeoBenchX's reference solutions are ordered `{function_name, arguments}` steps, and its own evaluation scores the candidate trajectory as a whole with an LLM judge (one match score) plus a mean/median solution-length difference. Earth-Bench remains the only benchmark here that defines step-by-step metrics, implements them, and publishes per-model results. `geobenchx_steps` still computes TAO/TIO/TEM/parameter accuracy as our addition, max over references, and labels them as such. Unsolvable tasks use `reject_task` as the reference (m=1). `evaluator_tuning_set.json` is 50 tasks scored by both a human and GPT-4.1, the calibration set for judge disagreement.

## Metrics

Implement exactly the Earth-Bench dual-level protocol. Do not invent variants or silently change a definition; if a definition seems wrong, raise it rather than patching it.

Notation: ground-truth tool sequence `t* = (t*_1..t*_m)`, predicted `t = (t_1..t_n)`, with corresponding argument sequences `in*` and `in`.

**End-to-end**

- **Accuracy** — indicator that the final answer matches the reference, averaged over tasks.
- **Efficiency** — `|τ| / |τ*|`, the ratio of predicted to ground-truth tool-call counts. Note this is a ratio, not a score in [0,1]: values above 1 mean the agent took more calls than the expert. Never report it as a percentage.

**Step-by-step (trajectory)**

- **Tool-Any-Order (TAO)** — `|Set(t*) ∩ Set(t)| / |Set(t*)|`. Coverage of required tools, ignoring order and duplicates.
- **Tool-In-Order (TIO)** — length of the longest prefix of `t*` matched as a *subsequence* of `t`, divided by `m`.
- **Tool-Exact-Match (TEM)** — length of the longest common *prefix* of `t` and `t*`, divided by `m`.
- **Parameter Accuracy** — longest prefix where both the tool identifier and its arguments match, divided by `m`.

TAO and TIO tend to look strong while TEM and Parameter Accuracy collapse, because agents insert irrelevant steps. Expect that pattern; it is the interesting finding, not a bug in the harness.

Final-answer scoring follows each benchmark's own protocol: string matching for Earth-Bench (upstream has no judge), an LLM semantic judge for EO-Gym. Where a judge is used, it is bound to Inspect's `grader` model role; record the judge model and its prompt alongside every result: a score is not comparable across runs with different judges. The hands-on session explicitly teaches the limitations of LLM-as-a-judge, so the harness should make judge disagreement inspectable rather than hiding it behind a single number: run each judge as a separate scorer so every verdict is stored per sample, rather than a panel reduced to one majority vote. Extra judges we add (e.g. agentevals' trajectory judge) are reported as our additions, never mixed into a benchmark's own metrics.

## Conventions

- **Traces are the primary artifact.** Every run emits a structured trace of tool name, arguments, data sources, and outputs — our own trace in each sample's `store()`, persisted in the Inspect `.eval` log alongside the full transcript. Metrics are computed *from* persisted traces by scorers, and must be re-runnable offline with `inspect score` without re-executing agents, which matters because agent runs cost money and hit rate-limited EO APIs. W&B (via `inspect-wandb`) is a view onto these logs, not the source of truth.
- **Runs are immutable and fully described.** A result is meaningless without the model, agent configuration, benchmark version, regime (AP/IF), judge, and seed recorded next to it.
- **Agent configurations must go through the same path.** When the multi-agent configuration arrives, the comparison is only valid if it differs from the single-agent one solely in orchestration. Nothing in the core should be reachable only by a single-agent runner.
- Keep benchmark-specific logic inside its task folder (`src/agentic_evals/tasks/<benchmark>/`). Agents and shared code must not know that Earth-Bench, EO-Gym, or GeoBenchX exist.
- Network-touching and model-touching code needs offline fixtures so tests can run without API keys or ESA credentials.

## Repository layout

```
pyproject.toml            uv project; registers the package with Inspect ([project.entry-points.inspect_ai])
                          and configures inspect-wandb ([tool.inspect-wandb.*])
upstream.lock.toml        pinned upstream revisions + the tools-venv package lists
scripts/
  fetch_upstream.py       fetch pinned upstreams into third_party/, build their venvs, export EO-Gym tool schemas and the GeoBenchX judge prompt
  export_eo_gym_tools.py  run with EO-Gym's own python; writes third_party/eo_gym_tools.json
  export_geobenchx_judge_prompt.py  run with GeoBenchX's python; writes third_party/geobenchx_judge_prompt.txt
  geobenchx_mcp.py        GeoBenchX's 24 tools as an MCP stdio server (their venv)
src/agentic_evals/
  _registry.py            everything Inspect can address by name (agentic_evals/<name>): tasks, agents, scorers
  paths.py                locations of third_party/, venvs, runs/ (each overridable by env var)
  tooling.py              task <-> agent contract: tasks publish MCP server configs into the sample store
  run_identity.py         setup solver: records what the endpoint's /v1/models reports
  max_tokens.py           `--max-tokens -1` is recorded on the eval and omitted from the provider request
  wandb_compare.py        task-end hook: shared summary keys plus results and trace tables on the W&B run
  agents/                 benchmark-agnostic agents (Inspect solvers)
    langgraph_react.py    LangGraph ReAct agent via agent_bridge(); MCP tools from tooling.mcp_servers()
    trace_callback.py     LangChain callback writing our tool-call trace into the sample store
  scorers/                benchmark-agnostic scoring extras
    judges.py             optional whole-trajectory LLM judges (agentevals prompt), one scorer per judge
    agreement.py          pairwise agreement / Cohen's kappa between verdicts in a log
  tasks/<benchmark>/      one folder per benchmark, same shape:
    task.py               the @task (dataset + setup solvers + default agent + scorers + metadata)
    dataset.py            upstream records -> Inspect Samples (prompts reproduced from upstream)
    tools.py              per-sample MCP server configs for the upstream tool servers
    metrics.py            pure metric functions, cited and tested against upstream
    scorers.py            Inspect scorers/reducers wrapping metrics.py, reading the trace
    mcp_proxy.py          (eo_gym only) MCP stdio server in front of EO-Gym's HTTP /execute
tests/                    offline tests; `upstream`-marked ones skip if third_party/ is missing
third_party/              (gitignored) Earth-Agent/, EO-Gym/, GeoBenchX/, venvs/{earth-agent,eo-gym,geobenchx}, eo_gym_tools.json, geobenchx_judge_prompt.txt
logs/, runs/              (gitignored) Inspect .eval logs; per-sample tool output directories
```

How a run flows: `inspect eval agentic_evals/<task>` builds the task → setup solvers record the run identity and publish this sample's MCP servers into the store → the solver (default `langgraph_react`, or `--solver ...`) connects to those servers, runs, and writes its trace to the store → scorers read the trace plus the task's gold data from `Sample.metadata` → everything is persisted in the `.eval` log and mirrored to W&B.

Dependency direction: `tasks/*` may import `agents/`, `scorers/`, `tooling`, `paths`; `agents/` and `scorers/` must never import `tasks/*`. Upstream code is never imported by `agentic_evals`; it runs as separate processes (MCP/HTTP) or, in tests, inside its own venv via subprocess.

Extending:

- **New benchmark:** add `tasks/<name>/` with the five files above, import the `@task` in `_registry.py`, publish gold steps as `metadata["reference_trajectory"]` (list of `{name, arguments, output}`) so the generic judges work, and add an offline test that validates metrics against the benchmark's own reference implementation or published numbers.
- **New agent:** add an `@agent` in `agents/`, get tools from `tooling.mcp_servers()`, record calls with `TraceRecorder`, register it in `_registry.py`; select it with `--solver agentic_evals/<name>`. It must not contain benchmark-specific prompts or tools.

## Running

```bash
uv sync
uv run python scripts/fetch_upstream.py earth-agent --modality spectrum   # code, tools venv (needs gdal-config), rasters
uv run python scripts/fetch_upstream.py eo-gym --build-venv               # code, test set, tool schemas (data: --with-data)
uv run python scripts/fetch_upstream.py geobenchx --build-venv            # code, tools venv, judge prompt (data: --with-data)
uv run pytest                                                             # offline: metric reproduction and differential tests

export EVE_BASE_URL=... EVE_API_KEY=...
uv run inspect eval agentic_evals/earth_bench -T regime=ap -T modality=spectrum \
  --max-tokens -1 \
  --model openai-api/eve/<model> -M strict_tools=false
uv run inspect eval agentic_evals/eo_gym --model openai-api/<name>/<vlm> -M strict_tools=false \
  --max-tokens -1 \
  --model-role grader=openai/gpt-4.1-mini          # needs EO-Gym's tool server + data
uv run inspect eval agentic_evals/geobenchx -T group=spatial \
  --max-tokens -1 \
  --model openai-api/<name>/<model> -M strict_tools=false \
  --model-role grader=openai/gpt-4.1-2025-04-14    # needs the GeoBenchX tools venv + data
uv run inspect view --host 127.0.0.1 --port 7575   # on the pod; then forward the port locally
uv run inspect score <log> --scorer agentic_evals/earth_bench_scorer   # re-score offline
```

Experiments run on the pod `ssh root@157.157.221.177 -p 32257 -i ~/.ssh/id_ed25519`. Every run does all of the following:

- **No output cap.** Pass `--max-tokens -1`. The value stays on the eval record and the provider request omits `max_tokens`, so the only ceiling is the endpoint's own context window (`src/agentic_evals/max_tokens.py`). A positive `--max-tokens` is a hard cap. `inspect eval-retry` keeps the cap stored in the log it resumes; apply `-1` by starting a new eval.
- **Viewer forwarding.** On the pod, run `inspect view --log-dir <log-dir> --host 127.0.0.1 --port 7575` and leave it up. From the local machine, forward that port for the whole run: `ssh -N -L 7575:127.0.0.1:7575 -p 32257 -i ~/.ssh/id_ed25519 root@157.157.221.177`. Open `http://127.0.0.1:7575`. Bind to `127.0.0.1`: Inspect rejects a browser Origin of https when the server is plain HTTP, so the Runpod HTTPS proxy is not the viewer.
- **W&B stays on.** `inspect-wandb` mirrors the run to Forge project `LLM4EO/evaluations_eo`. The workspace UI is `https://forge.coreweave.com/wandb`. The `.eval` log remains the source of truth. `[tool.inspect-wandb.models]` in `pyproject.toml` stays enabled with that entity and project. On the pod, `WANDB_BASE_URL` is `https://forge.coreweave.com/api/wandb` and `WANDB_APP_URL` is `https://forge.coreweave.com/wandb`. The UI host rejects the GraphQL POST the client sends (`405 Method Not Allowed`); the API host is the one Forge's own shell config names as `BACKEND_HOST`. `WANDB_API_KEY` authenticates and is not committed. At task end, `wandb_compare` writes shared summary keys (`geobenchx/tao`, `tio`, `tem`, `param_acc`, `match`, `length_diff` when the log has those scores), the `evaluation/eval_results` table, and the `evaluation/samples` trace table onto that run, which is what makes a workspace comparison appear when several runs are selected. Leave the Weave integration off while it crashes model calls.

- `-M strict_tools=false`: Inspect's `openai-api` provider marks tools strict by default, and upstream tool schemas contain untyped arguments that OpenAI's strict mode rejects.
- Add whole-trajectory LLM judges with `-T trajectory_judges=openai/gpt-4.1-mini,<other>`; compare verdicts with `uv run python -m agentic_evals.scorers.agreement <log> --verdict <scorer>:<key> ...`.
- A text-only model on EO-Gym: `-T vision=false` records every task as not applicable.

After a major change — harness behavior, a metric, a task protocol, or how runs are launched — commit before the next experiment, so the history matches what was run. Leave secrets, `.env`, logs, and `third_party/` uncommitted.

## Research discipline

This repo implements metrics and evaluation protocols from published work, so:

- Verify any metric, loss, or protocol against its source paper before implementing. Cite the paper and section in the module docstring.
- Verify library APIs against the installed version, not from memory.
- If you implement something without a solid reference, say so explicitly and explain why.

## References

- Feng et al. *Earth-Agent: Unlocking the Full Landscape of Earth Observation with Agents.* ICLR 2026. [arXiv:2509.23141](https://arxiv.org/abs/2509.23141) · [github.com/opendatalab/Earth-Agent](https://github.com/opendatalab/Earth-Agent). Metric formulas in Appendix B.2.
- Yao et al. *ReAct: Synergizing Reasoning and Acting in Language Models.* [arXiv:2210.03629](https://arxiv.org/abs/2210.03629).
- Atrio et al. *EVE: A Domain-Specific LLM Framework for Earth Intelligence.* ACL 2026. Model card: [huggingface.co/eve-esa/EVE-Instruct](https://huggingface.co/eve-esa/EVE-Instruct).
- OpenAI. *gpt-oss-120b & gpt-oss-20b Model Card.* [arXiv:2508.10925](https://arxiv.org/abs/2508.10925) — parameter breakdown in Table 1.

- Ma et al. *EO-Gym: A Multimodal, Interactive Environment for Earth Observation Agents.* [arXiv:2605.01250](https://arxiv.org/abs/2605.01250) — Gymnasium-style workspace, 35 tools, 9,078 trajectories (1,436-task test split). Code and data: [huggingface.co/datasets/paperuploadacount/EO-Gym](https://huggingface.co/datasets/paperuploadacount/EO-Gym).
- Krechetova & Kochedykov. *GeoBenchX: Benchmarking LLMs in Agent Solving Multistep Geospatial Tasks.* SIGSPATIAL 2025. [arXiv:2503.18129](https://arxiv.org/abs/2503.18129) — 202 tasks, 24 tools, LLM-as-judge match score. [github.com/Solirinai/GeoBenchX](https://github.com/Solirinai/GeoBenchX).
- Chen et al. *Evaluating Large Language Models Trained on Code.* [arXiv:2107.03374](https://arxiv.org/abs/2107.03374) — unbiased Pass@k estimator.
- UK AI Security Institute. *Inspect AI.* [inspect.aisi.org.uk](https://inspect.aisi.org.uk/).

Deferred benchmarks, for when tasks are added:

- Shabbir et al. *ThinkGeo: Evaluating Tool-Augmented Agents for Remote Sensing Tasks.* [arXiv:2505.23752](https://arxiv.org/abs/2505.23752) — 486 tasks, 1,778 expert-verified steps, 14 tools.
- *Towards LLM Agents for Earth Observation (UnivEARTH).* [arXiv:2504.12110](https://arxiv.org/abs/2504.12110) — answer-only, 140 yes/no questions over Google Earth Engine.
- Dihan et al. *MapEval: A Map-Based Evaluation of Geo-Spatial Reasoning in Foundation Models.* ICML 2025.

## Open questions

Confirm before building on any of these:

- Whether the hands-on session pre-stages the 16.6 GB of Earth-Bench rasters and the ~47 GB of EO-Gym data on hosted infrastructure or runs on a pinned subset (see Data footprint).
- Whether a pinned EO-Gym subset can be extracted without downloading the full `EO_GYM_DATA.zip` / `image_cache.zip`.
