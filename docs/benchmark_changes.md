# Changes from the original benchmarks

This repository runs public benchmarks, so its scores should match each benchmark's published protocol. Where that protocol has a known bug, we fix it here and say so. Each section below lists what differs from the pinned upstream revision, and why. The revisions are in `src/agentic_evals/upstream.lock.toml`.

How a tool port differs file by file (library swaps, paths, licences) is in each task's `PORTING.md`. This page covers what changes a score.

## Earth-Bench

Upstream: [opendatalab/Earth-Agent](https://github.com/opendatalab/Earth-Agent) at `ca1dfae`. Paper: Feng et al., ICLR 2026, [arXiv:2509.23141](https://arxiv.org/abs/2509.23141), Appendix B.2. The metric functions reproduce all 26 published score files (`tests/test_earth_bench_metrics.py`).

- **Packed gold trajectory ([issue 13](https://github.com/opendatalab/Earth-Agent/issues/13)).** In 53 of the 99 Spectrum tasks, a gold step passes a list of files to a tool parameter typed `str`. The published tools reject that list, so a correct agent needs one call per file where the gold shows one call. Efficiency is then 2–3× too high for every model that uses tools, and TEM and parameter accuracy are capped below 1. We split those gold steps into one call per file before scoring (`metrics.expand_scalar_gold`). The agent's own trace is not changed. 8 steps whose lists differ in length stay as published. Products and RGB have no packed steps.
- **GPT-4o answer polishing ([issue 14](https://github.com/opendatalab/Earth-Agent/issues/14)).** Upstream rewrites each reply with GPT-4o (`process/polish_answer.py`) before extracting the letter. Across the published runs this adds a choice letter to 10.4% of replies that contained none. We extract the letter from the agent's own reply, with upstream's extraction rule.
- **Missing gold for question 43 ([issue 11](https://github.com/opendatalab/Earth-Agent/issues/11)).** The gold file has no entry for question 43, and question 164 has a null answer. Their trajectory metrics are left unscored (NaN, out of the means), and accuracy uses the answer in `question.json`. Upstream never scores question 43 and always marks question 164 wrong.
- **Two answer keys.** `question.json` and the gold file disagree on 28 tasks. The target is the gold file, as in upstream's evaluator. `accuracy_other_key` reports accuracy against `question.json`, so the disagreement stays visible.
- **Runs that hit the recursion limit.** Upstream logs such a run with no tool calls, so every trajectory metric is 0 and the answer is an error. We keep the calls made before the limit and score them.
- **Prompts.** Prompts are rebuilt from the pinned `question.json`. For 16 tasks the published runs used a `benchmark/supported_data/` path that no longer exists, and 12 of those had their question text revised afterwards. Published scores on those tasks are not comparable with ours.
- **Tools.** GDAL is replaced with rasterio, so scores are for this port and the library versions in the `earth-bench` extra.

The task argument `upstream_compat: true` scores against the published gold, with upstream's TEM quirk on an empty gold trajectory. Use it only to reproduce the published trajectory scores. It does not add the GPT-4o polish step.

## GeoBenchX

Upstream: [Solirinai/GeoBenchX](https://github.com/Solirinai/GeoBenchX) at `0edf610`. Paper: Krechetova & Kochedykov, SIGSPATIAL 2025, [arXiv:2503.18129](https://arxiv.org/abs/2503.18129), Section 5.

- **Length difference follows the code.** The paper describes `length_diff` relative to the candidate's length. Upstream's `generate_eval_stats` divides by the mean reference length, and the paper's tables were computed with that code. We follow the code.
- **Step metrics are an addition.** `geobenchx_steps` reports TAO, TIO, TEM and parameter accuracy with Earth-Bench's formulas, taking the best match over a task's reference solutions. These numbers are not part of the GeoBenchX protocol and never enter the official score.
- **Judges are kept separate.** The official score is the share of no match, partial and match, one row per judge. The panel number is the unweighted mean of the judges' shares (Section 5.2), with no majority vote. The paper's judges include retired checkpoints. We do not substitute another model for a retired one; record the judge with every result.
- **Tool names.** Some hosts return `reject_task` with trailing whitespace and then reject the next request with HTTP 400. The agent strips that whitespace before the call.
- **Tools.** GDAL is imported only by `generate_contours_display`. The other 23 tools run without it.

## EO-Gym

Upstream: [paperuploadacount/EO-Gym](https://huggingface.co/datasets/paperuploadacount/EO-Gym) at `d168c28`. Paper: [arXiv:2605.01250](https://arxiv.org/abs/2605.01250), Section 5 and Appendix D. The metric functions are checked against upstream's `agent_passk.py` (`tests/test_eo_gym_metrics.py`).

- **Pass@k follows the code; the paper's estimator is also reported.** The released code judges trajectories in order and stops at the first correct one, so Pass@k is 1 when that trajectory is among the first k. The paper describes the unbiased estimator of Chen et al. (2021). We report the code's value as Pass@k and the estimator as `unbiased_pass@k`.
- **Image references.** The prompt gives images as the dataset's relative `EO_GYM_DATA/...` paths, as upstream's driver does for vLLM runs. Upstream's OpenAI default is base64 data URLs, which would paste every image into the text a second time.
- **Text-only models.** EO-Gym puts imagery in the prompt. With the task argument `vision: false` no agent runs, and every task is recorded as not applicable instead of scored as wrong.
- **Tool mode.** The released code only has verified tools (perception backed by ground truth). `tool_mode="unverified"` is rejected rather than silently run as verified.
- **Server.** The copied server runs on CPU, preloads no models, and reads imagery from `data/eo_gym`. Scores are for this copy and the library versions in `server/pyproject.toml`.
