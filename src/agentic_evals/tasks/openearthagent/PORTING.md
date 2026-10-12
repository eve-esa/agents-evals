# OpenEarthAgent

Source: [mbzuai-oryx/OpenEarthAgent](https://github.com/mbzuai-oryx/OpenEarthAgent) at `9229c53676a22a9232e49ba1d06cad41ca629f18`. Paper: Shabbir et al., arXiv:2602.17665 (ECCV 2026). Test split: Hugging Face dataset [MBZUAI/OpenEarthAgent](https://huggingface.co/datasets/MBZUAI/OpenEarthAgent) at `f4d4e181ba2b3d2b2782c5799e8964906390e0f7` (Apache-2.0). Both pins are in `upstream.lock.toml`.

## Permission

The GitHub repository has no `LICENSE` file (GitHub reports none). Copying the evaluation prompts into this package, and cloning the tool server for a run, is authorized for this harness. Tool-server code is still **not committed**. `setup.py` clones that revision into `data/openearthagent/upstream` (gitignored).

`data/test.json` is the Hugging Face test split and is Apache-2.0.

`prompts/rs_agent_prompt.txt` and `prompts/answer_judge.txt` are the evaluation protocol text (`RS_AGENT_PROMPT`, `EVAL_PROMPT`) from the pinned revision, copied so step mode and the judge can run offline. They are protocol text, not the tool workers.

`prompts/e2e_system.txt` is written for this harness. It is not upstream text.

## What was reimplemented, not copied

- `metrics.py` reimplements `extract_action`, `extract_tool_cfg`, `extract_final_answer`, `extract_score`, `gettype`, `compare_tool_sequences`, and the step and e2e branches of `evaluate_function`.
- `tools/cache.py` reimplements `CacheManager.get_response` for the five cache-served tools. An index-layer hit returns the stored `layer_summary` when the column matches and the QGIS raster copy succeeds. A failed copy is a miss (`error_code` 1). The caller then sends that call to the live worker, as `BaseToolInferencer` does.
- `tools/session.py` reimplements the inferencer's per-item overwrite of `image`, `pre_image`, `post_image`, `gpkg`, and `geotiff`.
- `tools/mcp_server.py` is a stdio MCP proxy. It speaks the controller protocol (`POST /list_models`, `POST /get_worker_address`, `POST /worker_generate`). It does not import `tool_server/`.
- `tools/catalog.py` is a schema list paraphrased from the system prompt so the agent can call tools natively. It is not a copy of a schema file.

Scores are for this proxy and this reimplementation, not for the authors' three conda environments.

## Protocol differences

- **End-to-end tool calling is native.** Upstream asks the model to emit `{"thought", "actions"}` as text and parses it with a regex. This harness's LangGraph agent uses MCP tool calls. Step mode still uses the text protocol (`RS_AGENT_PROMPT` and `extract_action`), because there is no agent loop there.
- **`Terminate` is not a tool.** The final assistant message is the answer. If that message is still a `Terminate` JSON object, the `ans` field is the answer. Gold `Terminate` is `reference_answer`, not part of `reference_trajectory`.
- **Vision.** Tasks with an image attach the file as an image part when `vision=true` (the default). `vision=false` marks those 533 tasks not applicable and does not score them. The paper's text-only runs put no pixels in the prompt and still scored them; this follows the EO-Gym precedent instead.
- **Cache then worker.** For `GetAreaBoundary`, `AddPoisLayer`, `ComputeDistance`, `AddIndexLayer`, and `ComputeIndexChange`, a cache hit (`error_code` 0) is the observation. Any other result is discarded and the same call goes to the live worker. Every other tool goes straight to the worker.
- **End-to-end start.** `regime=e2e` writes `data/openearthagent/server.yaml` from the clone's `all_service_example_local.yaml` and runs `conda run --no-capture-output -n tool-server-e1 python start_server_local.py`. The eval waits until `POST /list_models` lists those workers, then sends SIGINT to that process group when the eval process exits. An already-running controller is left running. Step mode does not start it. The host must already have Conda environments `tool-server-e1` and `tool-server-e2`, the weight files that YAML names, `OPENEARTHAGENT_SERPER_API_KEY`, and `OPENEARTHAGENT_GEE_KEY`. `GetBboxFromGeotiff` and `DisplayOnGeotiff` are not in that YAML, so this launch does not register them.
- **Guards.** Upstream raises `IndexError` on `API_names_gt[-2]` when the gold has only `Terminate`, and `last_tool_in_gen` raises when every call is `Terminate`. Both are treated as not a generation task. A judge call that fails returns `nan` rather than failing the sample.
- **`n_steps` in the scorer** is the number of tool calls in our trace. Upstream's `avg_steps_taken` is the mean of `current_round` (model turns that did not finish the item). `summarize_e2e` takes `n_rounds` from the caller so the differential test can pass `current_round`.
- **Same-order.** Table 4's caption says "exact sequence match". Section 4.1 says same-order is the correct sequence and the right frequency. The pinned `compare_tool_sequences` is multiset coverage plus the gold names as a subsequence, and its docstring allows extra predicted calls. This package follows that function.
- **Logged scores are fractions in `[0, 1]`.** `evaluate_function` multiplies F1 and both answer accuracies by 100 and leaves the tool-order ratios as fractions. Tables 3 and 4 print the tool-order ratios as percentages as well. Multiply a logged rate by 100 to read those tables.
- **One scorer per judge.** The official judge is Inspect's `grader` role (`gpt-4o-mini`, temperature 0, `max_tokens` 150, `EVAL_PROMPT`). Extra judges are separate scorers and are not folded into `summ` or `ans`. `avg_summ` and `avg_ans` are the unweighted mean of those judges, each judge once (the same construction as GeoBenchX Section 5.2). Tables 3 and 4 do not define that mean.
- **Controller timeouts** use httpx, not `SIGALRM`. The client does not trust env proxies, matching upstream `disable_proxy`.
- **Image directory.** Upstream `config.yaml` says `data/test_images`. This package stores the Hugging Face `test/` folder at `data/openearthagent/test`.

Benchmark notes stay in this file. `AGENTS.md` is the short rules file and does not get a per-benchmark section.
