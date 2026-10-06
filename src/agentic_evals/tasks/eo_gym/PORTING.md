# EO-Gym server copy

Source: [paperuploadacount/EO-Gym](https://huggingface.co/datasets/paperuploadacount/EO-Gym) at `d168c28e870ccbf75f63ce60d087420a6babed9e` (CC-BY-4.0; see `LICENSE`).

`server/eo_gym/` is that revision's tool server (`src/eo_gym`), without `trainer/` and `runtime/gym_envs/`. The test split is `data/eo_gym_test_set.jsonl`. Tool schemas exported from `AVAILABLE_TOOL_CLASSES` are `tools/tools.json`. The semantic judge template is `prompts/semantic_final_answer.txt`. The differential-test oracle (`agent_passk.py` and the prompt package it imports) is `tests/fixtures/eo_gym/upstream`.

## Modifications

- The package is flattened from `src/eo_gym` to `server/eo_gym`, and setuptools looks in `.` instead of `src`. `config_loader.REPO_ROOT` and the LLM log path count three parents (the `server/` directory) instead of four.
- `config/paths.toml` points imagery at `data/eo_gym/EO_GYM_DATA`, the image cache at `data/eo_gym/image_cache`, and the lookup tables at `data/eo_gym/index_files`. Relative `EO_GYM_DATA/...` paths resolve under `data_dir`. `RUN_ON_LOCAL` defaults to true so those paths are used; `EO_GYM_RUN_ON_LOCAL=0` restores the upstream remote-URL default.
- The server defaults to `device = "cpu"` and preloads no models. Flags on `python -m eo_gym.server` still override that.
- The image archives and `index_files/` are not in git. `tasks/eo_gym/setup.py` downloads them.

Scores from this server are for this copy and the library versions in `server/pyproject.toml`.
