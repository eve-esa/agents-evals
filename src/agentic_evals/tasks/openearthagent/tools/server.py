"""Start OpenEarthAgent's tool server for one end-to-end eval.

Upstream launches the controller and every worker with
``start_server_local.py`` (Shabbir et al., the ``tool_server`` launcher at
the revision in ``upstream.lock.toml``). That script blocks until SIGINT,
then shuts the workers down. This module renders a local copy of
``all_service_example_local.yaml``, runs the script inside the
``tool-server-e1`` Conda environment, and waits until ``POST /list_models``
lists every worker in that file.

The Conda environments, CUDA weights, Serper key, and Earth Engine key are
installed on the host. This module does not install them and does not copy
``tool_server/`` into the package. Step mode does not call it.

``GetBboxFromGeotiff`` and ``DisplayOnGeotiff`` are absent from the upstream
YAML, so this launch does not register them.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any

import httpx

from agentic_evals.paths import OPENEARTHAGENT_DATA, OPENEARTHAGENT_UPSTREAM

EXAMPLE_REL = Path(
    "tool_server/tool_workers/scripts/launch_scripts/config/all_service_example_local.yaml"
)
LAUNCH_DIR = Path("tool_server/tool_workers/scripts/launch_scripts")
ENV_E1 = "tool-server-e1"
ENV_E2 = "tool-server-e2"
# Model files the example YAML points at under the OpenEarthAgent/ prefix.
WEIGHTS = (
    "models/groundingdino_swint_ogc.pth",
    "models/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "models/src/sam-2/checkpoints/sam2.1_hiera_large.pt",
    "models/src/sam-2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
    "models/lae_dino_swint_lae1m-28ca3a15.pth",
    "models/LAE-DINO/mmdetection_lae/configs/lae_dino/lae_dino_swin-t_pretrain_LAE-1M.py",
)
_CUDA = re.compile(r'cuda_visible_devices:\s*"[^"]*"')
_WORKER = re.compile(r"worker_name:\s*(\S+)")
_LOCK = threading.Lock()
_PROCESS: subprocess.Popen[bytes] | None = None
_LOG: Any = None
_OWNED = False
_ATEXIT = False


def render_server_yaml(
    example: str,
    clone: Path,
    *,
    cuda: str,
    serper_key: str,
    gee_key: Path,
    gee_account: str,
) -> str:
    """Rewrite the upstream example so paths, the GPU, and secrets are local.

    The example prefixes every path with ``OpenEarthAgent/``, which the
    README says to replace with the checkout. GPU ids in that file pin some
    workers to device 3; ``cuda`` replaces every ``cuda_visible_devices``.
    """
    if '"' in cuda or "\n" in cuda:
        raise RuntimeError("OPENEARTHAGENT_CUDA_DEVICES must be a single device id.")
    for secret in (serper_key, gee_account):
        if '"' in secret or "\n" in secret:
            raise RuntimeError("OpenEarthAgent server secrets cannot contain quotes or newlines.")
    root = clone.resolve().as_posix().rstrip("/")
    text = example.replace("OpenEarthAgent/", f"{root}/")
    text = _CUDA.sub(f'cuda_visible_devices: "{cuda}"', text)
    text = text.replace('api-key: "<SERPER_API_KEY>"', f'api-key: "{serper_key}"')
    text = text.replace('service_account: "<service_account>"', f'service_account: "{gee_account}"')
    text = text.replace("key_file: gee_key/<key_file>", f"key_file: {gee_key.resolve().as_posix()}")
    return text


def expected_workers(yaml_text: str) -> list[str]:
    """Worker names the controller must list. The controller itself is not one."""
    return [name for name in _WORKER.findall(yaml_text) if name != "controller"]


def render_server_config(clone: Path, *, environ: dict[str, str] | None = None) -> str:
    """Read the example from ``clone`` and apply this process's environment.

    ``OPENEARTHAGENT_SERPER_API_KEY`` and ``OPENEARTHAGENT_GEE_KEY`` are
    required. ``OPENEARTHAGENT_CUDA_DEVICES`` defaults to ``0``. The Earth
    Engine JSON must contain ``client_email``; that value is
    ``service_account``.
    """
    env = os.environ if environ is None else environ
    example_path = clone / EXAMPLE_REL
    if not example_path.is_file():
        raise RuntimeError(
            f"OpenEarthAgent launch config not found at {example_path}. "
            "Run `uv run python scripts/setup_benchmark.py openearthagent`."
        )
    serper = env.get("OPENEARTHAGENT_SERPER_API_KEY", "").strip()
    gee = env.get("OPENEARTHAGENT_GEE_KEY", "").strip()
    if not serper:
        raise RuntimeError(
            "OPENEARTHAGENT_SERPER_API_KEY is not set. GoogleSearch cannot start."
        )
    if not gee:
        raise RuntimeError("OPENEARTHAGENT_GEE_KEY is not set. AddIndexLayer cannot start.")
    gee_path = Path(gee).expanduser()
    if not gee_path.is_file():
        raise RuntimeError(f"OPENEARTHAGENT_GEE_KEY file not found: {gee_path}")
    try:
        account = json.loads(gee_path.read_text(encoding="utf-8")).get("client_email")
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"OPENEARTHAGENT_GEE_KEY is not a JSON service account: {gee_path}") from exc
    if not isinstance(account, str) or not account:
        raise RuntimeError(f"{gee_path} has no client_email.")
    cuda = env.get("OPENEARTHAGENT_CUDA_DEVICES", "0").strip() or "0"
    return render_server_yaml(
        example_path.read_text(encoding="utf-8"),
        clone,
        cuda=cuda,
        serper_key=serper,
        gee_key=gee_path,
        gee_account=account,
    )


def missing_weights(clone: Path) -> list[Path]:
    """Weight and config files the example YAML names, when they are absent."""
    return [clone / relative for relative in WEIGHTS if not (clone / relative).is_file()]


def conda_env_names() -> set[str]:
    """Names from ``conda env list --json``. Raises when conda is missing."""
    conda = shutil.which("conda")
    if conda is None:
        raise RuntimeError(
            "conda is not on PATH. OpenEarthAgent's tool server needs the "
            f"{ENV_E1} and {ENV_E2} environments."
        )
    try:
        completed = subprocess.run(
            [conda, "env", "list", "--json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"conda env list failed: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(f"conda env list failed: {detail}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("conda env list did not return JSON.") from exc
    names: set[str] = set()
    for entry in payload.get("envs") or []:
        names.add(Path(str(entry)).name)
    return names


def _is_local(url: str) -> bool:
    host = urllib.parse.urlparse(url).hostname
    return host in {"127.0.0.1", "localhost", "::1"}


def _listed(url: str) -> list[str] | None:
    """Model names, or None when the controller does not answer."""
    try:
        response = httpx.post(url.rstrip("/") + "/list_models", timeout=5, trust_env=False)
        response.raise_for_status()
        models = response.json().get("models")
    except (httpx.HTTPError, ValueError):
        return None
    if not isinstance(models, list):
        return None
    return [str(name) for name in models]


def _log_tail(path: Path, lines: int = 40) -> str:
    if not path.is_file():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def _stop() -> None:
    """SIGINT the server we started, then SIGTERM if it is still alive."""
    global _PROCESS, _OWNED, _LOG
    proc = _PROCESS
    owned = _OWNED
    if owned and proc is not None and proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGINT)
            proc.wait(timeout=30)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
    _OWNED = False
    if _LOG is not None:
        _LOG.close()
        _LOG = None


def _register_stop() -> None:
    global _ATEXIT
    if not _ATEXIT:
        atexit.register(_stop)
        _ATEXIT = True


def _record(url: str, models: list[str], *, started: bool) -> dict[str, Any]:
    # Imported lazily so tools/__init__ can stay free of this module at import.
    from agentic_evals.tasks.openearthagent import tools as tools_mod

    record = {
        "mode": "controller",
        "controller_url": url,
        "models": models,
        "started_by_eval": started,
    }
    tools_mod._PROBE[url] = dict(record)
    return dict(record)


def ensure_controller(url: str, *, clone: Path | None = None) -> dict[str, Any]:
    """Return the controller record, starting the upstream server when it is local and down.

    A controller that already lists tools is left running and is not stopped
    at process exit. A remote URL that does not answer is an error: this
    function only spawns the server for localhost.
    """
    listed = _listed(url)
    if listed:
        return _record(url, listed, started=_OWNED)
    if not _is_local(url):
        raise RuntimeError(
            f"OpenEarthAgent tool controller not reachable at {url}. "
            "A remote controller is not started by this eval."
        )
    with _LOCK:
        listed = _listed(url)
        if listed:
            return _record(url, listed, started=_OWNED)
        return _start(url, clone or OPENEARTHAGENT_UPSTREAM)


def _start(url: str, clone: Path) -> dict[str, Any]:
    global _PROCESS, _OWNED, _LOG
    launch = clone / LAUNCH_DIR / "start_server_local.py"
    if not launch.is_file():
        raise RuntimeError(
            f"OpenEarthAgent launcher not found at {launch}. "
            "Run `uv run python scripts/setup_benchmark.py openearthagent`."
        )
    names = conda_env_names()
    missing_envs = [name for name in (ENV_E1, ENV_E2) if name not in names]
    if missing_envs:
        raise RuntimeError(
            "Missing Conda environments: "
            + ", ".join(missing_envs)
            + ". Install the OpenEarthAgent tool-server environments before an end-to-end run."
        )
    absent = missing_weights(clone)
    if absent:
        raise RuntimeError(
            "OpenEarthAgent model files are missing:\n" + "\n".join(str(path) for path in absent)
        )
    rendered = render_server_config(clone)
    wanted = expected_workers(rendered)
    if not wanted:
        raise RuntimeError("The OpenEarthAgent server config lists no workers.")
    OPENEARTHAGENT_DATA.mkdir(parents=True, exist_ok=True)
    config_path = OPENEARTHAGENT_DATA / "server.yaml"
    log_path = OPENEARTHAGENT_DATA / "server.log"
    config_path.write_text(rendered, encoding="utf-8")
    conda = shutil.which("conda")
    if conda is None:
        raise RuntimeError(
            "conda is not on PATH. OpenEarthAgent's tool server needs the "
            f"{ENV_E1} and {ENV_E2} environments."
        )
    log_file = log_path.open("w", encoding="utf-8")
    _LOG = log_file
    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            [
                conda,
                "run",
                "--no-capture-output",
                "-n",
                ENV_E1,
                "python",
                "start_server_local.py",
                "--config",
                str(config_path),
            ],
            cwd=launch.parent,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as exc:
        _stop()
        raise RuntimeError(f"Failed to start the OpenEarthAgent tool server: {exc}") from exc
    _PROCESS = proc
    _OWNED = True
    _register_stop()
    timeout = float(os.environ.get("OPENEARTHAGENT_SERVER_TIMEOUT", "900"))
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"OpenEarthAgent tool server exited with status {proc.returncode}.\n{_log_tail(log_path)}"
                )
            listed = _listed(url)
            if listed is not None and set(wanted) <= set(listed):
                return _record(url, listed, started=True)
            time.sleep(5)
    except Exception:
        _stop()
        raise
    missing = sorted(set(wanted) - set(_listed(url) or []))
    _stop()
    raise RuntimeError(
        "OpenEarthAgent tool server did not register: "
        + ", ".join(missing)
        + f"\n{_log_tail(log_path)}"
    )
