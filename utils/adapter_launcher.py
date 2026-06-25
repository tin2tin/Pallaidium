"""Launch and manage a bundled reference adapter from inside Blender.

The reference adapters in ``remote_backends/`` are stdlib-only HTTP servers, so
Pallaidium can start one with Blender's own Python — no console, no ``pip``. This
module:

  * discovers adapters by their ``*.manifest.json`` sidecar (:func:`discover_adapters`);
  * resolves Blender's bundled Python (:func:`resolve_python`);
  * starts one adapter as a subprocess on a free port, waits for ``/v1/health``,
    and returns its base URL (:func:`start_adapter`);
  * stops / reports / restarts it (:func:`stop_adapter`, :func:`adapter_status`,
    :func:`restart_adapter`).

Exactly **one** managed adapter runs at a time (single active backend). The
subprocess is always stopped on Blender exit / add-on unregister so it never
lingers. The add-on stays provider-agnostic — it only ever talks HTTP to the
adapter's URL, which the launcher writes into ``prefs.remote_backend_url``.
"""

import os
import sys
import json
import time
import socket
import shutil
import atexit
import subprocess
import urllib.request
from pathlib import Path

_ADDON_ROOT = Path(__file__).resolve().parent.parent
REMOTE_BACKENDS_DIR = _ADDON_ROOT / "remote_backends"
WORKFLOWS_DIR = REMOTE_BACKENDS_DIR / "comfyui_workflows"

# State of the single managed adapter subprocess.
_PROC: "subprocess.Popen | None" = None
_STATE: dict = {
    "adapter_id": None, "port": None, "base_url": None,
    "log_path": None, "manifest": None, "config": None, "external": False,
}


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------
def discover_adapters() -> list:
    """Return adapter manifests found in remote_backends/, sorted for the enum.

    Each manifest is the parsed ``<name>.manifest.json`` plus two resolved paths:
    ``_path`` (the adapter script) and ``_manifest`` (the json itself).
    """
    out = []
    if REMOTE_BACKENDS_DIR.is_dir():
        for p in sorted(REMOTE_BACKENDS_DIR.glob("*.manifest.json")):
            try:
                m = json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:  # noqa: BLE001
                print(f"[pallaidium] bad adapter manifest {p.name}: {e}")
                continue
            entry = m.get("entry")
            if not entry or not (REMOTE_BACKENDS_DIR / entry).is_file():
                print(f"[pallaidium] manifest {p.name}: missing entry script {entry!r}")
                continue
            m["_path"] = str(REMOTE_BACKENDS_DIR / entry)
            m["_manifest"] = str(p)
            out.append(m)
    out.sort(key=lambda m: (m.get("order", 99), m.get("label", "")))
    return out


def get_manifest(adapter_id: str):
    for m in discover_adapters():
        if m.get("id") == adapter_id:
            return m
    return None


def build_config_from_prefs(manifest: dict, prefs) -> dict:
    """Resolve each manifest config field to a value, preferring the named pref.

    A field declares ``pref`` (a preference property name) and/or a ``default``.
    Returns ``{field_key: value}`` for :func:`start_adapter` to turn into env
    vars / CLI args.
    """
    cfg = {}
    for f in manifest.get("config_fields", []):
        key = f.get("key")
        if not key:
            continue
        val = getattr(prefs, f["pref"], None) if f.get("pref") else None
        if val in (None, ""):
            val = f.get("default", "")
        cfg[key] = val
    return cfg


# --------------------------------------------------------------------------
# Python interpreter
# --------------------------------------------------------------------------
def resolve_python():
    """Path to Blender's bundled Python (NOT the Blender binary), or None.

    The adapters are plain scripts, so any CPython works. We must avoid
    ``sys.executable`` because inside Blender that is ``blender.exe`` — launching
    it with a script path would open Blender, not run the script. Prefer the
    interpreter shipped under ``sys.prefix`` (``.../python/bin/python.exe``).
    """
    names = ("python.exe", "python3.exe") if os.name == "nt" else ("python3", "python")
    dirs = [os.path.join(sys.prefix, "bin"), sys.prefix,
            os.path.join(sys.exec_prefix, "bin"), sys.exec_prefix]
    for d in dirs:
        for n in names:
            c = os.path.join(d, n)
            if os.path.isfile(c):
                return c
    base = getattr(sys, "_base_executable", "") or ""
    if base and "python" in os.path.basename(base).lower() and os.path.isfile(base):
        return base
    return shutil.which("python") or shutil.which("python3")


# --------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------
def _log_dir() -> str:
    try:
        import bpy
        d = os.path.join(bpy.utils.user_resource("DATAFILES", create=True), "Pallaidium")
    except Exception:  # noqa: BLE001 — outside Blender (tests)
        d = os.path.join(os.path.expanduser("~"), ".pallaidium")
    os.makedirs(d, exist_ok=True)
    return d


def _free_port(preferred=None) -> int:
    if preferred:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", int(preferred)))
                return int(preferred)
            except OSError:
                pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _health_ok(base_url: str, timeout=1.5) -> bool:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/v1/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def _no_window():
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def _tail(path: str, n=25) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except Exception:  # noqa: BLE001
        return ""


def is_running() -> bool:
    return _PROC is not None and _PROC.poll() is None


def start_adapter(manifest: dict, config: dict | None = None,
                  port=None, host="127.0.0.1", wait=25.0) -> str:
    """Launch ``manifest``'s adapter, wait for health, return its base URL.

    Stops any previously-managed adapter first (single active backend). Config
    fields become both an env var (``key``) and, if the manifest gives an
    ``arg``, a CLI flag. Raises RuntimeError with the log tail on failure.
    """
    global _PROC
    config = config or {}
    stop_adapter()

    port = _free_port(port or manifest.get("default_port", 8000))
    base_url = f"http://{host}:{port}"

    py = resolve_python()
    if not py:
        raise RuntimeError(
            "Could not locate Blender's bundled Python interpreter to launch the "
            "adapter. Use the 'Custom URL' backend and start an adapter manually.")

    argv = [py, manifest["_path"], "--port", str(port), "--host", host]
    env = dict(os.environ)
    for f in manifest.get("config_fields", []):
        key, arg = f.get("key"), f.get("arg")
        val = config.get(key, f.get("default", ""))
        val = "" if val is None else str(val)
        if key:
            env[key] = val
        if arg and val != "":
            argv += [arg, val]

    log_path = os.path.join(_log_dir(), f"adapter_{manifest['id']}.log")
    logf = open(log_path, "w", encoding="utf-8")
    logf.write(f"$ {' '.join(argv)}\n\n")
    logf.flush()
    _PROC = subprocess.Popen(
        argv, cwd=str(REMOTE_BACKENDS_DIR), env=env,
        stdout=logf, stderr=subprocess.STDOUT, creationflags=_no_window(),
    )
    _STATE.update(adapter_id=manifest.get("id"), port=port, base_url=base_url,
                  log_path=log_path, manifest=manifest, config=config, external=False)

    deadline = time.time() + wait
    while time.time() < deadline:
        if _PROC.poll() is not None:
            raise RuntimeError(
                f"Adapter '{manifest.get('id')}' exited immediately.\n"
                f"Log: {log_path}\n--- last lines ---\n{_tail(log_path)}")
        if _health_ok(base_url):
            return base_url
        time.sleep(0.25)
    raise RuntimeError(
        f"Adapter '{manifest.get('id')}' did not respond on {base_url} within "
        f"{wait:.0f}s.\nLog: {log_path}\n--- last lines ---\n{_tail(log_path)}")


def stop_adapter() -> None:
    """Terminate the managed adapter subprocess if one is running."""
    global _PROC
    if _PROC is not None and _PROC.poll() is None:
        try:
            _PROC.terminate()
            try:
                _PROC.wait(timeout=5)
            except Exception:  # noqa: BLE001
                _PROC.kill()
        except Exception:  # noqa: BLE001
            pass
    _PROC = None
    _STATE.update(adapter_id=None, port=None, base_url=None,
                  manifest=None, config=None, external=False)


def restart_adapter() -> str:
    """Restart the current adapter with the same manifest/config/port.

    Used after importing a ComfyUI workflow so the adapter re-scans the folder.
    """
    manifest, config, port = _STATE["manifest"], _STATE["config"], _STATE["port"]
    if not manifest:
        raise RuntimeError("No adapter is running to restart.")
    return start_adapter(manifest, config, port=port)


def adapter_status() -> dict:
    running = is_running()
    return {
        "running": running,
        "adapter_id": _STATE["adapter_id"],
        "port": _STATE["port"],
        "base_url": _STATE["base_url"],
        "pid": _PROC.pid if running else None,
        "log_path": _STATE["log_path"],
    }


# --------------------------------------------------------------------------
# ComfyUI workflow helpers (for the in-Blender Import / Open-folder operators)
# --------------------------------------------------------------------------
def workflows_dir() -> str:
    WORKFLOWS_DIR.mkdir(parents=True, exist_ok=True)
    return str(WORKFLOWS_DIR)


def import_workflow(src_path: str) -> str:
    """Copy an API-format workflow JSON into comfyui_workflows/. Returns dest.

    Validates it parses as a non-empty graph (accepts a UI export that wraps the
    graph under ``"prompt"``). The file stem becomes the model id.
    """
    src = Path(src_path)
    if not src.is_file():
        raise RuntimeError(f"File not found: {src_path}")
    try:
        graph = json.loads(src.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Not valid JSON: {e}") from e
    if isinstance(graph, dict) and "prompt" in graph and "nodes" not in graph:
        graph = graph["prompt"]
    if not isinstance(graph, dict) or not graph:
        raise RuntimeError("Not an API-format ComfyUI graph (export with 'Save (API Format)').")

    dest = Path(workflows_dir()) / src.name
    if not dest.name.endswith(".json"):
        dest = dest.with_suffix(".json")
    shutil.copyfile(str(src), str(dest))
    return str(dest)


# Always tear the adapter down when Blender quits.
atexit.register(stop_adapter)
