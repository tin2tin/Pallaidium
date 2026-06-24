"""Reference adapter: Pallaidium Backend Contract v0.1  ->  ComfyUI.

A small local service that implements the contract and forwards generation to a
running **ComfyUI** server. Pallaidium stays provider-agnostic; all ComfyUI
specifics (the workflow graphs, node ids, checkpoint names) live here.

Unlike fal/Replicate, ComfyUI has no per-task REST endpoints. It runs *workflow
graphs*: you POST a graph (API-format JSON) to `/prompt`, poll `/history/{id}`
for the result, and fetch bytes from `/view`. So this adapter maps each contract
model to a **workflow template** and injects prompt/seed/size/steps/etc. into it.

ComfyUI API used:
    POST /prompt              {prompt: <graph>, client_id}  -> {prompt_id}
    GET  /history/{id}        -> {<id>: {outputs: {node: {images:[...]}}}}
    GET  /queue               -> {queue_running, queue_pending}  (progress)
    POST /upload/image        multipart "image"  -> {name, subfolder, type}
    GET  /view?filename=&subfolder=&type=output  -> image bytes

Two ways to add models:

  1. **Workflow files (recommended).** Drop ComfyUI **API-format** workflow
     exports (Settings -> "Enable Dev mode", then Save (API Format)) into the
     `comfyui_workflows/` folder next to this file. Each `<id>.json` becomes a
     model. The adapter detects its media type from the output node and patches
     your prompt / negative / seed / size / init image into nodes **by title**
     (rename a node in ComfyUI: right-click -> Title). See that folder's README.
     Optional `<id>.meta.json` overrides the display name / type / hints.

  2. **Built-in templates.** The hardcoded MODELS below cover a standard
     checkpoint txt2img / img2img (SD1.5 / SDXL) and **LTX-Video** txt2video /
     i2v, as reference examples.

Requirements in ComfyUI: the models each workflow/template uses, plus the
**VideoHelperSuite** custom node (`VHS_VideoCombine`) for mp4 video output.

Run:
    pip install -r requirements.txt
    set COMFYUI_URL=http://127.0.0.1:8188   # default; PowerShell: $env:COMFYUI_URL=...
    uvicorn comfyui_adapter:app --port 8000

Then in Pallaidium preferences set Remote Backend URL = http://localhost:8000,
switch Model Source to Remote, and click "Refresh Remote Models".
"""

import os
import json
import copy
import uuid
import base64
from pathlib import Path

import httpx
from fastapi import FastAPI, UploadFile, Form, Response
from fastapi.responses import JSONResponse

app = FastAPI(title="Pallaidium <-> ComfyUI adapter")

CONTRACT_VERSIONS = ["v0.1"]
COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
CLIENT_ID = "pallaidium-adapter"

# Map contract model ids -> ComfyUI checkpoint + media type + capability hints.
# `ckpt` must be a checkpoint file present in ComfyUI's models/checkpoints folder.
# `builder` selects which workflow template to use.
MODELS = {
    "sdxl": {
        "type": "image",
        "modes": ["t2i", "i2i"],
        "ckpt": "sd_xl_base_1.0.safetensors",
        "builder": "checkpoint",
        "display_name": "SDXL Base (ComfyUI)",
        "default_steps": 25,
        "default_guidance": 7.0,
        "max_ref_images": 1,        # single init image for img2img
        "max_width": 1024,
        "max_height": 1024,
    },
    "sd15": {
        "type": "image",
        "modes": ["t2i", "i2i"],
        "ckpt": "v1-5-pruned-emaonly.safetensors",
        "builder": "checkpoint",
        "display_name": "SD 1.5 (ComfyUI)",
        "default_steps": 20,
        "default_guidance": 7.0,
        "max_ref_images": 1,
        "max_width": 512,
        "max_height": 512,
    },
    "ltxv": {
        "type": "video",
        "modes": ["t2v", "i2v"],
        "ckpt": "ltx-video-2b-v0.9.5.safetensors",
        "builder": "ltxv",
        "display_name": "LTX-Video 2B (ComfyUI)",
        "default_steps": 30,
        "default_guidance": 3.0,
        "max_ref_images": 1,        # single first-frame image for i2v
        "max_width": 768,
        "max_height": 512,
    },
}

# In-memory stores (use a real store in production).
_FILES: dict[str, tuple[bytes, str]] = {}   # file_id -> (bytes, content_type)
_JOBS: dict[str, dict] = {}                 # job_id  -> {prompt_id, model_id, kind}

# Loaded workflow graphs for `builder == "workflow"` models, keyed by model id.
_WORKFLOWS: dict[str, dict] = {}
WORKFLOWS_DIR = Path(__file__).with_name("comfyui_workflows")

# ComfyUI output node class -> media type, and which history key holds its files.
_OUTPUT_NODES = {
    "SaveImage": ("image", "images"),
    "PreviewImage": ("image", "images"),
    "VHS_VideoCombine": ("video", "gifs"),
    "SaveAnimatedWEBP": ("video", "gifs"),
    "SaveWEBM": ("video", "gifs"),
    "SaveVideo": ("video", "gifs"),
    "SaveAudio": ("audio", "audio"),
    "SaveAudioMP3": ("audio", "audio"),
    "SaveAudioOpus": ("audio", "audio"),
}

# Nodes that load a source video (for video-edit / character-replacement graphs).
_VIDEO_LOAD_CLASSES = {
    "VHS_LoadVideo", "VHS_LoadVideoPath", "VHS_LoadVideoUpload",
    "VHS_LoadVideoFFmpeg", "LoadVideo",
}


def _discover_workflows() -> None:
    """Scan comfyui_workflows/ for API-format graphs and register each as a model.

    Media type/modes/hints are auto-detected from the graph; an optional sidecar
    `<id>.meta.json` overrides any of: display_name, type, modes, max_ref_images,
    default_steps, default_guidance, max_width, max_height, description.
    """
    if not WORKFLOWS_DIR.is_dir():
        return
    for path in sorted(WORKFLOWS_DIR.glob("*.json")):
        if path.name.endswith(".meta.json"):
            continue
        mid = path.stem
        try:
            graph = json.loads(path.read_text(encoding="utf-8"))
            # Accept either a bare API graph or a UI export wrapping it under "prompt".
            if isinstance(graph, dict) and "prompt" in graph and "nodes" not in graph:
                graph = graph["prompt"]
            if not isinstance(graph, dict) or not graph:
                raise ValueError("not a non-empty API-format graph")
        except Exception as e:  # noqa: BLE001
            print(f"[comfyui] skip {path.name}: {e}")
            continue

        mtype, _ = _detect_output(graph)
        n_images = _count_class(graph, "LoadImage")
        n_videos = sum(_count_class(graph, c) for c in _VIDEO_LOAD_CLASSES)
        modes = _detect_modes(mtype, n_images, n_videos)
        spec = {
            "_id": mid,
            "type": mtype, "modes": modes, "builder": "workflow",
            "display_name": mid,
        }
        if n_images:
            spec["max_ref_images"] = n_images

        meta_path = path.with_name(path.stem + ".meta.json")
        if meta_path.is_file():
            try:
                spec.update(json.loads(meta_path.read_text(encoding="utf-8")))
            except Exception as e:  # noqa: BLE001
                print(f"[comfyui] bad meta {meta_path.name}: {e}")

        MODELS[mid] = spec
        _WORKFLOWS[mid] = graph
        print(f"[comfyui] workflow model '{mid}' ({spec['type']}, {spec['modes']})")


def _detect_output(graph: dict):
    """Return (media_type, history_key) from the graph's output node."""
    for node in graph.values():
        cls = node.get("class_type") if isinstance(node, dict) else None
        if cls in _OUTPUT_NODES:
            return _OUTPUT_NODES[cls]
    return "image", "images"   # safe default


def _count_class(graph: dict, cls: str) -> int:
    return sum(1 for n in graph.values()
               if isinstance(n, dict) and n.get("class_type") == cls)


def _detect_modes(mtype: str, n_images: int, n_videos: int = 0) -> list:
    if mtype == "video":
        modes = ["t2v"]
        if n_images:
            modes.append("i2v")          # first-frame / reference image
        if n_videos:
            modes.append("control")      # source video (edit / character replace)
        return modes
    if mtype == "audio":
        return ["tts"]
    return ["t2i", "i2i"] if n_images else ["t2i"]


# --------------------------------------------------------------------------
# Contract endpoints
# --------------------------------------------------------------------------
@app.get("/v1/health")
def health():
    return {"status": "ok", "contract_versions": CONTRACT_VERSIONS}


@app.get("/v1/models")
def models():
    data = []
    for mid, spec in MODELS.items():
        entry = {"id": mid, "type": spec["type"], "modes": spec["modes"]}
        for k in ("display_name", "description", "default_steps", "default_guidance",
                  "max_ref_images", "max_width", "max_height",
                  "needs_speaker_ref", "needs_ref_text", "control_types"):
            if k in spec:
                entry[k] = spec[k]
        data.append(entry)
    return {"data": data}


@app.post("/v1/files")
async def upload(file: UploadFile, purpose: str = Form("reference")):
    data = await file.read()
    fid = "file-" + uuid.uuid4().hex
    _FILES[fid] = (data, file.content_type or "application/octet-stream")
    return {"file_id": fid}


@app.get("/v1/files/{file_id}")
def get_file(file_id: str):
    item = _FILES.get(file_id)
    if not item:
        return JSONResponse({"error": "no such file"}, status_code=404)
    data, ctype = item
    return Response(content=data, media_type=ctype)


@app.post("/v1/images/generations")
async def images(payload: dict):
    return _submit(payload, kind="image")


@app.post("/v1/videos")
async def videos(payload: dict):
    return _submit(payload, kind="video")


@app.post("/v1/audio/speech")
async def audio(payload: dict):
    return _submit(payload, kind="audio")


@app.get("/v1/jobs/{job_id}")
def job_status(job_id: str):
    job = _JOBS.get(job_id)
    if not job:
        return JSONResponse({"error": "no such job"}, status_code=404)

    pid = job["prompt_id"]
    try:
        with httpx.Client(timeout=30) as c:
            hist = c.get(f"{COMFYUI_URL}/history/{pid}").json()
    except Exception as e:  # noqa: BLE001
        return {"id": job_id, "status": "failed", "error": f"comfyui unreachable: {e}"}

    entry = hist.get(pid)
    # A failed run may have no outputs at all, so check the error state first.
    if entry:
        status = entry.get("status", {})
        if status.get("status_str") == "error":
            return {"id": job_id, "status": "failed", "phase": "error",
                    "progress": 0.0, "file_id": None,
                    "error": _history_error(status)}
    if entry and entry.get("outputs"):
        view = _first_media_ref(entry["outputs"])
        if not view:
            return {"id": job_id, "status": "failed", "error": "no media in outputs"}
        try:
            with httpx.Client(timeout=300) as c:
                r = c.get(f"{COMFYUI_URL}/view", params=view)
                r.raise_for_status()
                fid = "out-" + job_id
                _FILES[fid] = (r.content, r.headers.get("content-type", "image/png"))
        except Exception as e:  # noqa: BLE001
            return {"id": job_id, "status": "failed", "error": f"fetch failed: {e}"}
        return {"id": job_id, "status": "succeeded", "phase": "done",
                "progress": 1.0, "file_id": fid, "error": None}

    # Not finished yet: distinguish running vs queued via /queue.
    phase, progress, running = "queued", 0.0, False
    try:
        with httpx.Client(timeout=10) as c:
            q = c.get(f"{COMFYUI_URL}/queue").json()
        running = any(_pid_in(item, pid) for item in q.get("queue_running", []))
    except Exception:  # noqa: BLE001
        pass
    if running:
        phase, progress = "generating", 0.5
    return {"id": job_id, "status": "running" if running else "queued",
            "phase": phase, "progress": progress, "file_id": None, "error": None}


# --------------------------------------------------------------------------
# ComfyUI mapping
# --------------------------------------------------------------------------
def _submit(payload: dict, kind: str):
    mid = payload.get("model")
    spec = MODELS.get(mid)
    if not spec:
        return JSONResponse({"error": f"unknown model {mid!r}"}, status_code=400)

    try:
        graph = _build_graph(payload, spec)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"workflow build failed: {e}"}, status_code=400)

    _log_patch(mid, spec, payload, graph)

    try:
        with httpx.Client(timeout=60) as c:
            r = c.post(f"{COMFYUI_URL}/prompt",
                       json={"prompt": graph, "client_id": CLIENT_ID})
            if r.status_code >= 400:
                # ComfyUI validates the graph and returns a JSON body describing
                # exactly what's wrong (missing node type, bad input, absent
                # checkpoint, ...). Surface that instead of the bare status line.
                detail = _comfy_error_detail(r)
                return JSONResponse(
                    {"error": f"comfyui /prompt rejected the graph "
                              f"(HTTP {r.status_code}): {detail}"},
                    status_code=502,
                )
            pid = r.json()["prompt_id"]
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"comfyui /prompt failed: {e}"}, status_code=502)

    job_id = f"{kind}-{uuid.uuid4().hex}"
    _JOBS[job_id] = {"prompt_id": pid, "kind": kind, "model_id": mid}
    return {"id": job_id, "status": "queued"}


def _comfy_error_detail(resp) -> str:
    """Turn ComfyUI's /prompt error response into a compact, readable string.

    ComfyUI returns JSON like::

        {"error": {"type": "...", "message": "...", "details": "..."},
         "node_errors": {"<node_id>": {"class_type": "KSampler",
                          "errors": [{"message": "...", "details": "..."}]}}}

    We flatten the top-level message plus every node error (node id + class +
    each message/details) so the cause is visible in Blender's queue error.
    """
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001 — not JSON; fall back to raw text
        text = (resp.text or "").strip()
        return text[:500] or "(empty response body)"

    parts: list = []
    err = body.get("error")
    if isinstance(err, dict):
        msg = err.get("message") or err.get("type") or ""
        det = err.get("details") or ""
        parts.append(": ".join(p for p in (msg, det) if p) or str(err))
    elif err:
        parts.append(str(err))

    for nid, ne in (body.get("node_errors") or {}).items():
        cls = ne.get("class_type", "?") if isinstance(ne, dict) else "?"
        msgs = []
        for e in (ne.get("errors") or []) if isinstance(ne, dict) else []:
            m = e.get("message", "") if isinstance(e, dict) else str(e)
            d = e.get("details", "") if isinstance(e, dict) else ""
            msgs.append(" ".join(p for p in (m, d) if p).strip())
        joined = "; ".join(m for m in msgs if m) or "invalid"
        parts.append(f"node {nid} ({cls}): {joined}")

    return " | ".join(p for p in parts if p) or json.dumps(body)[:500]


def _norm_seed(seed) -> int:
    """ComfyUI seeds must be 0 <= seed <= 2**64-1; wrap negatives into range.

    Pallaidium may send a signed/negative seed (e.g. its random sentinel); a raw
    negative value makes KSampler reject the graph ("smaller than min of 0").
    """
    try:
        s = int(seed)
    except (TypeError, ValueError):
        return 0
    return s & 0xFFFFFFFFFFFFFFFF if s < 0 else s


def _available_checkpoints() -> list:
    """List checkpoint filenames ComfyUI currently offers (empty if unreachable)."""
    try:
        with httpx.Client(timeout=10) as c:
            info = c.get(f"{COMFYUI_URL}/object_info/CheckpointLoaderSimple").json()
        return list(info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0])
    except Exception:  # noqa: BLE001 — fall back to the configured name
        return []


def _resolve_ckpt(spec: dict) -> str:
    """Resolve the checkpoint to actually load against what ComfyUI has on disk.

    Order: env override (``COMFYUI_CKPT_<ID>`` then ``COMFYUI_CKPT``) -> the
    spec's exact ``ckpt`` if present -> first available whose name contains the
    spec's keyword hint (``ckpt_match`` or the leading token of the configured
    name, e.g. ``ltx``) -> first available. If ComfyUI can't be queried we keep
    the configured name and let ComfyUI validate it.
    """
    mid = (spec.get("_id") or _model_id_for(spec) or "").upper()
    env = os.environ.get(f"COMFYUI_CKPT_{mid}") or os.environ.get("COMFYUI_CKPT")
    if env:
        return env

    desired = spec.get("ckpt", "")
    avail = _available_checkpoints()
    if not avail or desired in avail:
        return desired

    hint = (spec.get("ckpt_match") or (desired.split("-")[0] if desired else "")).lower()
    if hint:
        for name in avail:
            if hint in name.lower():
                return name
    return avail[0]


def _build_graph(payload: dict, spec: dict) -> dict:
    """Dispatch to the workflow template named by spec['builder']."""
    builder = spec.get("builder", "checkpoint")
    if builder == "workflow":
        return _build_workflow_graph(payload, spec)
    if builder == "checkpoint":
        return _build_checkpoint_graph(payload, spec)
    if builder == "ltxv":
        return _build_ltxv_graph(payload, spec)
    raise ValueError(f"no builder {builder!r}")


# Node titles (case-insensitive) the patcher recognises in a saved workflow.
_PROMPT_TITLES = {"prompt", "positive", "positive prompt"}
_NEGATIVE_TITLES = {"negative", "negative prompt"}
_WIDTH_TITLES = {"width"}
_HEIGHT_TITLES = {"height"}
_STEPS_TITLES = {"steps"}
_CFG_TITLES = {"cfg", "guidance"}


def _build_workflow_graph(payload: dict, spec: dict) -> dict:
    """Patch a user-saved API-format workflow with the request's parameters.

    Replacement rules (a node's title is set in ComfyUI via right-click ->
    Title):
      * node titled prompt/positive  -> text = prompt   (audio uses `input`)
      * node titled negative         -> text = negative_prompt
      * node titled width/height/steps/cfg -> that single input value
      * any node with a `seed`/`noise_seed` input -> seed (safe to set globally)
      * every LoadImage node          -> the supplied reference image(s), in order
      * the first LoadVideo node       -> the supplied source video (video edit /
                                          character replacement)
    Unmatched nodes are left exactly as saved, so the workflow runs as designed.
    """
    mid = spec.get("_id") or _model_id_for(spec)
    template = _WORKFLOWS.get(mid)
    if template is None:
        raise ValueError(f"no workflow loaded for {mid!r}")
    g = copy.deepcopy(template)

    text = payload.get("prompt") or payload.get("input") or ""
    negative = payload.get("negative_prompt", "")
    seed = payload.get("seed")

    # Upload reference image(s) and bind them to LoadImage nodes in order.
    ref_names = _upload_refs(payload)
    load_nodes = [nid for nid, n in g.items()
                  if isinstance(n, dict) and n.get("class_type") == "LoadImage"]
    load_nodes.sort(key=lambda nid: _title(g[nid]) or nid)

    for nid, node in g.items():
        if not isinstance(node, dict):
            continue
        inputs = node.setdefault("inputs", {})
        title = _title(node)
        if title in _PROMPT_TITLES and "text" in inputs:
            inputs["text"] = text
        elif title in _NEGATIVE_TITLES and "text" in inputs:
            inputs["text"] = negative
        if title in _WIDTH_TITLES and payload.get("width") and "width" in inputs:
            inputs["width"] = int(payload["width"])
        if title in _HEIGHT_TITLES and payload.get("height") and "height" in inputs:
            inputs["height"] = int(payload["height"])
        if title in _STEPS_TITLES and payload.get("num_inference_steps") and "steps" in inputs:
            inputs["steps"] = int(payload["num_inference_steps"])
        if title in _CFG_TITLES and payload.get("guidance_scale") and "cfg" in inputs:
            inputs["cfg"] = float(payload["guidance_scale"])
        if seed is not None:
            if "seed" in inputs:
                inputs["seed"] = _norm_seed(seed)
            if "noise_seed" in inputs:
                inputs["noise_seed"] = _norm_seed(seed)

    for name, nid in zip(ref_names, load_nodes):
        g[nid].setdefault("inputs", {})["image"] = name

    # Explicit node bindings from <id>.meta.json (unambiguous; win over titles).
    # For graphs where parameters live on non-obvious nodes (primitives, an LLM
    # prompt-enhance chain, identically-titled encoders), the sidecar names the
    # exact {node, input} each request field maps to. See comfyui_workflows/README.
    _apply_bindings(g, spec, payload, text, negative, seed)

    # Bind a source video (control_file_id / video_file_id) to the first loader.
    vid_name = _upload_source_video(payload)
    if vid_name:
        for nid, node in g.items():
            if isinstance(node, dict) and node.get("class_type") in _VIDEO_LOAD_CLASSES:
                vinputs = node.setdefault("inputs", {})
                for key in ("video", "file", "video_path"):
                    if key in vinputs:
                        vinputs[key] = vid_name
                        break
                else:
                    vinputs["video"] = vid_name
                break

    return g


def _as_int(v):
    try:
        return int(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _as_float(v):
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _apply_bindings(g: dict, spec: dict, payload: dict,
                    text: str, negative: str, seed) -> None:
    """Set request values onto explicit nodes named by ``spec['bindings']``.

    ``bindings`` maps a logical field -> a target ``{"node": id, "input": key}``
    (or a list of targets). Supported fields: prompt, negative, width, height,
    steps, cfg, fps, num_frames, seed, strength. A field is skipped when the
    request didn't supply it, so the workflow keeps its saved default. The target
    node's input key can be anything (``value`` for primitives, ``text`` for a
    CLIPTextEncode, ``noise_seed`` for RandomNoise, ...).
    """
    bindings = spec.get("bindings") or {}
    if not bindings:
        return

    values = {
        "prompt": text or None,
        "negative": negative or None,
        "width": _as_int(payload.get("width")),
        "height": _as_int(payload.get("height")),
        "steps": _as_int(payload.get("num_inference_steps")),
        "cfg": _as_float(payload.get("guidance_scale")),
        "fps": _as_int(payload.get("fps")),
        "num_frames": _as_int(payload.get("num_frames")),
        "strength": _as_float(payload.get("strength")),
        "seed": _norm_seed(seed) if seed is not None else None,
    }

    for field, target in bindings.items():
        val = values.get(field)
        if val is None:
            continue
        for t in (target if isinstance(target, list) else [target]):
            if not isinstance(t, dict):
                continue
            node = g.get(str(t.get("node")))
            key = t.get("input")
            if isinstance(node, dict) and key:
                node.setdefault("inputs", {})[key] = val


def _log_patch(mid, spec: dict, payload: dict, graph: dict) -> None:
    """Print what arrived and what the patch actually wrote, so a 'it ran my
    saved workflow, not Pallaidium's request' problem is visible at a glance."""
    prompt = payload.get("prompt") or payload.get("input") or ""
    print(f"[comfyui] >>> job for model '{mid}' (builder={spec.get('builder')})")
    print(f"[comfyui]     payload prompt={prompt[:80]!r} "
          f"neg={str(payload.get('negative_prompt',''))[:40]!r} "
          f"size={payload.get('width')}x{payload.get('height')} "
          f"seed={payload.get('seed')} fps={payload.get('fps')} "
          f"frames={payload.get('num_frames')}")
    bindings = spec.get("bindings") or {}
    if bindings:
        for field, target in bindings.items():
            for t in (target if isinstance(target, list) else [target]):
                if not isinstance(t, dict):
                    continue
                node = graph.get(str(t.get("node")))
                key = t.get("input")
                cur = (node or {}).get("inputs", {}).get(key) if node else None
                here = "OK" if node else "NODE-MISSING"
                print(f"[comfyui]     bind {field:<9} -> {t.get('node')}.{key} "
                      f"= {str(cur)[:60]!r}  [{here}]")
    else:
        print("[comfyui]     (no bindings in meta; patching by node title only)")


def _title(node: dict) -> str:
    return ((node.get("_meta") or {}).get("title", "") or "").strip().lower()


def _model_id_for(spec: dict) -> str:
    for mid, s in MODELS.items():
        if s is spec:
            return mid
    return ""


def _build_checkpoint_graph(payload: dict, spec: dict) -> dict:
    """Standard checkpoint txt2img / img2img graph in ComfyUI API format.

    If an init image is supplied (img2img) it is uploaded to ComfyUI and routed
    through LoadImage -> VAEEncode with denoise from `strength`; otherwise a
    txt2img EmptyLatentImage is used.
    """
    prompt = payload.get("prompt", "")
    negative = payload.get("negative_prompt", "")
    width = int(payload.get("width") or spec.get("max_width", 1024))
    height = int(payload.get("height") or spec.get("max_height", 1024))
    steps = int(payload.get("num_inference_steps") or spec.get("default_steps", 20))
    cfg = float(payload.get("guidance_scale") or spec.get("default_guidance", 7.0))
    seed = _norm_seed(payload.get("seed") or 0)

    init_name = _maybe_upload_init(payload)

    g = {
        "4": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": _resolve_ckpt(spec)}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode",
              "inputs": {"text": negative, "clip": ["4", 1]}},
        "8": {"class_type": "VAEDecode",
              "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": "pallaidium", "images": ["8", 0]}},
    }

    if init_name:
        # img2img: encode the uploaded image into latent space.
        denoise = float(payload.get("strength") or 0.75)
        g["10"] = {"class_type": "LoadImage", "inputs": {"image": init_name}}
        g["11"] = {"class_type": "VAEEncode",
                   "inputs": {"pixels": ["10", 0], "vae": ["4", 2]}}
        latent = ["11", 0]
    else:
        denoise = 1.0
        g["5"] = {"class_type": "EmptyLatentImage",
                  "inputs": {"width": width, "height": height, "batch_size": 1}}
        latent = ["5", 0]

    g["3"] = {"class_type": "KSampler",
              "inputs": {"seed": seed, "steps": steps, "cfg": cfg,
                         "sampler_name": "euler", "scheduler": "normal",
                         "denoise": denoise, "model": ["4", 0],
                         "positive": ["6", 0], "negative": ["7", 0],
                         "latent_image": latent}}
    return g


def _build_ltxv_graph(payload: dict, spec: dict) -> dict:
    """LTX-Video txt2video / image2video graph (ComfyUI API format).

    Requires LTX-Video model files in ComfyUI and the **VideoHelperSuite**
    custom node (`VHS_VideoCombine`) for mp4 output. If an init image is present
    it is uploaded and routed through `LTXVImgToVideo` (first-frame i2v);
    otherwise `EmptyLTXVLatentVideo` is used for pure txt2video.

    Note: this uses a plain KSampler for portability. For best quality, swap in
    the LTXVScheduler + SamplerCustom chain from the official LTXV workflow.
    """
    prompt = payload.get("prompt", "")
    negative = payload.get("negative_prompt", "") or "low quality, worst quality, blurry"
    # LTXV wants width/height multiples of 32 and length = 8*k + 1.
    width = _round_to(int(payload.get("width") or spec.get("max_width", 768)), 32)
    height = _round_to(int(payload.get("height") or spec.get("max_height", 512)), 32)
    n_frames = int(payload.get("num_frames") or 97)
    length = max(9, ((n_frames - 1) // 8) * 8 + 1)
    fps = int(payload.get("fps") or 25)
    steps = int(payload.get("num_inference_steps") or spec.get("default_steps", 30))
    cfg = float(payload.get("guidance_scale") or spec.get("default_guidance", 3.0))
    seed = _norm_seed(payload.get("seed") or 0)

    init_name = _maybe_upload_init(payload)

    g = {
        "4": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": _resolve_ckpt(spec)}},
        "6": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode",
              "inputs": {"text": negative, "clip": ["4", 1]}},
    }

    if init_name:
        # image-to-video: the node conditions and builds the latent together.
        g["10"] = {"class_type": "LoadImage", "inputs": {"image": init_name}}
        g["20"] = {"class_type": "LTXVImgToVideo",
                   "inputs": {"positive": ["6", 0], "negative": ["7", 0],
                              "vae": ["4", 2], "image": ["10", 0],
                              "width": width, "height": height,
                              "length": length, "batch_size": 1}}
        pos, neg, latent = ["20", 0], ["20", 1], ["20", 2]
    else:
        g["21"] = {"class_type": "LTXVConditioning",
                   "inputs": {"positive": ["6", 0], "negative": ["7", 0],
                              "frame_rate": fps}}
        g["5"] = {"class_type": "EmptyLTXVLatentVideo",
                  "inputs": {"width": width, "height": height,
                             "length": length, "batch_size": 1}}
        pos, neg, latent = ["21", 0], ["21", 1], ["5", 0]

    g["3"] = {"class_type": "KSampler",
              "inputs": {"seed": seed, "steps": steps, "cfg": cfg,
                         "sampler_name": "euler", "scheduler": "normal",
                         "denoise": 1.0, "model": ["4", 0],
                         "positive": pos, "negative": neg,
                         "latent_image": latent}}
    g["8"] = {"class_type": "VAEDecode",
              "inputs": {"samples": ["3", 0], "vae": ["4", 2]}}
    # VideoHelperSuite combines the decoded frames into an mp4.
    g["9"] = {"class_type": "VHS_VideoCombine",
              "inputs": {"images": ["8", 0], "frame_rate": fps, "loop_count": 0,
                         "filename_prefix": "pallaidium", "format": "video/h264-mp4",
                         "pingpong": False, "save_output": True}}
    return g


def _round_to(value: int, multiple: int) -> int:
    return max(multiple, round(value / multiple) * multiple)


def _maybe_upload_init(payload: dict) -> str:
    """Upload a single init image (file id or inline b64); return its name."""
    refs = _upload_refs(payload)
    return refs[0] if refs else ""


def _upload_refs(payload: dict) -> list:
    """Upload every reference image in the payload to ComfyUI, in order.

    Order: explicit `reference_file_ids` (multi-ref), else the single
    `image_file_id` / inline `image_b64`. Returns the stored ComfyUI names.
    """
    blobs: list = []
    for fid in payload.get("reference_file_ids") or []:
        if fid in _FILES:
            blobs.append(_FILES[fid][0])
    if not blobs:
        if payload.get("image_file_id") and payload["image_file_id"] in _FILES:
            blobs.append(_FILES[payload["image_file_id"]][0])
        elif payload.get("image_b64"):
            blobs.append(base64.b64decode(payload["image_b64"]))
    return [_upload_image_bytes(b) for b in blobs]


def _upload_image_bytes(data: bytes) -> str:
    return _upload_to_comfy(data, f"pallaidium-{uuid.uuid4().hex}.png", "image/png")


def _upload_source_video(payload: dict) -> str:
    """Upload the source video (control_file_id / video_file_id) to ComfyUI.

    The contract uploads an edited/source video via POST /v1/files; Pallaidium
    sends it as `control_file_id` (control mode) — `video_file_id` is also
    accepted. Returns the stored ComfyUI name for a LoadVideo node.
    """
    for key in ("control_file_id", "video_file_id"):
        fid = payload.get(key)
        if fid and fid in _FILES:
            data, ctype = _FILES[fid]
            return _upload_to_comfy(data, f"pallaidium-{uuid.uuid4().hex}.mp4",
                                    ctype or "video/mp4")
    return ""


def _upload_to_comfy(data: bytes, fname: str, content_type: str) -> str:
    """POST bytes to ComfyUI /upload/image (used for images and videos);
    return the stored name (with subfolder)."""
    with httpx.Client(timeout=120) as c:
        r = c.post(f"{COMFYUI_URL}/upload/image",
                   files={"image": (fname, data, content_type)},
                   data={"overwrite": "true"})
        r.raise_for_status()
        info = r.json()
    name = info.get("name", fname)
    sub = info.get("subfolder")
    return f"{sub}/{name}" if sub else name


def _first_media_ref(outputs: dict) -> dict | None:
    """Find the first media output and return /view query params for it.

    Checks known keys first — SaveImage (`images`), VideoHelperSuite/animated
    (`gifs`), audio nodes (`audio`), and the native SaveVideo (`video`) — then
    falls back to scanning every output list for the first file-like entry, so
    new/unknown save nodes still work as long as they emit `{filename, ...}`.
    """
    def as_ref(items):
        if (isinstance(items, list) and items and isinstance(items[0], dict)
                and "filename" in items[0]):
            m = items[0]
            return {"filename": m["filename"],
                    "subfolder": m.get("subfolder", ""),
                    "type": m.get("type", "output")}
        return None

    for key in ("images", "gifs", "audio", "video", "videos"):
        for node in outputs.values():
            if isinstance(node, dict):
                ref = as_ref(node.get(key))
                if ref:
                    return ref
    # Fallback: any list of file-like dicts under any key.
    for node in outputs.values():
        if isinstance(node, dict):
            for items in node.values():
                ref = as_ref(items)
                if ref:
                    return ref
    return None


def _history_error(status: dict) -> str:
    """Extract the real failure from a ComfyUI history `status` block.

    ComfyUI records the failure in ``status.messages`` as an
    ``("execution_error", {node_id, node_type, exception_type,
    exception_message, ...})`` tuple. Flatten the first such entry so the queue
    error names the failing node and Python exception instead of a generic
    "execution error".
    """
    for item in status.get("messages") or []:
        try:
            kind, data = item[0], item[1]
        except (IndexError, TypeError):
            continue
        if kind == "execution_error" and isinstance(data, dict):
            node = data.get("node_type") or data.get("node_id") or "?"
            etype = data.get("exception_type", "")
            emsg = (data.get("exception_message") or "").strip()
            return f"node {node} failed: {etype}: {emsg}".strip(": ")
    return "comfyui execution error"


def _pid_in(queue_item, pid: str) -> bool:
    """A /queue entry is [number, prompt_id, graph, ...]; match the prompt id."""
    try:
        return queue_item[1] == pid
    except (IndexError, TypeError):
        return False


def _check_comfyui() -> None:
    """Ping COMFYUI_URL once at startup and print a clear status line."""
    try:
        with httpx.Client(timeout=5) as c:
            r = c.get(f"{COMFYUI_URL}/system_stats")
            r.raise_for_status()
        print(f"[comfyui] connected to ComfyUI at {COMFYUI_URL}")
    except Exception as e:  # noqa: BLE001
        print("=" * 70)
        print(f"[comfyui] WARNING: could not reach ComfyUI at {COMFYUI_URL}")
        print(f"          ({type(e).__name__}: {e})")
        print("          Start ComfyUI first (its console prints the address,")
        print("          e.g. http://127.0.0.1:8188), then set COMFYUI_URL to it:")
        print('            PowerShell:  $env:COMFYUI_URL="http://127.0.0.1:8188"')
        print("          The adapter will keep running and retry on each request.")
        print("=" * 70)


# Discover workflow-file models at import time so /v1/models lists them.
_discover_workflows()
_check_comfyui()
