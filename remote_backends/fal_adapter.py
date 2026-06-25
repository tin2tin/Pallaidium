"""Reference adapter: Pallaidium Backend Contract v0.1  ->  fal.ai.

A small local service that implements the contract and forwards generation to
fal.ai (so Pallaidium can drive real cloud models such as ByteDance **Seedance**
video, FLUX image, and TTS/Whisper). Pallaidium itself stays provider-agnostic;
all fal-specific code and the API key live here.

**Stdlib only** — no fastapi / uvicorn / httpx / fal-client. It talks to fal's
documented **queue REST API** (``https://queue.fal.run``) directly over
``urllib``, so the Pallaidium add-on can launch it with Blender's own Python.

This is a *reference* implementation — adjust the MODELS table and the per-model
argument mapping to the exact fal endpoints you want. It demonstrates the 1:1
shape match between fal's queue API and the contract's job model.

Run (or let Pallaidium's "Start Backend" button do it):
    python fal_adapter.py --port 8000 --fal-key YOUR_FAL_KEY
    # or set FAL_KEY in the environment

Then in Pallaidium the backend URL is filled in automatically; if you run it by
hand, set Remote Backend URL = http://localhost:8000, switch Model Source to
Remote, and click "Refresh Remote Models".
"""

import os
import uuid
import base64
import argparse

from _adapter_http import (
    BaseAdapterHandler, serve,
    get_json, get_bytes, post_json,
)

CONTRACT_VERSIONS = ["v0.1"]
FAL_QUEUE = "https://queue.fal.run"
FAL_KEY = os.environ.get("FAL_KEY", "")

# Map contract model ids -> fal endpoint + media type + capability hints.
# Endpoint ids are examples; confirm the current ones on fal.ai.
MODELS = {
    "seedance-2.0": {
        "type": "video",
        "modes": ["t2v", "i2v"],
        "fal": "fal-ai/bytedance/seedance/v1/pro/text-to-video",
        "fal_i2v": "fal-ai/bytedance/seedance/v1/pro/image-to-video",
        "display_name": "Seedance 2.0 (fal)",
    },
    "flux-dev": {
        "type": "image",
        "modes": ["t2i"],
        "fal": "fal-ai/flux/dev",
        "display_name": "FLUX.1 [dev] (fal)",
        "default_steps": 28,
    },
}

# In-memory stores (use a real store/CDN in production).
_FILES: dict[str, tuple[bytes, str]] = {}   # file_id -> (bytes, content_type)
_JOBS: dict[str, dict] = {}                 # job_id  -> {status_url, response_url, ...}


def _auth() -> dict:
    return {"Authorization": f"Key {FAL_KEY}"} if FAL_KEY else {}


# --------------------------------------------------------------------------
# Contract route handlers — each returns (status_code, json_obj)
# --------------------------------------------------------------------------
def route_health():
    return 200, {"status": "ok", "contract_versions": CONTRACT_VERSIONS}


def route_models():
    data = []
    for mid, spec in MODELS.items():
        entry = {"id": mid, "type": spec["type"], "modes": spec["modes"]}
        for k in ("display_name", "default_steps", "max_ref_images",
                  "needs_speaker_ref", "needs_ref_text", "control_types"):
            if k in spec:
                entry[k] = spec[k]
        data.append(entry)
    return 200, {"data": data}


def route_upload(files: dict):
    if not files:
        return 400, {"error": "no file part in multipart body"}
    _name, ctype, data = next(iter(files.values()))
    fid = "file-" + uuid.uuid4().hex
    _FILES[fid] = (data, ctype or "application/octet-stream")
    return 200, {"file_id": fid}


def route_job_status(job_id: str):
    job = _JOBS.get(job_id)
    if not job:
        return 404, {"error": "no such job"}
    try:
        status = get_json(job["status_url"], headers=_auth(), timeout=30)
    except Exception as e:  # noqa: BLE001
        return 200, {"id": job_id, "status": "failed", "error": str(e)}

    state = (status.get("status") or "").upper()
    if state == "COMPLETED":
        try:
            result = get_json(job["response_url"], headers=_auth(), timeout=60)
        except Exception as e:  # noqa: BLE001
            return 200, {"id": job_id, "status": "failed",
                         "error": f"result fetch failed: {e}"}
        url = _first_media_url(result)
        if not url:
            return 200, {"id": job_id, "status": "failed", "error": "no media in result"}
        try:
            data, ctype = get_bytes(url, timeout=300)
        except Exception as e:  # noqa: BLE001
            return 200, {"id": job_id, "status": "failed", "error": f"download failed: {e}"}
        fid = "out-" + job_id
        _FILES[fid] = (data, ctype or "application/octet-stream")
        return 200, {"id": job_id, "status": "succeeded", "phase": "done",
                     "progress": 1.0, "file_id": fid, "error": None}
    if state == "IN_PROGRESS":
        return 200, {"id": job_id, "status": "running", "phase": "generating",
                     "progress": 0.5, "file_id": None, "error": None}
    # IN_QUEUE (or anything else not terminal) -> still queued.
    return 200, {"id": job_id, "status": "queued", "phase": "queued",
                 "progress": 0.0, "file_id": None, "error": None}


# --------------------------------------------------------------------------
# fal mapping
# --------------------------------------------------------------------------
def _submit(payload: dict, kind: str):
    if not FAL_KEY:
        return 400, {"error": "FAL_KEY is not set (pass --fal-key or set the "
                              "Remote Backend Key in Pallaidium)"}
    mid = payload.get("model")
    spec = MODELS.get(mid)
    if not spec:
        return 400, {"error": f"unknown model {mid!r}"}

    args = _to_fal_args(payload, spec, kind)
    fal_app = spec["fal"]
    # image-to-video uses a different fal endpoint when an init image is present.
    if kind == "video" and (payload.get("image_file_id") or payload.get("image_b64")) \
            and spec.get("fal_i2v"):
        fal_app = spec["fal_i2v"]
        args["image_url"] = _data_url(payload)

    try:
        code, body = post_json(f"{FAL_QUEUE}/{fal_app}", args,
                               headers=_auth(), timeout=60)
    except Exception as e:  # noqa: BLE001
        return 502, {"error": f"fal submit failed: {e}"}
    if code >= 400 or not isinstance(body, dict):
        return 502, {"error": f"fal submit rejected (HTTP {code}): {str(body)[:300]}"}

    request_id = body.get("request_id")
    # fal returns status_url / response_url directly; fall back to building them.
    status_url = body.get("status_url") or f"{FAL_QUEUE}/{fal_app}/requests/{request_id}/status"
    response_url = body.get("response_url") or f"{FAL_QUEUE}/{fal_app}/requests/{request_id}"
    if not request_id:
        return 502, {"error": f"fal submit returned no request_id: {str(body)[:300]}"}

    job_id = f"{kind}-{uuid.uuid4().hex}"
    _JOBS[job_id] = {"status_url": status_url, "response_url": response_url,
                     "fal_app": fal_app, "kind": kind, "model_id": mid}
    return 200, {"id": job_id, "status": "queued"}


def _to_fal_args(payload: dict, spec: dict, kind: str) -> dict:
    """Translate contract fields -> fal arguments (per-model tweaks as needed)."""
    args = {"prompt": payload.get("prompt", "")}
    if payload.get("negative_prompt"):
        args["negative_prompt"] = payload["negative_prompt"]
    if payload.get("seed"):
        args["seed"] = payload["seed"]
    if kind == "image":
        if payload.get("num_inference_steps"):
            args["num_inference_steps"] = payload["num_inference_steps"]
        if payload.get("guidance_scale"):
            args["guidance_scale"] = payload["guidance_scale"]
        if payload.get("width") and payload.get("height"):
            args["image_size"] = {"width": payload["width"], "height": payload["height"]}
    if kind == "video":
        # Seedance uses resolution + duration; map from contract.
        if payload.get("num_frames") and payload.get("fps"):
            args["duration"] = max(1, round(payload["num_frames"] / payload["fps"]))
        if payload.get("height"):
            args["resolution"] = "720p" if payload["height"] >= 720 else "480p"
    return args


def _data_url(payload: dict) -> str:
    """Return a data: URL for the init image (uploaded file or inline b64)."""
    if payload.get("image_file_id") and payload["image_file_id"] in _FILES:
        data, ctype = _FILES[payload["image_file_id"]]
        return f"data:{ctype};base64," + base64.b64encode(data).decode()
    if payload.get("image_b64"):
        return "data:image/png;base64," + payload["image_b64"]
    return ""


def _first_media_url(result: dict) -> str:
    """Find a media URL in a fal result (shapes vary by model)."""
    if not isinstance(result, dict):
        return ""
    for key in ("video", "image", "audio"):
        v = result.get(key)
        if isinstance(v, dict) and v.get("url"):
            return v["url"]
    for key in ("images", "videos"):
        v = result.get(key)
        if isinstance(v, list) and v and isinstance(v[0], dict) and v[0].get("url"):
            return v[0]["url"]
    if result.get("url"):
        return result["url"]
    return ""


# --------------------------------------------------------------------------
# HTTP handler
# --------------------------------------------------------------------------
class Handler(BaseAdapterHandler):
    log_tag = "fal"

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/v1/health":
            return self.send_json(*_swap(route_health()))
        if path == "/v1/models":
            return self.send_json(*_swap(route_models()))
        if path.startswith("/v1/jobs/"):
            return self.send_json(*_swap(route_job_status(path.rsplit("/", 1)[1])))
        if path.startswith("/v1/files/"):
            fid = path.rsplit("/", 1)[1]
            item = _FILES.get(fid)
            if not item:
                return self.send_json({"error": "no such file"}, 404)
            data, ctype = item
            return self.send_bytes(data, ctype)
        return self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/v1/files":
            _fields, files = self.read_multipart()
            return self.send_json(*_swap(route_upload(files)))
        if path in ("/v1/videos", "/v1/images/generations"):
            payload = self.read_json()
            kind = "video" if path == "/v1/videos" else "image"
            return self.send_json(*_swap(_submit(payload, kind)))
        return self.send_json({"error": "not found"}, 404)


def _swap(result):
    """Route fns return (code, obj); send_json takes (obj, code)."""
    code, obj = result
    return obj, code


def main():
    global FAL_KEY
    ap = argparse.ArgumentParser(description="Pallaidium <-> fal.ai adapter")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--fal-key", default=FAL_KEY, help="fal.ai API key (or set FAL_KEY)")
    args = ap.parse_args()
    FAL_KEY = args.fal_key or ""

    banner = "[fal] FAL_KEY set" if FAL_KEY else \
        "[fal] WARNING: no FAL_KEY — set it in Pallaidium's Remote Backend Key field"
    serve(Handler, host=args.host, port=args.port, banner=banner)


if __name__ == "__main__":
    main()
