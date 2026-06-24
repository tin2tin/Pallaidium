"""Reference adapter: Pallaidium Backend Contract v0.1  ->  fal.ai.

A small local service that implements the contract and forwards generation to
fal.ai (so Pallaidium can drive real cloud models such as ByteDance **Seedance**
video, FLUX image, and TTS/Whisper). Pallaidium itself stays provider-agnostic;
all fal-specific code and the API key live here.

This is a *reference* implementation — adjust the MODELS table and the per-model
argument mapping to the exact fal endpoints you want. It demonstrates the 1:1
shape match between fal's queue API and the contract's job model.

Run:
    pip install -r requirements.txt
    set FAL_KEY=...            # your fal.ai key (PowerShell: $env:FAL_KEY="...")
    uvicorn fal_adapter:app --port 8000

Then in Pallaidium preferences set Remote Backend URL = http://localhost:8000,
switch Model Source to Remote, and click "Refresh Remote Models".
"""

import os
import uuid
import httpx
import fal_client
from fastapi import FastAPI, UploadFile, Form, Response
from fastapi.responses import JSONResponse

app = FastAPI(title="Pallaidium ↔ fal.ai adapter")

CONTRACT_VERSIONS = ["v0.1"]

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
_JOBS: dict[str, dict] = {}                 # job_id  -> {handle, model_id, kind}


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
        for k in ("display_name", "default_steps", "max_ref_images",
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


@app.post("/v1/videos")
async def videos(payload: dict):
    return _submit(payload, kind="video")


@app.post("/v1/images/generations")
async def images(payload: dict):
    return _submit(payload, kind="image")


@app.get("/v1/jobs/{job_id}")
def job_status(job_id: str):
    job = _JOBS.get(job_id)
    if not job:
        return JSONResponse({"error": "no such job"}, status_code=404)
    try:
        status = fal_client.status(job["fal_app"], job["handle"], with_logs=False)
    except Exception as e:
        return {"id": job_id, "status": "failed", "error": str(e)}

    name = type(status).__name__  # Queued | InProgress | Completed
    if name == "Completed":
        result = fal_client.result(job["fal_app"], job["handle"])
        url = _first_media_url(result)
        if not url:
            return {"id": job_id, "status": "failed", "error": "no media in result"}
        fid = "out-" + job_id
        with httpx.Client(timeout=300) as c:
            r = c.get(url)
            r.raise_for_status()
            _FILES[fid] = (r.content, r.headers.get("content-type", "application/octet-stream"))
        return {"id": job_id, "status": "succeeded", "phase": "done",
                "progress": 1.0, "file_id": fid, "error": None}
    if name == "InProgress":
        return {"id": job_id, "status": "running", "phase": "generating",
                "progress": 0.5, "file_id": None, "error": None}
    return {"id": job_id, "status": "queued", "phase": "queued",
            "progress": 0.0, "file_id": None, "error": None}


# --------------------------------------------------------------------------
# fal mapping
# --------------------------------------------------------------------------
def _submit(payload: dict, kind: str):
    mid = payload.get("model")
    spec = MODELS.get(mid)
    if not spec:
        return JSONResponse({"error": f"unknown model {mid!r}"}, status_code=400)

    args = _to_fal_args(payload, spec, kind)
    fal_app = spec["fal"]
    # image-to-video uses a different fal endpoint when an init image is present.
    if kind == "video" and (payload.get("image_file_id") or payload.get("image_b64")) \
            and spec.get("fal_i2v"):
        fal_app = spec["fal_i2v"]
        args["image_url"] = _data_url(payload)

    handle = fal_client.submit(fal_app, arguments=args)
    job_id = f"{kind}-{uuid.uuid4().hex}"
    _JOBS[job_id] = {"handle": handle, "fal_app": fal_app, "kind": kind, "model_id": mid}
    return {"id": job_id, "status": "queued"}


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
    import base64
    if payload.get("image_file_id") and payload["image_file_id"] in _FILES:
        data, ctype = _FILES[payload["image_file_id"]]
        return f"data:{ctype};base64," + base64.b64encode(data).decode()
    if payload.get("image_b64"):
        return "data:image/png;base64," + payload["image_b64"]
    return ""


def _first_media_url(result: dict) -> str:
    """Find a media URL in a fal result (shapes vary by model)."""
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
