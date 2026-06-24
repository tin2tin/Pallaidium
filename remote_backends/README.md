# Pallaidium Remote Backends (test/reference servers)

Standalone servers that implement Pallaidium's **Backend Contract v0.1**. They
are **separate from the add-on** — Pallaidium contains no provider code and only
talks to whatever URL you configure, so these are swappable and optional.

Contract version targeted: **v0.1** (+ the additive reference fields in the
add-on's `docs/BACKEND_CONTRACT_EXTENSIONS.md`).

## 1. `mock_backend.py` — local mock (free, no deps)

Exercises the *entire* contract on your own machine: discovery, async
image/video/audio jobs, `/v1/files` upload, progress, cancellation, and
transcription. Returns small canned media (real PNG/WAV; real MP4 if `ffmpeg` is
installed, otherwise a placeholder).

```bash
python mock_backend.py --port 8000
```

In Pallaidium → Preferences → Remote Backend:
- **Remote Backend URL** = `http://localhost:8000`
- **Model Source** = Remote (or Local & Remote)
- click **Refresh Remote Models** → `mock-image / mock-video / mock-tts / mock-asr`

Use this for development and CI. For genuine video, install `ffmpeg` or use a
real backend below.

## 2. `fal_adapter.py` — reference cloud adapter (fal.ai)

Forwards the contract to **fal.ai** so you can generate with real models such as
ByteDance **Seedance** video and FLUX image. This is a reference — adjust the
`MODELS` table and per-model argument mapping to the fal endpoints you use.

```bash
pip install -r requirements.txt
export FAL_KEY=...                 # PowerShell: $env:FAL_KEY="..."  |  cmd: set FAL_KEY=...
uvicorn fal_adapter:app --port 8000
```

Then point the add-on at `http://localhost:8000` and Refresh as above.

## 3. `comfyui_adapter.py` — local ComfyUI adapter

Forwards the contract to a running **ComfyUI** server, mapping each model to a
workflow graph. Built-in templates: SDXL / SD 1.5 txt2img + img2img, and
**LTX-Video** txt2video + image-to-video. Pallaidium's prompt / size / steps /
guidance / seed / init image are injected into the graph; ComfyUI runs locally so
there is no per-image cloud cost.

`COMFYUI_URL` defaults to `http://127.0.0.1:8188`, so if ComfyUI is on the
default port you can **skip setting it** entirely:

```bash
pip install -r requirements.txt
uvicorn comfyui_adapter:app --port 8000
```

To point at a non-default ComfyUI address, set `COMFYUI_URL` first — the syntax
depends on your shell:

```bash
# Linux / macOS (bash)
export COMFYUI_URL=http://127.0.0.1:8188
```
```powershell
# Windows PowerShell
$env:COMFYUI_URL="http://127.0.0.1:8188"
```
```bat
:: Windows Command Prompt (cmd.exe) — no spaces around =, no quotes
set COMFYUI_URL=http://127.0.0.1:8188
```

Then run `uvicorn comfyui_adapter:app --port 8000`, point the add-on at
`http://localhost:8000`, and Refresh as above. (On startup the adapter prints
whether it could reach ComfyUI.)

### Adding your own models = dropping in workflow files (recommended)

Every ComfyUI "model" is a **workflow graph**, so the adapter loads them from
files. Export a workflow from ComfyUI in **API format** (Settings → enable Dev
mode → *Save (API Format)*) into **`comfyui_workflows/`** next to the adapter.
Each `<id>.json` becomes a `[Remote] <id>` model; its media type is auto-detected
from the output node and your prompt / negative / seed / size / reference
image(s) are injected **by node title**. Optional `<id>.meta.json` sets a pretty
display name and overrides. Full convention: see
[`comfyui_workflows/README.md`](comfyui_workflows/README.md). This is how to wire
custom models such as character-replacement, video-edit, image-edit, or audio
graphs — no adapter code changes needed.

### Built-in example templates

The adapter also ships hardcoded reference graphs: `[Remote] SDXL Base`,
`[Remote] SD 1.5` (txt2img + img2img), and `[Remote] LTX-Video 2B` (txt2video +
i2v). For these you need the named checkpoints in `models/checkpoints/`.

In ComfyUI you must have the models each workflow/template uses, plus — for any
mp4 video output — the
[VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)
custom node (`VHS_VideoCombine`).

### Swapping providers

The contract is provider-agnostic. To target Replicate, Atlas Cloud, a
self-hosted server, etc., write another adapter that exposes the same `/v1/*`
endpoints — **no changes to Pallaidium are needed**. You can even run several
adapters and switch by changing the Remote Backend URL.

## Endpoints implemented

`GET /v1/health`, `GET /v1/models`, `POST /v1/videos`,
`POST /v1/images/generations`, `POST /v1/audio/speech`,
`POST /v1/audio/transcriptions`, `POST /v1/files`, `GET /v1/files/{id}`,
`GET /v1/jobs/{id}`.

A backend only needs the endpoints for the capabilities it offers.
