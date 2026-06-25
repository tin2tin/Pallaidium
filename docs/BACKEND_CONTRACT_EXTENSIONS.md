# Pallaidium Backend Contract — Extensions to v0.1

Pallaidium drives a remote backend that speaks the OpenAI-`/v1`-dialect
**Backend Contract v0.1**. That base contract covers the common cases
(text→media, a single first-frame `image_b64`, one control file, and a single
`speaker_reference_id`).

Pallaidium can send richer inputs than v0.1 names — multiple reference images,
img2img on the image endpoint, reference transcripts for voice cloning, etc.
Those travel through the contract's existing **`POST /v1/files`** upload
mechanism plus a small set of **additive, optional** body fields documented here.

A backend only needs to implement the fields for the capabilities it offers. Any
field it doesn't understand it can ignore — the corresponding model simply won't
expose that input. None of this breaks a v0.1-only backend.

---

## Tutorial: set up a remote backend

This walks you from nothing to generating with a remote model. It uses the
included **local mock** first (zero cost, no account), then shows the **cloud
adapter** and **ComfyUI**. Everything runs on your own computer, and Pallaidium
launches the connectors for you — **no console, no `pip install`**.

> The example connectors ship in the project's **`remote_backends/`** folder
> (mock + ComfyUI + fal.ai + their README). They are **stdlib-only** and **not**
> part of the installable add-on (excluded from the built extension) — Pallaidium
> only ever talks to a URL. Pallaidium starts the one you pick using Blender's own
> Python; you can still run them by hand (see § B/C) and choose **Custom URL**.

### A. Try it with the local mock (recommended first)

1. In Blender: **Edit → Preferences → Add-ons → Pallaidium**, open the **Remote
   Backend** box and set:
   - **Model Source** → `Remote` (or `Local & Remote` to keep local models too)
   - **Adapter** → `Mock (local test, no deps)`

2. Click **Start Backend**. Pallaidium launches the mock on a free port, fills in
   the URL, and reports e.g. “Mock started — 4 model(s) loaded”. The models appear
   in the dropdowns prefixed `[Remote]` (`mock-image`, `mock-video`, `mock-tts`,
   `mock-asr`). (Install `ffmpeg` if you want real playable video from the mock.)

3. **Generate.** In the VSE sidebar (**N-panel → Generative AI**): pick an Output
   type, choose a `[Remote]` model, type a prompt, and click **Generate**. Watch
   the queue panel show *generating* / progress; the result is downloaded and
   added to the timeline. **Cancel** mid-job ends it as *Cancelled*. Click **Stop
   Backend** when finished.

If Start reports an error, the message (and the log at
`<Blender DATAFILES>/Pallaidium/adapter_mock.log`) tells you what failed.

### B. Switch to a real cloud backend (e.g. Seedance video via fal.ai)

The included **fal.ai** connector forwards the contract to fal's cloud:

1. **Model Source** → `Remote`, **Adapter** → `fal.ai (cloud)`.
2. Paste your fal key into **Remote Backend Key** (it is passed to the connector
   as `FAL_KEY`; it is never sent to Blender's own servers).
3. Click **Start Backend**, then generate as in § A.3 — `[Remote] FLUX`,
   `[Remote] Seedance`, etc. now produce real cloud results.

Prefer to run it yourself? `python fal_adapter.py --port 8000 --fal-key YOUR_KEY`,
then choose **Custom URL** = `http://localhost:8000`. To target a different
provider, drop a stdlib `<name>_adapter.py` + `<name>.manifest.json` into
`remote_backends/` and it appears in the Adapter dropdown — **no add-on changes**.

### C. Run models locally with ComfyUI

If you already use **ComfyUI**, the included connector drives it through the
contract — fully local, no cloud cost. In ComfyUI every "model" is a *workflow
graph*, so you add models by importing **workflow files**.

1. **Start ComfyUI** (default `http://127.0.0.1:8188`).

2. In Pallaidium: **Model Source** → `Remote`, **Adapter** → `ComfyUI (local)`.
   Set **ComfyUI URL** if yours isn't on the default address, then click **Start
   Backend**. The connector's log notes whether it reached ComfyUI.

3. **Add your models as workflow files.** In ComfyUI: **Settings → enable
   "Dev mode"**, build/open a workflow, then **Save (API Format)**. Back in
   Pallaidium click **Import Workflow** and pick that `.json` (or **Open Folder**
   and drop files into `remote_backends/comfyui_workflows/` yourself). Each
   `<id>.json` becomes a `[Remote] <id>` model and the running backend reloads
   automatically. The adapter:
   - detects the **media type** from the output node (`SaveImage` → image,
     `VHS_VideoCombine`/`SaveVideo`/`SaveAnimatedWEBP`/`SaveWEBM` → video,
     `SaveAudio*` → audio);
   - fills in your request **by node title** — rename nodes (right-click →
     *Title*) to `prompt`, `negative`, `width`, `height`, `steps`, `cfg`;
   - sets the **seed** on any `seed`/`noise_seed` input automatically, and routes
     uploaded **reference image(s)** to each `LoadImage` node in order (one =
     img2img/i2v, several = multi-reference);
   - for **video edit / character replacement**, routes the selected source
     **video** to a `LoadVideo` node (`VHS_LoadVideo*` / core `LoadVideo`); such a
     workflow is advertised with `control` mode so Pallaidium shows the video
     input picker. Character replacement = source `LoadVideo` + a `LoadImage`
     for the new character;
   - an optional `<id>.meta.json` sidecar sets a pretty display name (e.g. with a
     `:` the file name can't contain) and overrides the detected type/modes/hints.

   The built-in templates (`[Remote] SDXL Base`, `[Remote] SD 1.5`,
   `[Remote] LTX-Video 2B`) work too if you have those checkpoints. mp4 video
   output needs the **VideoHelperSuite** custom node (`VHS_VideoCombine`).
   Full convention: `remote_backends/comfyui_workflows/README.md`.

   The imported model appears right away (the running backend reloads); otherwise
   click **Refresh Models**. Generate as in § A.3 — your ComfyUI workflows now run
   from the VSE.

### D. Configure without the UI (optional)

This applies to the **Custom URL** adapter (when you start a connector yourself).
Instead of the preference fields you can set environment variables before
launching Blender; the URL/key fields fall back to them when left empty:

```bash
# Linux / macOS (bash)
export PALLAIDIUM_BACKEND_URL=http://localhost:8000
export PALLAIDIUM_BACKEND_KEY=          # optional
```
```powershell
# Windows PowerShell
$env:PALLAIDIUM_BACKEND_URL="http://localhost:8000"
$env:PALLAIDIUM_BACKEND_KEY=""          # optional
```
```bat
:: Windows Command Prompt (cmd.exe) — no spaces around =, no quotes
set PALLAIDIUM_BACKEND_URL=http://localhost:8000
set PALLAIDIUM_BACKEND_KEY=             :: optional
```

### Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Start Backend fails immediately | See the connector log at `<Blender DATAFILES>/Pallaidium/adapter_<id>.log` |
| “Could not locate Blender's Python …” | Rare; use **Custom URL** and start a connector by hand instead |
| Refresh: “Backend speaks […], add-on needs 'v0.1'” | Backend advertises a different `contract_versions` |
| Dropdown shows “No remote … — click Refresh” | You switched to Remote but haven’t started/refreshed yet |
| fal.ai: “FAL_KEY is not set” | Paste your key into **Remote Backend Key**, then **Start Backend** |
| Video strip won’t play (mock) | Mock placeholder; install `ffmpeg` or use a real backend |
| ComfyUI: workflow not listed | Must be **API-format** export; use **Import Workflow** (it reloads the backend) |
| ComfyUI: prompt/seed ignored | Title the nodes (`prompt`, `negative`, …); seed needs a `seed`/`noise_seed` input present |
| ComfyUI: "no media in outputs" | Workflow has no recognised save node, or a node/model is missing in ComfyUI |

---

## 1. Discovery hints on `GET /v1/models`

Each model entry is `{"id", "type", "modes"}` in v0.1. Pallaidium reads these
**optional** extra keys to build the right UI without per-model code:

| Key | Type | Meaning |
|---|---|---|
| `display_name` | string | Friendlier label (else the `id` is used) |
| `description` | string | Tooltip text |
| `max_ref_images` | int | Enable N reference-image pickers (e.g. `9` for Klein or Seedance reference-to-video); `1` = single img2img/i2v. Works for **image and video** models. |
| `needs_speaker_ref` | bool | TTS model accepts a reference voice sample |
| `needs_ref_text` | bool | TTS model accepts a reference transcript |
| `needs_audio_ref` | bool | **Video** model accepts a reference audio clip (e.g. Seedance reference-to-video). Shows an Audio Ref. picker. |
| `supports_audio_output` | bool | **Video** model can generate a soundtrack. Shows a *Generate Audio* toggle whose state is sent as `generate_audio`. |
| `control_types` | list[str] | Allowed control maps, e.g. `["canny","depth","pose"]` |
| `default_steps` | int | Default inference steps shown in the UI |
| `default_guidance` | float | Default guidance scale |
| `max_width` / `max_height` | int | Default/maximum resolution |

`type` ∈ `video | image | audio | text`. `modes` may include
`t2i/i2i`, `t2v/i2v/control`, `tts`, `transcription`.

Example:

```json
{ "data": [
  { "id": "flux-klein", "type": "image", "modes": ["t2i","i2i"],
    "max_ref_images": 9, "default_steps": 28 },
  { "id": "seedance-2.0", "type": "video", "modes": ["t2v","i2v","control"],
    "control_types": ["canny","depth","pose"], "supports_audio_output": true },
  { "id": "seedance-2-mini-ref", "type": "video", "modes": ["i2v","control"],
    "max_ref_images": 9, "needs_audio_ref": true, "supports_audio_output": true },
  { "id": "xtts", "type": "audio", "modes": ["tts"],
    "needs_speaker_ref": true, "needs_ref_text": true },
  { "id": "whisper-large", "type": "text", "modes": ["transcription"] }
] }
```

---

## 2. File uploads — `POST /v1/files`

All binary references are uploaded as `multipart/form-data` with a `purpose`
and the add-on then passes the returned `file_id` in the generation body:

| `purpose` | Used for |
|---|---|
| `reference` | init / reference / last / anchor / IP-Adapter images, source video, **reference audio for reference-to-video** |
| `control` | motion/structure control video |
| `speaker_reference` | voice-clone reference audio (TTS) |

Response: `{ "file_id": "<id>" }`.

---

## 3. Additive body fields

Sent on the generation endpoints (`/v1/videos`, `/v1/images/generations`,
`/v1/audio/speech`) in addition to the v0.1 fields. All are optional.

| Field | Type | Endpoint(s) | Meaning |
|---|---|---|---|
| `image_file_id` | string | images, videos | Single init image (img2img / img2vid). On videos, `image_b64` is also sent for v0.1 compatibility. |
| `reference_file_ids` | list[str] | images | Multiple reference images (multi-ref / triple-image edit) |
| `reference_prompts` | list[str] | images | Per-reference-image prompts (e.g. OmniGen) |
| `last_image_file_id` | string | videos | Last-frame condition (FLF) |
| `anchor_file_ids` | list[{file_id, fraction}] | videos | Interior keyframe anchors at timeline fractions |
| `reference_file_ids` | list[str] | videos | Multiple reference images for reference-to-video (e.g. Seedance Mini reference-to-video → fal `image_urls`) |
| `reference_audio_ids` | list[str] | videos | Reference audio clip(s) for reference-to-video (→ fal `audio_urls`). Uploaded with `purpose: "reference"`. |
| `generate_audio` | bool | videos | Request a generated soundtrack. Sent for models that advertise `supports_audio_output`; mirrors the *Generate Audio* toggle. |
| `control_file_id` | string | videos | Control / source video (v0.1). Used as the source for video-edit & character-replacement workflows. |
| `video_file_id` | string | videos | Source video alias accepted by some adapters (e.g. ComfyUI) when not used as a control map |
| `control_type` | string | videos | One of the model's `control_types` (v0.1) |
| `control_strength` | float | videos | Control influence (v0.1) |
| `speaker_reference_id` | string | audio | Voice-clone reference sample (v0.1) |
| `speaker_reference_text` | string | audio | Transcript of the reference sample |
| `ip_face_file_ids` | list[str] | images | IP-Adapter face images |
| `ip_style_file_ids` | list[str] | images | IP-Adapter style images |

Standard generation fields used as in v0.1: `model`, `prompt`,
`negative_prompt`, `width`, `height`, `num_frames`, `fps`, `seed`, `strength`,
`num_inference_steps`, `guidance_scale`, and for audio `input`, `voice`,
`response_format`, `speed`.

---

## 4. Responses

Unchanged from v0.1. The add-on accepts, in order:

1. async job — `{ "id", "status": "queued" }`, polled at `GET /v1/jobs/{id}`
   (`status` ∈ `queued|running|succeeded|failed`, plus `phase`, `progress`
   0–1, and `file_id` on success), then `GET /v1/files/{id}` for the bytes;
2. direct `{ "file_id" }`;
3. OpenAI-style `{ "data": [ { "url" | "b64_json" } ] }` (or top-level `url` /
   `b64_json`).

Transcription (`POST /v1/audio/transcriptions`, multipart) returns
`{ "text": "..." }` and is inserted as a text strip.

---

## 5. Versioning

`GET /v1/health` → `{ "status": "ok", "contract_versions": ["v0.1"] }`.
Pallaidium currently speaks `v0.1`; if a backend advertises versions that don't
include it, the add-on refuses with a clear message. The extension fields above
are layered **on top of** `v0.1` and do not change the version string.
