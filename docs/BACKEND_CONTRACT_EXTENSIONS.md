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
adapter** for real models. Everything runs on your own computer.

> The example servers ship in the project's **`remote_backends/`** folder (mock +
> fal.ai adapter + their README). They are **not** part of the installable add-on
> (excluded from the built extension and never imported by Blender) — Pallaidium
> only talks to a URL. Run the commands below *from that folder*. It sits next to
> this `docs/` folder inside the Pallaidium source, e.g.:
>
> - find your Pallaidium install path (shown in Blender → Preferences → Add-ons →
>   Pallaidium → the file path under the add-on name), then `cd` into its
>   `remote_backends` subfolder, **or**
> - copy the `remote_backends/` folder anywhere with Python installed and run it
>   there. It only needs Python 3 (the mock has no dependencies).

### A. Try it with the local mock (recommended first)

1. **Start the mock server** (stdlib only, no install needed) — from the
   `remote_backends` folder:
   ```bash
   cd <pallaidium>/remote_backends      # the folder beside docs/
   python mock_backend.py --port 8000
   ```
   On Windows the path looks like
   `…\extensions\user_default\pallaidium_generative_ai\remote_backends`.
   It prints `Mock backend on http://127.0.0.1:8000`. (Install `ffmpeg` if you
   want it to return a real playable video; otherwise video is a placeholder.)

2. **Point Pallaidium at it.** In Blender: **Edit → Preferences → Add-ons →
   Pallaidium**, open the **Remote Backend** box and set:
   - **Model Source** → `Remote` (or `Local & Remote` to keep local models too)
   - **Remote Backend URL** → `http://localhost:8000`
   - **Remote Backend Key** → leave empty (the mock needs none)

3. **Load the models.** Click **Refresh Remote Models**. You should see a report
   like “Loaded 4 remote model(s)”. They appear in the dropdowns prefixed
   `[Remote]` (`mock-image`, `mock-video`, `mock-tts`, `mock-asr`).

4. **Generate.** In the VSE sidebar (**N-panel → Generative AI**): pick an Output
   type, choose a `[Remote]` model, type a prompt, and click **Generate**. Watch
   the queue panel show *generating* / progress; the result is downloaded and
   added to the timeline. **Cancel** mid-job ends it as *Cancelled*.

If Refresh reports an error, the message tells you what failed (bad URL,
unreachable server, or an unsupported contract version).

### B. Switch to a real cloud backend (e.g. Seedance video via fal.ai)

The contract is provider-agnostic; a small **adapter** translates it to a
provider. The included reference adapter targets **fal.ai**:

1. **Install and run the adapter** (from the `remote_backends` folder):
   ```bash
   cd <pallaidium>/remote_backends
   pip install -r requirements.txt
   export FAL_KEY=...        # PowerShell: $env:FAL_KEY="..."  |  cmd: set FAL_KEY=...
   uvicorn fal_adapter:app --port 8000
   ```
   The API key stays here, in the adapter — never in Blender.

2. In Pallaidium, keep **Remote Backend URL** = `http://localhost:8000`, click
   **Refresh Remote Models**, and generate as in step A.4. Now `[Remote] Seedance`
   etc. produce real cloud results.

To use a different provider (Replicate, Atlas Cloud, a self-hosted server, …),
run an adapter exposing the same `/v1/*` endpoints and change only the URL —
**no add-on changes needed**.

### C. Run models locally with ComfyUI

If you already use **ComfyUI**, the included `comfyui_adapter.py` drives it
through the contract — fully local, no cloud cost. In ComfyUI every "model" is a
*workflow graph*, so you add models by dropping **workflow files** into a folder;
the adapter exposes each one and injects Pallaidium's prompt / size / seed /
reference image(s) into it.

1. **Start ComfyUI** (default `http://127.0.0.1:8188`).

2. **Run the adapter** (from the `remote_backends` folder). `COMFYUI_URL`
   defaults to `http://127.0.0.1:8188`, so on the default port you can skip
   setting it:
   ```bash
   cd <pallaidium>/remote_backends
   pip install -r requirements.txt
   uvicorn comfyui_adapter:app --port 8000
   ```
   For a non-default ComfyUI address, set `COMFYUI_URL` first — the syntax
   depends on your shell:
   ```bash
   export COMFYUI_URL=http://127.0.0.1:8188          # Linux / macOS (bash)
   ```
   ```powershell
   $env:COMFYUI_URL="http://127.0.0.1:8188"          # Windows PowerShell
   ```
   ```bat
   set COMFYUI_URL=http://127.0.0.1:8188             :: Windows cmd.exe (no spaces/quotes)
   ```
   The adapter prints on startup whether it could reach ComfyUI.

3. **Add your models as workflow files.** In ComfyUI: **Settings → enable
   "Dev mode"**, build/open a workflow, then **Save (API Format)** into
   `remote_backends/comfyui_workflows/`. Each `<id>.json` becomes a
   `[Remote] <id>` model. The adapter:
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

4. In Pallaidium, keep **Remote Backend URL** = `http://localhost:8000`, click
   **Refresh Remote Models**, and generate as in step A.4 — your ComfyUI
   workflows now run from the VSE.

### D. Configure without the UI (optional)

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
| Refresh: “connection failed” | Server not running, or wrong port/URL |
| Refresh: “Backend speaks […], add-on needs 'v0.1'” | Backend advertises a different `contract_versions` |
| Dropdown shows “No remote … — click Refresh” | You switched to Remote but haven’t refreshed yet |
| Remote models gone after restarting Blender | Expected — click **Refresh Remote Models** again |
| Video strip won’t play (mock) | Mock placeholder; install `ffmpeg` or use a real backend |
| ComfyUI: workflow not listed | Must be **API-format** export, in `comfyui_workflows/`; restart the adapter (it scans at startup) |
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
| `max_ref_images` | int | Enable N reference-image pickers (e.g. `9` for Klein); `1` = single img2img |
| `needs_speaker_ref` | bool | TTS model accepts a reference voice sample |
| `needs_ref_text` | bool | TTS model accepts a reference transcript |
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
    "control_types": ["canny","depth","pose"] },
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
| `reference` | init / reference / last / anchor / IP-Adapter images, source video |
| `control` | motion/structure control video |
| `speaker_reference` | voice-clone reference audio |

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
