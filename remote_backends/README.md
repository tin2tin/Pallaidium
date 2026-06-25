# Pallaidium Remote Backends (one-click connectors)

Pallaidium can generate through an **external backend** that speaks a small
OpenAI-style `/v1` contract ("Backend Contract v0.1"). The add-on stays
provider-agnostic — it only ever talks HTTP to a URL — so backends are swappable.

The connectors here are **stdlib-only**: the add-on launches one for you with
Blender's own Python. **No `pip install`, no console.**

## One-click flow (in Blender)

1. **Edit → Preferences → Add-ons → Pallaidium → Remote Backend.**
2. Set **Model Source** to *Remote* (or *Local & Remote*).
3. Pick an **Adapter** from the dropdown:
   - **Mock** — tiny canned media; verifies the wiring with zero setup.
   - **ComfyUI** — forwards to a running ComfyUI (set the ComfyUI URL field; start ComfyUI first).
   - **fal.ai** — cloud models; paste your key in **Remote Backend Key**.
   - **Custom URL** — connect to a backend you started yourself (type its URL).
4. Click **Start Backend**. The add-on launches the adapter on a free port, fills
   in the URL, and loads the models. Generate as usual; click **Stop Backend** when done.

The adapter's log is written to `<Blender DATAFILES>/Pallaidium/adapter_<id>.log`.

## Adding ComfyUI workflows (no console)

Every ComfyUI "model" is a **workflow graph**. With the ComfyUI adapter selected:

- **Import Workflow** — pick a workflow exported from ComfyUI with
  *Settings → enable Dev mode → Save (API Format)*. It's copied into
  `comfyui_workflows/` and becomes a `[Remote] <filename>` model (the backend
  auto-reloads if it's running).
- **Open Folder** — opens `comfyui_workflows/` to manage files directly.

Prompt / negative / size / steps / cfg / seed / reference image(s) are injected
into the graph **by node title**; complex graphs use a `<id>.meta.json` sidecar
with explicit `bindings`. Full convention: [`comfyui_workflows/README.md`](comfyui_workflows/README.md).

## The three bundled connectors

| File | What it does | Config |
|---|---|---|
| `mock_backend.py` | Canned PNG/WAV/MP4 for every contract path | none |
| `comfyui_adapter.py` | Maps each model to a ComfyUI workflow graph | ComfyUI URL |
| `fal_adapter.py` | Forwards to fal.ai's queue REST API | `FAL_KEY` |

Each ships a `<name>.manifest.json` that the add-on reads to build the Adapter
dropdown and its config fields. You can still run any of them by hand:

```bash
python comfyui_adapter.py --port 8000 --comfyui-url http://127.0.0.1:8188
python fal_adapter.py     --port 8000 --fal-key YOUR_KEY
python mock_backend.py    --port 8000
```

…then choose **Custom URL** and point the add-on at `http://localhost:8000`.

## Write your own connector

1. Create `myservice_adapter.py` next to these. Implement the contract endpoints
   you need (`GET /v1/health`, `GET /v1/models`, `POST /v1/images/generations`
   and/or `/v1/videos` / `/v1/audio/speech`, `GET /v1/jobs/{id}`, `POST /v1/files`,
   `GET /v1/files/{id}`). The shared helpers in [`_adapter_http.py`](_adapter_http.py)
   give you a stdlib `BaseAdapterHandler`, multipart parsing, and `urllib` JSON/
   bytes/multipart calls — copy the structure of `comfyui_adapter.py`.
2. Add a `myservice_adapter.manifest.json`:
   ```json
   {
     "id": "myservice",
     "label": "My Service",
     "entry": "myservice_adapter.py",
     "order": 3,
     "description": "What it does.",
     "default_port": 8000,
     "config_fields": [
       {"key": "MYSERVICE_KEY", "arg": "--key", "pref": "remote_backend_key",
        "label": "API Key", "type": "secret", "default": ""}
     ],
     "supports_workflow_import": false
   }
   ```
   `config_fields[].pref` names the Blender preference the value comes from
   (`remote_backend_key` for a secret, or `comfyui_url`); each becomes both an
   env var (`key`) and, if `arg` is set, a CLI flag.
3. Restart Blender — your connector appears in the Adapter dropdown. **No add-on
   code changes needed.** Keep it stdlib-only so the add-on can launch it.

## Contract endpoints

`GET /v1/health`, `GET /v1/models`, `POST /v1/videos`,
`POST /v1/images/generations`, `POST /v1/audio/speech`,
`POST /v1/audio/transcriptions`, `POST /v1/files`, `GET /v1/files/{id}`,
`GET /v1/jobs/{id}`. A backend only needs the endpoints for the capabilities it
offers. See [`../docs/BACKEND_CONTRACT_EXTENSIONS.md`](../docs/BACKEND_CONTRACT_EXTENSIONS.md)
for discovery hints (`display_name`, `max_ref_images`, `default_steps`, …).
