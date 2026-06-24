# ComfyUI workflow models

Drop ComfyUI **API-format** workflow exports here. `comfyui_adapter.py` registers
each `*.json` as a remote model, so it shows up in Pallaidium as
`[Remote] <name>` after you click *Refresh Remote Models*.

## 1. Export a workflow from ComfyUI

1. In ComfyUI: **Settings → enable "Dev mode"** (adds API-export buttons).
2. Build/open the workflow you want (e.g. your SCAIL-2 character replacement,
   Bernini-R video edit, Stable Audio, or Boogu edit graph).
3. **Save (API Format)** → save the `.json` into **this folder**.
   The file name (without `.json`) becomes the model id and default label, e.g.
   `bernini-r.json` → `[Remote] bernini-r`.

The adapter also accepts a normal UI export that wraps the graph under a
`"prompt"` key, but API format is preferred.

## 2. Mark the nodes the adapter should fill in

The adapter injects the request's parameters by **node title** (in ComfyUI:
right-click a node → *Title*). Rename nodes to:

| Title | Node it goes on | Filled with |
|---|---|---|
| `prompt` (or `positive`) | the positive CLIPTextEncode / text node | the prompt |
| `negative` | the negative text node | the negative prompt |
| `width` / `height` | EmptyLatent / size node | resolution |
| `steps` | sampler | inference steps |
| `cfg` | sampler | guidance scale |

Also handled automatically (no title needed):

- **Seed** — any node with a `seed` or `noise_seed` input is set to the request seed.
- **Reference image(s)** — every `LoadImage` node receives an uploaded image, in
  title order (`image`, `image2`, …). One LoadImage → img2img / i2v; several →
  multi-reference (e.g. a character reference). Leave the LoadImage's own default
  in place for txt2-only runs (it's only overwritten when an image is sent).
- **Source video** — for **video edit / character replacement**, a `LoadVideo`
  node (`VHS_LoadVideo` / `VHS_LoadVideoPath` / `VHS_LoadVideoUpload` /
  `VHS_LoadVideoFFmpeg` / core `LoadVideo`) receives the source video you select
  in the VSE. The adapter advertises such a workflow with the `control` mode, so
  Pallaidium shows the **video input** picker; the chosen strip is uploaded and
  bound to the first video loader. Character replacement = `LoadVideo` (source)
  **+** `LoadImage` (the new character).

Anything you don't title is left exactly as saved, so the workflow runs as designed.

## 3. Media type is auto-detected

From the workflow's **output node**:

- `SaveImage` / `PreviewImage` → **image**
- `VHS_VideoCombine` / `SaveAnimatedWEBP` / `SaveWEBM` / `SaveVideo` → **video**
  (use VideoHelperSuite's `VHS_VideoCombine` with format `video/h264-mp4` for mp4)
- `SaveAudio` / `SaveAudioMP3` / `SaveAudioOpus` → **audio**

`i2i`/`i2v` is enabled when the graph has a `LoadImage` node.

## 4. Optional overrides — `<id>.meta.json`

Put a sidecar next to the workflow to override the label or detected values, e.g.
`bernini-r.meta.json`:

```json
{
  "display_name": "Bernini-R: Video Edit",
  "type": "video",
  "modes": ["i2v"],
  "max_ref_images": 1,
  "default_steps": 30
}
```

Use this to set the pretty names from your screenshots (the file name can't
contain `:`), or to correct anything the auto-detector got wrong.

### Explicit node bindings (for complex graphs)

When a workflow's parameters don't live on conventionally-titled nodes — e.g.
the prompt sits on a `PrimitiveStringMultiline`, width/height are `PrimitiveInt`
nodes using a `value` input, an LLM **prompt-enhance** chain feeds the encoder,
or there are several identically-titled `CLIPTextEncode` nodes — titling can't
disambiguate them. Add a **`bindings`** map to the sidecar that names the exact
`{node, input}` each request field maps to (node id = the JSON key, e.g.
`"320:319"`; input = the key inside that node's `inputs`):

```json
{
  "display_name": "LTX-Video 2.3 i2v",
  "type": "video",
  "modes": ["i2v"],
  "max_ref_images": 1,
  "bindings": {
    "prompt":   {"node": "320:319", "input": "value"},
    "negative": {"node": "320:313", "input": "text"},
    "width":    {"node": "320:312", "input": "value"},
    "height":   {"node": "320:299", "input": "value"},
    "fps":      {"node": "320:300", "input": "value"}
  }
}
```

- Bindable fields: `prompt`, `negative`, `width`, `height`, `steps`, `cfg`,
  `fps`, `num_frames`, `strength`, `seed`. A field you omit (or that the request
  didn't supply) leaves that node at its saved value.
- A field may map to **several** nodes — use a list:
  `"seed": [{"node": "a", "input": "noise_seed"}, {"node": "b", "input": "noise_seed"}]`.
- Bindings **override** title-matching, and the automatic handling still applies:
  the **source image** is uploaded and bound to `LoadImage`, the **source video**
  to a `LoadVideo`, and any `seed`/`noise_seed` input is set even without an
  explicit binding.
- Find node ids by opening the API-format JSON — each top-level key (`"320:319"`,
  `"269"`, …) is a node id; its `class_type` and `_meta.title` tell you which is
  which.

A complete worked example ships in this folder: [`ltx-2.3-i2v.json`](ltx-2.3-i2v.json)
+ [`ltx-2.3-i2v.meta.json`](ltx-2.3-i2v.meta.json).

## 5. Use it

Restart the adapter (it scans this folder at startup), then in Pallaidium click
**Refresh Remote Models**. Your workflows appear in the matching dropdown
(Movie / Image / Audio).
