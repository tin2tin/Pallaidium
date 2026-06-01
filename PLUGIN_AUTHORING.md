# Writing Plugins for Pallaidium Generative AI

Pallaidium discovers AI models at startup from a folder of plain Python files.
Adding a new model means dropping one `.py` file in the right folder — no
registration code, no `__init__.py` edits, no restart of the registry.

---

## Table of Contents

- [How Discovery Works](#how-discovery-works)
- [Quick Start](#quick-start)
- [File and Folder Layout](#file-and-folder-layout)
- [Class Attributes Reference](#class-attributes-reference)
  - [Identity](#identity)
  - [InputSpec — what data the plugin needs](#inputspec--what-data-the-plugin-needs)
  - [UISection — what controls to show](#uisection--what-controls-to-show)
  - [ParamSpec — parameter defaults](#paramspec--parameter-defaults)
  - [Capability Flags](#capability-flags)
  - [REQUIRED_PACKAGES](#required_packages)
- [Implementing load()](#implementing-load)
- [Implementing generate()](#implementing-generate)
- [ModelInputs Fields](#modelinputs-fields)
- [Available Helpers](#available-helpers)
- [Custom UI with draw_custom_ui()](#custom-ui-with-draw_custom_ui)
- [Multiple Variants from One Base Class](#multiple-variants-from-one-base-class)
- [Complete Examples](#complete-examples)
  - [Minimal image plugin (text-to-image)](#minimal-image-plugin-text-to-image)
  - [Image plugin with img2img and inpaint](#image-plugin-with-imgtoimg-and-inpaint)
  - [Video plugin](#video-plugin)
  - [Audio / TTS plugin](#audio--tts-plugin)
- [Testing and Debugging](#testing-and-debugging)

---

## How Discovery Works

When Blender loads Pallaidium, `models/__init__.py` calls `discover()`.
It scans every `.py` file under `models_plugins/`, imports each one, and
instantiates every class that:

- subclasses `ModelPlugin`
- has a non-empty `MODEL_ID`

Each instance is stored in `PLUGIN_REGISTRY[MODEL_ID]` and added to the
dropdown for its `MODEL_TYPE`. Files or directories whose names start with
`_` are silently skipped (use this for shared base classes or drafts).

> **You never need to edit any existing file.** Drop your `.py` in the right
> folder and restart Blender.

---

## Quick Start

1. Copy `models_plugins/_template.py` to the correct sub-folder.
2. Fill in the four identity attributes and implement `load()` + `generate()`.
3. Restart Blender (or use *Reload Scripts* in the Text Editor).

```
models_plugins/
    image/my_cool_model.py   ← new file, done
```

---

## File and Folder Layout

```
models_plugins/
    _template.py             ← starter template (ignored by loader)
    image/                   → plugins that produce images (.png / .jpg)
    video/                   → plugins that produce video (.mp4)
    audio/                   → plugins that produce audio (.wav)
    text/                    → plugins that produce a text strip
```

Put your file in the folder that matches what your model outputs.
One file may define multiple plugin classes (e.g. a base class plus
txt2vid / img2vid sub-classes that share a `load()` implementation).

Files named `_something.py` or inside directories named `_something/`
are ignored by the loader — use this convention for shared helpers or
work-in-progress files.

---

## Class Attributes Reference

### Identity

```python
MODEL_ID     = "author/my-model"          # unique key — HuggingFace repo ID is ideal
DISPLAY_NAME = "Image: My Model (1024)"   # shown in the dropdown
MODEL_TYPE   = "image"                    # "image" | "video" | "audio" | "text"
DESCRIPTION  = "Short tooltip text"       # shown as a tooltip in the UI
```

`MODEL_ID` must be unique across all plugins. Duplicate IDs are skipped
with a warning in the Blender console.

---

### InputSpec — what data the plugin needs

`INPUTS` is a bitflag that tells the framework which fields to populate in
the `ModelInputs` object passed to `generate()`.

```python
from ...models.base import InputSpec

INPUTS = InputSpec.PROMPT | InputSpec.IMAGE
```

| Flag | Field populated in `ModelInputs` | Notes |
|---|---|---|
| `PROMPT` | `inputs.prompt` | Main text prompt |
| `NEG_PROMPT` | `inputs.neg_prompt` | Negative prompt |
| `IMAGE` | `inputs.image` | `PIL.Image` from the selected strip |
| `MULTI_IMAGE` | `inputs.images` | List of `PIL.Image`; count set by `PARAMS.max_multi_images` |
| `AUDIO_REF` | `inputs.audio_ref` | Path to a reference `.wav` / `.mp3` (optional) |
| `AUDIO_REF_REQ` | `inputs.audio_ref` | Same, but UI marks it required |
| `TEXT_REF` | `inputs.text_ref` | Reference transcription string |
| `VIDEO` | `inputs.video_path` | Path to an input video file |
| `FACE_FOLDER` | `inputs.face_folder` | Path for IP-Adapter face images |
| `STYLE_FOLDER` | `inputs.style_folder` | Path for IP-Adapter style images |
| `LORA` | `inputs.lora_files` | List of `(path, weight)` tuples |
| `API_KEY` | *(read manually)* | Signals that an external API key is required |
| `HF_TOKEN` | *(read from prefs)* | Shows the HuggingFace token field; call `login()` in `load()` |

---

### UISection — what controls to show

`UI_SECTIONS` is a list of `UISection` values. The panel renders exactly
these sections, in the order listed.

```python
from ...models.base import UISection

UI_SECTIONS = [
    UISection.PROMPT,
    UISection.NEG_PROMPT,
    UISection.IMAGE_STRIP,
    UISection.RESOLUTION,
    UISection.STEPS,
    UISection.GUIDANCE,
    UISection.SEED,
]
```

| Value | What it renders |
|---|---|
| `PROMPT` | Main prompt textarea |
| `NEG_PROMPT` | Negative prompt textarea |
| `IMAGE_STRIP` | Single image strip eyedropper |
| `MULTI_IMAGES` | Dynamic add/remove image strip pickers |
| `TRIPLE_IMAGE` | Three fixed image strip pickers |
| `TRIPLE_PROMPT_IMG` | Three (prompt textarea + image picker) pairs |
| `AUDIO_REF` | Speaker reference file picker |
| `TEXT_REF` | Reference text input |
| `VIDEO_STRIP` | Video strip eyedropper |
| `RESOLUTION` | Width × Height dropdowns |
| `FRAMES` | Frame count slider |
| `STEPS` | Inference steps slider |
| `GUIDANCE` | Guidance / word-power slider |
| `IMAGE_STRENGTH` | img2img / inpaint strength slider |
| `SEED` | Seed input + randomise toggle |
| `LORA` | LoRA folder + weighted file list |
| `IP_ADAPTER` | Face folder + style folder pickers |
| `AUDIO_DURATION` | Duration slider |
| `SPEED` | Playback speed / CPS slider |
| `CHAT_PARAMS` | Exaggeration, pace, temperature sliders |
| `ILLUMINATION` | Lighting style + direction dropdowns |
| `POSE_TOGGLE` | "Read as OpenPose Rig Image" checkbox |
| `SCRIBBLE_TOGGLE` | "Read as Scribble Image" checkbox |
| `ENHANCE` | Quality / Speed / Faces / Upscale toggles |

Sections not in `UI_SECTIONS` are hidden — the user never sees controls
they cannot use with your model.

---

### ParamSpec — parameter defaults

Override only the fields that differ from the generic defaults.

```python
from ...models.base import ParamSpec

PARAMS = ParamSpec(
    width=1024,
    height=1024,
    steps=20,
    guidance=7.5,
    strength=0.8,          # img2img / inpaint strength
    max_multi_images=3,    # for MULTI_IMAGE input
)
```

| Field | Default | Description |
|---|---|---|
| `width` | 1024 | Output width in pixels |
| `height` | 576 | Output height in pixels |
| `frames` | 49 | Video frame count |
| `steps` | 25 | Inference steps |
| `guidance` | 7.5 | Guidance / CFG scale |
| `strength` | 0.8 | img2img / inpaint strength |
| `audio_length` | 5.0 | Audio duration in seconds |
| `max_multi_images` | 1 | Maximum strips for `MULTI_IMAGE` mode |

---

### Capability Flags

These four booleans tell the framework what modes the model supports.
The defaults are permissive; set `False` only when a feature truly does
not apply.

```python
supports_inpaint:          bool = True   # set False → inpaint mode never activated
supports_img2img:          bool = True   # set False → img2img conversion never activated
requires_input_strip:      bool = False  # set True  → always requires a selected strip
uses_standard_input_strip: bool = True   # set False → plugin draws its own strip UI
```

**Common patterns:**

```python
# ControlNet / conditioning model — always needs an image, no inpaint/img2img
supports_inpaint           = False
supports_img2img           = False
requires_input_strip       = True

# Background remover — processes a strip, no conversion modes
supports_inpaint           = False
supports_img2img           = False
uses_standard_input_strip  = False   # plugin's draw_custom_ui() handles the UI

# Cloud API — no local strip selection at all
uses_standard_input_strip  = False
```

---

### REQUIRED_PACKAGES

A list of Python package names checked by `is_available()`. If any is
missing the plugin is still registered, but Blender will prompt the user
to install it before generating.

```python
REQUIRED_PACKAGES = ["torch", "diffusers", "transformers"]
```

Use the top-level importable name (e.g. `"PIL"` not `"Pillow"`).

---

## Implementing load()

```python
def load(self, prefs, scene, **kwargs) -> dict:
    """Load and return the model pipeline.

    Called once; the result is cached by MODEL_ID for the lifetime of
    the Blender session.
    """
```

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `prefs` | `AddonPreferences` | Add-on preferences (HuggingFace token, output folder, etc.) |
| `scene` | `bpy.types.Scene` | The active scene (rarely needed in load) |
| `**kwargs` | dict | Extra context; `kwargs.get("mode")` is `"txt2img"` / `"img2img"` / `"inpaint"` |

**Return value:** any object — it is passed back as the first argument to
`generate()`. Conventionally a dict:

```python
return {"pipe": pipe, "converter": pipe, "refiner": None, "preprocessor": None}
```

**Memory management:** use `enable_model_cpu_offload()` or
`enable_sequential_cpu_offload()` so GPU memory is freed between runs.

```python
def load(self, prefs, scene, **kw):
    import torch
    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(self.MODEL_ID, torch_dtype=torch.bfloat16)

    if gfx_device == "mps":
        pipe.to("mps")
    elif low_vram():
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.enable_model_cpu_offload()

    return {"pipe": pipe, "converter": pipe, "refiner": None, "preprocessor": None}
```

---

## Implementing generate()

```python
def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
    """Run inference and return a PIL.Image (for image plugins) or a
    file path string (for video/audio/text plugins)."""
```

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `pipe_obj` | any | Whatever `load()` returned |
| `inputs` | `ModelInputs` | Collected inputs (only fields in `INPUTS` are populated) |
| `scene` | `bpy.types.Scene` | Active scene |
| `prefs` | `AddonPreferences` | Add-on preferences |

**Return value:**

- **Image plugins:** return a `PIL.Image.Image` — the framework saves it and adds it to the sequencer.
- **Video / Audio / Text plugins:** return an **absolute file path** as a string.

---

## ModelInputs Fields

Only fields declared in `INPUTS` are guaranteed to be populated.
Everything else stays at its default.

```python
# Text
inputs.prompt          # str
inputs.neg_prompt      # str
inputs.text_ref        # str — reference transcription (Qwen3-TTS)

# Mode (set by the dispatcher)
inputs.mode            # "txt2img" | "img2img" | "inpaint"

# Media
inputs.image           # PIL.Image or None — single image from strip
inputs.inpaint_mask    # PIL.Image or None — white = paint here
inputs.images          # list of PIL.Image — MULTI_IMAGE
inputs.audio_ref       # str path or None
inputs.video_path      # str path or None

# LoRA
inputs.lora_files      # list of (path, weight) tuples

# IP Adapter
inputs.face_folder     # str path or None
inputs.style_folder    # str path or None

# Generation parameters
inputs.width           # int
inputs.height          # int
inputs.frames          # int
inputs.steps           # int
inputs.guidance        # float
inputs.strength        # float
inputs.seed            # int

# Audio
inputs.audio_length    # float (seconds)
inputs.speed           # float
inputs.exaggeration    # float
inputs.pace            # float
inputs.temperature     # float

# Lighting (Kontext Relight)
inputs.illumination_style  # str
inputs.light_direction     # str
```

---

## Available Helpers

Import from `...utils.helpers`:

```python
from ...utils.helpers import gfx_device, low_vram, solve_path, clean_filename, \
                              find_strip_by_name, get_strip_path, load_first_frame
```

| Helper | Signature | Description |
|---|---|---|
| `gfx_device` | `str` | `"cuda"` / `"mps"` / `"cpu"` |
| `low_vram()` | `() → bool` | `True` when VRAM < 8 GB |
| `solve_path(filename)` | `str → str` | Builds an absolute output path inside the user's Pallaidium media folder |
| `clean_filename(text)` | `str → str` | Strips characters that are invalid in file names |
| `find_strip_by_name(scene, name)` | `(scene, str) → Strip\|None` | Finds a sequencer strip by name |
| `get_strip_path(strip)` | `Strip → str` | Returns the absolute file path of an image or movie strip |
| `load_first_frame(path)` | `str → PIL.Image` | Opens the first frame of an image or video file as a `PIL.Image` |

---

## Custom UI with draw_custom_ui()

If your plugin needs controls that no standard `UISection` covers, override
`draw_custom_ui()`.

```python
def draw_custom_ui(self, col, context) -> bool:
    """
    col     — a Blender UILayout column inside the input-selector box.
    context — bpy.context

    Return True  if you completely replaced the standard input_strips
                 dropdown (like OmniGen's triple-prompt layout).
    Return False if you only added extra controls below the dropdown
                 (or added nothing at all).
    """
    scene = context.scene
    if scene.sequence_editor is None:
        return False

    row = col.row(align=True)
    row.prop_search(
        scene, "my_custom_strip",
        scene.sequence_editor, "strips",
        text="My Strip", icon="FILE_IMAGE",
    )
    row.operator("sequencer.strip_picker", text="", icon="EYEDROPPER").action = "my_select"
    return False
```

**When to return `True`:** only when your UI entirely replaces the
`Input` mode selector (the `txt2img` / `img2img` / `input strips`
dropdown). Returning `True` suppresses that dropdown.

**For video plugins** with `uses_standard_input_strip = False`, the
framework calls `draw_custom_ui()` in the strip-selector area of the
panel (below the prompt). Return value is ignored for video plugins.

---

## Multiple Variants from One Base Class

When a service offers text-to-video, image-to-video, and subject-to-video
variants, share the `load()` logic in a private base class and override
only what differs. Prefix the base class name with `_` so the loader
skips it (it has no `MODEL_ID` anyway, but the underscore makes intent
clear).

```python
class _MyModelBase(ModelPlugin):
    MODEL_TYPE        = "video"
    INPUTS            = InputSpec.PROMPT
    UI_SECTIONS       = [UISection.PROMPT, UISection.SEED]
    PARAMS            = ParamSpec(steps=1)
    REQUIRED_PACKAGES = ["torch", "diffusers"]

    def load(self, prefs, scene, **kw):
        # shared loading logic
        ...
        return {"pipe": pipe}


class MyModelTxt2VidPlugin(_MyModelBase):
    MODEL_ID     = "author/my-model-txt2vid"
    DISPLAY_NAME = "Video: My Model txt2vid"
    DESCRIPTION  = "Text to video"

    def generate(self, pipe_obj, inputs, scene, prefs):
        ...


class MyModelImg2VidPlugin(_MyModelBase):
    MODEL_ID     = "author/my-model-img2vid"
    DISPLAY_NAME = "Video: My Model img2vid"
    DESCRIPTION  = "Image to video"
    INPUTS       = InputSpec.PROMPT | InputSpec.IMAGE

    def generate(self, pipe_obj, inputs, scene, prefs):
        ...
```

Both sub-classes appear in the video dropdown independently.

---

## Complete Examples

### Minimal image plugin (text-to-image)

```python
"""Text-to-image via my-org/my-model."""

import torch
from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram


class MyModelPlugin(ModelPlugin):
    MODEL_ID     = "my-org/my-model"
    DISPLAY_NAME = "Image: My Model"
    MODEL_TYPE   = "image"
    DESCRIPTION  = "Text-to-image via my-org/my-model"

    INPUTS      = InputSpec.PROMPT | InputSpec.NEG_PROMPT
    UI_SECTIONS = [
        UISection.PROMPT, UISection.NEG_PROMPT,
        UISection.RESOLUTION, UISection.STEPS, UISection.GUIDANCE, UISection.SEED,
    ]
    PARAMS      = ParamSpec(steps=20, guidance=7.5)
    REQUIRED_PACKAGES = ["torch", "diffusers"]

    # No inpaint or img2img — this is a pure text-to-image model
    supports_inpaint  = False
    supports_img2img  = False

    def load(self, prefs, scene, **kw):
        from diffusers import StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained(
            self.MODEL_ID, torch_dtype=torch.float16,
        )
        if gfx_device == "mps":
            pipe.to("mps")
        elif low_vram():
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.enable_model_cpu_offload()
        return {"pipe": pipe, "converter": None, "refiner": None, "preprocessor": None}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        pipe = pipe_obj["pipe"]
        seed = inputs.seed
        generator = (
            torch.Generator("cuda").manual_seed(seed)
            if torch.cuda.is_available() and seed != 0 else None
        )
        return pipe(
            prompt=inputs.prompt,
            negative_prompt=inputs.neg_prompt,
            num_inference_steps=inputs.steps,
            guidance_scale=inputs.guidance,
            height=inputs.height,
            width=inputs.width,
            generator=generator,
        ).images[0]
```

---

### Image plugin with img2img and inpaint

```python
"""Text-to-image, img2img, and inpaint via my-org/my-model."""

import torch
from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram


class MyInpaintPlugin(ModelPlugin):
    MODEL_ID     = "my-org/my-inpaint-model"
    DISPLAY_NAME = "Image: My Model (inpaint)"
    MODEL_TYPE   = "image"
    DESCRIPTION  = "Text-to-image with img2img and inpaint support"

    INPUTS      = InputSpec.PROMPT | InputSpec.NEG_PROMPT | InputSpec.IMAGE
    UI_SECTIONS = [
        UISection.PROMPT, UISection.NEG_PROMPT, UISection.IMAGE_STRIP,
        UISection.RESOLUTION, UISection.STEPS, UISection.GUIDANCE,
        UISection.IMAGE_STRENGTH, UISection.SEED,
    ]
    PARAMS      = ParamSpec(steps=30, guidance=7.5, strength=0.75)
    REQUIRED_PACKAGES = ["torch", "diffusers"]

    def load(self, prefs, scene, **kw):
        from diffusers import (
            StableDiffusionPipeline,
            StableDiffusionImg2ImgPipeline,
            StableDiffusionInpaintPipeline,
        )

        mode = kw.get("mode", "txt2img")
        kwargs = dict(pretrained_model_name_or_path=self.MODEL_ID, torch_dtype=torch.float16)

        if mode == "inpaint":
            pipe = StableDiffusionInpaintPipeline.from_pretrained(**kwargs)
        elif mode == "img2img":
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(**kwargs)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(**kwargs)

        if gfx_device == "mps":
            pipe.to("mps")
        elif low_vram():
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(gfx_device)
        return {"pipe": pipe, "converter": pipe, "refiner": None, "preprocessor": None}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        pipe = pipe_obj["pipe"]
        seed = inputs.seed
        generator = (
            torch.Generator("cuda").manual_seed(seed)
            if torch.cuda.is_available() and seed != 0 else None
        )
        common = dict(
            prompt=inputs.prompt,
            negative_prompt=inputs.neg_prompt,
            num_inference_steps=inputs.steps,
            guidance_scale=inputs.guidance,
            generator=generator,
        )

        if inputs.mode == "inpaint" and inputs.image and inputs.inpaint_mask:
            return pipe(
                **common,
                image=inputs.image,
                mask_image=inputs.inpaint_mask,
                height=inputs.height,
                width=inputs.width,
            ).images[0]

        if inputs.mode == "img2img" and inputs.image:
            return pipe(
                **common,
                image=inputs.image,
                strength=1.0 - inputs.strength,
            ).images[0]

        return pipe(
            **common,
            height=inputs.height,
            width=inputs.width,
        ).images[0]
```

---

### Video plugin

```python
"""Text-to-video via my-org/my-video-model."""

import shutil
import torch
from diffusers.utils import export_to_video
from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram, solve_path, clean_filename


class MyVideoPlugin(ModelPlugin):
    MODEL_ID     = "my-org/my-video-model"
    DISPLAY_NAME = "Video: My Model"
    MODEL_TYPE   = "video"
    DESCRIPTION  = "Text-to-video"

    INPUTS      = InputSpec.PROMPT
    UI_SECTIONS = [
        UISection.PROMPT,
        UISection.RESOLUTION, UISection.FRAMES, UISection.STEPS,
        UISection.GUIDANCE, UISection.SEED,
    ]
    PARAMS      = ParamSpec(width=848, height=480, frames=49, steps=25, guidance=6.0)
    REQUIRED_PACKAGES = ["torch", "diffusers"]

    supports_inpaint  = False
    supports_img2img  = False

    def load(self, prefs, scene, **kw):
        from diffusers import CogVideoXPipeline  # replace with your pipeline class

        pipe = CogVideoXPipeline.from_pretrained(self.MODEL_ID, torch_dtype=torch.bfloat16)
        if gfx_device == "mps":
            pipe.to("mps")
        elif low_vram():
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.enable_model_cpu_offload()
        return {"pipe": pipe, "refiner": None, "last_model_card": self.MODEL_ID}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import bpy

        pipe = pipe_obj["pipe"]
        seed = inputs.seed
        generator = (
            torch.Generator("cuda").manual_seed(seed)
            if torch.cuda.is_available() and seed != 0 else None
        )

        output = pipe(
            prompt=inputs.prompt,
            num_inference_steps=inputs.steps,
            guidance_scale=inputs.guidance,
            height=inputs.height,
            width=inputs.width,
            num_frames=inputs.frames,
            generator=generator,
        )

        render = bpy.context.scene.render
        fps = round(render.fps / render.fps_base, 3)
        tmp_path = export_to_video(output.frames[0], fps=fps)
        dst_path = solve_path(clean_filename(str(seed) + "_" + inputs.prompt) + ".mp4")
        shutil.move(tmp_path, dst_path)
        return dst_path
```

---

### Audio / TTS plugin

```python
"""Text-to-speech via my-org/my-tts-model."""

import torch
from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import solve_path, clean_filename


class MyTTSPlugin(ModelPlugin):
    MODEL_ID     = "my-org/my-tts"
    DISPLAY_NAME = "TTS: My TTS Model"
    MODEL_TYPE   = "audio"
    DESCRIPTION  = "Text-to-speech via my-org/my-tts"

    INPUTS      = InputSpec.PROMPT | InputSpec.AUDIO_REF
    UI_SECTIONS = [
        UISection.PROMPT,
        UISection.AUDIO_REF,
        UISection.SEED,
    ]
    PARAMS      = ParamSpec()
    REQUIRED_PACKAGES = ["torch", "torchaudio"]

    def load(self, prefs, scene, **kw):
        # Load and return your model — the return value is cached.
        from my_tts_lib import TTSModel
        model = TTSModel.from_pretrained(self.MODEL_ID)
        return {"model": model}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs) -> str:
        import torchaudio as ta

        model = pipe_obj["model"]
        torch.manual_seed(inputs.seed)

        wav = model.generate(
            text=inputs.prompt,
            speaker_wav=inputs.audio_ref,  # optional; None if no reference
        )

        out_path = solve_path(clean_filename(inputs.prompt[:30]) + ".wav")
        ta.save(out_path, wav, model.sample_rate)
        return out_path
```

---

## Testing and Debugging

**Console output:** every plugin load attempt is logged to the Blender
system console (`Window → Toggle System Console` on Windows). Look for:

```
[Pallaidium] Registered image plugin: my-org/my-model
```

If the file fails to import, the full traceback is printed there.

**Skipping on error:** a broken plugin does not crash the add-on.
The registry skips it and continues loading the rest.

**Re-loading during development:** open the Blender Text Editor,
create a new script, and run:

```python
import importlib, sys
# Remove cached module so discover() re-imports your file
for key in list(sys.modules.keys()):
    if "pallaidium" in key and "my_model_name" in key:
        del sys.modules[key]

from bl_ext.user_default.pallaidium_generative_ai.models import discover
discover()
```

**Checking registration:**

```python
from bl_ext.user_default.pallaidium_generative_ai.models import PLUGIN_REGISTRY
print(list(PLUGIN_REGISTRY.keys()))
```

**Common mistakes:**

| Symptom | Likely cause |
|---|---|
| Plugin doesn't appear in dropdown | `MODEL_ID` is empty, or there is a syntax / import error — check the console |
| `generate()` receives `None` for a field | That `InputSpec` flag was not added to `INPUTS` |
| UI section not shown | That `UISection` value was not added to `UI_SECTIONS` |
| Inpaint / img2img activates unexpectedly | Set `supports_inpaint = False` / `supports_img2img = False` |
| Strip selector shown even though model doesn't need one | Set `uses_standard_input_strip = False` and implement `draw_custom_ui()` |
| `Duplicate MODEL_ID` warning | Two plugins share the same `MODEL_ID` string |
