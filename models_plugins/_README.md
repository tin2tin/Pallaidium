# Pallaidium Model Plugins

Drop a `.py` file into the right subfolder and restart Blender.
The model appears in the corresponding dropdown automatically.

```
models_plugins/
  video/   → models that generate MP4 video
  image/   → models that generate PNG/JPG images
  audio/   → models that generate WAV audio
  text/    → models that generate text strips
```

Files and directories whose names begin with `_` are ignored.

---

## Quickstart

1. Copy `_template.py` into the correct subfolder.
2. Rename it (`my_model.py` — lowercase, underscores).
3. Set `MODEL_ID`, `DISPLAY_NAME`, `DESCRIPTION`.
4. Declare `INPUTS` (what data to collect) and `UI_SECTIONS` (what to show in the panel).
5. Implement `load()` and `generate()`.
6. Restart Blender — your model is in the dropdown.

---

## InputSpec flags

Combine with `|` to declare what data your model needs:

| Flag | Data collected |
|---|---|
| `InputSpec.PROMPT` | Text prompt |
| `InputSpec.NEG_PROMPT` | Negative prompt |
| `InputSpec.IMAGE` | Single image from selected strip |
| `InputSpec.MULTI_IMAGE` | Up to `PARAMS.max_multi_images` dynamic image strips |
| `InputSpec.TRIPLE_IMAGE` | Exactly 3 fixed image pickers |
| `InputSpec.AUDIO_REF` | Speaker reference audio (optional) |
| `InputSpec.AUDIO_REF_REQ` | Speaker reference audio (required) |
| `InputSpec.TEXT_REF` | Reference transcription text |
| `InputSpec.VIDEO` | Video strip input |
| `InputSpec.FACE_FOLDER` | IP Adapter face image folder |
| `InputSpec.STYLE_FOLDER` | IP Adapter style image folder |
| `InputSpec.LORA` | LoRA files with per-file weights |
| `InputSpec.API_KEY` | External API key |

---

## UISection values

List in the order you want them rendered:

| Section | Renders |
|---|---|
| `PROMPT` | Text prompt textarea |
| `NEG_PROMPT` | Negative prompt textarea |
| `IMAGE_STRIP` | Single image eyedropper |
| `MULTI_IMAGES` | Dynamic add/remove image strips |
| `TRIPLE_IMAGE` | Three fixed image pickers |
| `TRIPLE_PROMPT_IMG` | Three (prompt + image) pairs |
| `AUDIO_REF` | Speaker reference file picker |
| `TEXT_REF` | Reference text input |
| `VIDEO_STRIP` | Video strip eyedropper |
| `RESOLUTION` | Width × height dropdowns |
| `FRAMES` | Frame count slider |
| `STEPS` | Inference steps slider |
| `GUIDANCE` | Guidance / word-power slider |
| `IMAGE_STRENGTH` | img2img / inpaint strength |
| `SEED` | Seed + randomise toggle |
| `LORA` | LoRA folder + file list with enable/weight |
| `IP_ADAPTER` | Face + style folder pickers |
| `AUDIO_DURATION` | Duration slider |
| `SPEED` | Audio speed slider |
| `CHAT_PARAMS` | Exaggeration + pace + temperature |
| `ILLUMINATION` | Lighting style + direction dropdowns |
| `POSE_TOGGLE` | "Read as OpenPose Rig" checkbox |
| `SCRIBBLE_TOGGLE` | "Read as Scribble" checkbox |
| `ENHANCE` | Quality / Speed / Faces / Upscale toggles |

For anything not listed, implement `draw_custom_ui(self, layout, context)`.

---

## ParamSpec defaults

Override only what differs:

```python
PARAMS = ParamSpec(
    width=832, height=480, frames=81,
    steps=30, guidance=5.0,
)
```

| Field | Default | Meaning |
|---|---|---|
| `width` | 1024 | Default output width |
| `height` | 576 | Default output height |
| `frames` | 49 | Default frame count |
| `steps` | 25 | Default inference steps |
| `guidance` | 7.5 | Default guidance scale |
| `strength` | 0.8 | Default img2img strength |
| `audio_length` | 5.0 | Default audio duration (seconds) |
| `max_multi_images` | 1 | Max dynamic image strips |
| `audio_ref_required` | False | Show "required" label on audio ref picker |

---

## Example: minimal text-to-image plugin

```python
from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs

class MyFluxPlugin(ModelPlugin):
    MODEL_ID     = "myuser/my-flux-model"
    DISPLAY_NAME = "My Flux Model (1024×1024)"
    MODEL_TYPE   = "image"
    DESCRIPTION  = "Custom Flux-based text-to-image model"

    INPUTS      = InputSpec.PROMPT | InputSpec.NEG_PROMPT | InputSpec.LORA
    UI_SECTIONS = [UISection.PROMPT, UISection.NEG_PROMPT, UISection.LORA,
                   UISection.STEPS, UISection.GUIDANCE, UISection.SEED]
    PARAMS      = ParamSpec(width=1024, height=1024, steps=20, guidance=3.5)
    REQUIRED_PACKAGES = ["diffusers", "torch"]

    def load(self, prefs, scene, **kw):
        import torch
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(self.MODEL_ID, torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()
        return pipe

    def generate(self, pipe, inputs: ModelInputs, scene, prefs) -> str:
        import torch
        result = pipe(
            prompt=inputs.prompt,
            num_inference_steps=inputs.steps,
            guidance_scale=inputs.guidance,
            generator=torch.Generator().manual_seed(inputs.seed),
        )
        return save_image(result.images[0], prefs.generator_ai)
```
