"""Bria Fibo 2.0: text-to-image from structured JSON (or plain) captions, multi-image
editing and masked editing, via diffusers' BriaFibo2Pipeline.

Two checkpoints, one plugin class each:
  - ml-resources/Fibo-2.0-dev          50 steps, CFG 5, negative prompt; text-to-image + editing
  - ml-resources/Fibo-2.0-edit-turbo   4 steps, no CFG; editing only

Ungated mirrors of Bria's briaai/Fibo-2.0-* repos (CC BY-NC 4.0, non-commercial use).

Prompt: a plain description / edit instruction, or a JSON caption (short_description,
objects, text_render, edit_instruction, ...). Reference images ride the shared
klein_strip_N pickers; the active input strip, when there is one, comes first. In an
edit prompt they are addressed as <image_1>, <image_2>, ... (max 4). An inpaint mask
turns the edit into a masked edit (exactly one image). Weights (scene.fibo2_weights):
"bf16" loads the Hub weights, "sdnq8"/"sdnq4" quantize transformer + text encoder on
the fly. scene.fibo2_render_text / scene.fibo2_font_path control the text images the
model reads to spell text_render entries.
"""

import json

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram


_SDNQ_DTYPE = {"sdnq8": "int8", "sdnq4": "uint4"}
_SDNQ_MIN   = (0, 2, 8)
_MAX_IMAGES = 4           # model card limit for multi-image editing
_SIZE_STEP  = 32          # output size granularity (VAE scale x patch, rounded up for safety)
_MAX_REF_PIXELS = 2048 * 2048

# Total VRAM (GiB) to keep the whole pipeline resident, per weight source.
_RESIDENT_GIB = {"sdnq8": 20, "sdnq4": 14, "bf16": 30}
_DOWNLOAD_GB  = {"sdnq8": 26, "sdnq4": 26, "bf16": 26}  # official bf16 weights are fetched in every case


def _weights_of(scene) -> str:
    w = getattr(scene, "fibo2_weights", "sdnq8")
    return w if w in _RESIDENT_GIB else "sdnq8"


def _check_sdnq_version() -> None:
    import re
    try:
        import sdnq
        version = getattr(sdnq, "__version__", "")
        if not version:
            from importlib.metadata import version as _dist_version
            version = _dist_version("sdnq")
    except Exception:
        return
    parts = tuple(int(p) for p in re.findall(r"\d+", version)[:3])
    if parts and parts < _SDNQ_MIN:
        raise RuntimeError(
            f"Fibo 2 SDNQ weights need sdnq {'.'.join(map(str, _SDNQ_MIN))} or newer (found {version}). "
            "Run Install Dependencies in the add-on Preferences, then restart Blender."
        )


def _total_vram_gib() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 2**30
    except Exception:
        pass
    return 0.0


def _in_cache(repo: str, files, cache_dir) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
        return all(isinstance(try_to_load_from_cache(repo, f, cache_dir=cache_dir or None), str)
                   for f in files)
    except Exception:
        return True  # can't tell: don't block or warn


def _repo_fonts(repo: str, hub: dict) -> list:
    """The fonts the model was trained with (Aileron, then DejaVu Sans), from the repo."""
    from huggingface_hub import hf_hub_download
    paths = []
    for name in ("fonts/Aileron-Regular.otf", "fonts/DejaVuSans.ttf"):
        try:
            paths.append(hf_hub_download(repo, name, **hub))
        except Exception as exc:
            print(f"Fibo 2: font {name} unavailable ({exc})")
    return paths


def _json_caption(prompt: str):
    """The prompt parsed as a JSON object caption, or None for plain text."""
    text = (prompt or "").strip()
    if not text.startswith("{"):
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


class _BriaFibo2Base(ModelPlugin):
    MODEL_ID          = ""   # abstract: plugin discovery skips classes without one
    MODEL_TYPE        = "image"
    REQUIRED_PACKAGES = ["torch", "diffusers", "transformers", "sdnq"]
    supports_inpaint  = True     # mask= (masked edit of exactly one image)
    inpaint_uses_strength = False
    uses_strip_power  = False    # images condition the model; no denoise blend
    # Scene props load() reads: a change makes the queue reload the model.
    LOAD_VARIANT_KEYS = ("fibo2_weights",)
    EDIT_ONLY         = False

    # ------------------------------------------------------------------ load

    def load(self, prefs, scene, **kw):
        import torch
        try:
            from diffusers import BriaFibo2Pipeline, BriaFibo2Transformer2DModel
        except ImportError as exc:
            raise RuntimeError(
                "Bria Fibo 2 needs a newer diffusers with BriaFibo2Pipeline. Run Install "
                "Dependencies in the add-on Preferences, then restart Blender."
            ) from exc
        from transformers import Qwen3VLForConditionalGeneration

        weights = _weights_of(scene)
        hub = dict(cache_dir=prefs.hf_cache_dir or None, local_files_only=prefs.local_files_only)
        if prefs.local_files_only and not _in_cache(self.MODEL_ID, ("model_index.json",), prefs.hf_cache_dir):
            raise RuntimeError(
                f"{self.DISPLAY_NAME} is not downloaded yet and 'Use Local Files Only' is on. "
                f"Uncheck it in the add-on Preferences to download (~{_DOWNLOAD_GB[weights]} GB, once)."
            )
        print(f"Loading {self.MODEL_ID} (weights={weights})…")

        on_gpu = gfx_device == "cuda" and not low_vram() and _total_vram_gib() >= _RESIDENT_GIB[weights]
        components = {}
        if weights in _SDNQ_DTYPE:
            _check_sdnq_version()
            from sdnq import SDNQConfig   # importing sdnq registers its quantizer
            q_device = "cuda" if gfx_device == "cuda" else None
            ret = "cuda" if on_gpu else "cpu"
            cfg = dict(weights_dtype=_SDNQ_DTYPE[weights], quantization_device=q_device, return_device=ret)
            components["transformer"] = BriaFibo2Transformer2DModel.from_pretrained(
                self.MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16,
                quantization_config=SDNQConfig(**cfg), **hub,
            )
            # The vision tower reads the edit images: keep it in bf16.
            components["text_encoder"] = Qwen3VLForConditionalGeneration.from_pretrained(
                self.MODEL_ID, subfolder="text_encoder", torch_dtype=torch.bfloat16,
                quantization_config=SDNQConfig(modules_to_not_convert=["visual"], **cfg), **hub,
            )
        pipe = BriaFibo2Pipeline.from_pretrained(
            self.MODEL_ID, torch_dtype=torch.bfloat16, **components, **hub,
        )
        # Edit images go through Qwen3-VL's vision tower: its bf16 Conv3d patch embed
        # is pathologically slow on this torch/cuDNN — run it as a matmul.
        from ...utils.fast_patch_embed import install_fast_patch_embed
        install_fast_patch_embed(pipe.text_encoder, "Fibo 2")

        self._load_loras(pipe, scene, kw.get("enabled_items") or [], weights)

        if gfx_device == "mps":
            pipe.to("mps")
            placement = "on Apple GPU (MPS)"
        elif on_gpu:
            pipe.to("cuda")
            placement = "resident on GPU"
        elif gfx_device == "cuda":
            # Whole-component moves only: group offload can strand SDNQ scale buffers.
            pipe.enable_model_cpu_offload()
            placement = f"component CPU offload ({_total_vram_gib():.0f} GB VRAM)"
        else:
            placement = "on CPU (no CUDA GPU found) — expect very slow generation"
        print(f"Fibo 2: {placement}")
        return {"pipe": pipe, "converter": None, "refiner": None, "preprocessor": None}

    @staticmethod
    def _load_loras(pipe, scene, enabled_items, weights):
        if not enabled_items:
            return
        from ...utils.helpers import clean_filename, bpy
        folder = bpy.path.abspath(getattr(scene, "lora_folder", "") or "")
        names, strengths = [], []
        try:
            for item in enabled_items:
                name = clean_filename(item.name).replace(".", "")
                pipe.load_lora_weights(folder, weight_name=item.name + ".safetensors", adapter_name=name)
                names.append(name)
                strengths.append(item.weight_value)
                print(f"Fibo 2: LoRA '{item.name}' loaded (adapter='{name}', weight={item.weight_value:.2f})")
            pipe.set_adapters(names, adapter_weights=strengths)
        except Exception as exc:
            hint = "" if weights == "bf16" else " LoRAs may not attach to quantized weights: set Weights to BF16."
            raise RuntimeError(f"Fibo 2: LoRA loading failed ({exc}).{hint}") from exc

    # ------------------------------------------------------------------ UI

    def draw_custom_ui(self, col, context) -> bool:
        scene = context.scene
        # Strip refs live in the scene shown in the VSE (context.sequencer_scene
        # in Blender 5.x), which can differ from the active scene.
        vse_scene = getattr(context, "sequencer_scene", None) or scene
        try:
            col.prop(scene, "input_strips", text="Input")
        except Exception:
            pass
        if vse_scene.sequence_editor is None:
            return True
        for i in range(1, scene.klein_visible_strips + 1):
            row = col.row(align=True)
            row.prop_search(
                vse_scene, f"klein_strip_{i}", vse_scene.sequence_editor, "strips",
                text="Ref.", icon="FILE_IMAGE",
            )
            row.operator("sequencer.strip_picker", text="", icon="EYEDROPPER").action = f"klein_select{i}"
            if i == scene.klein_visible_strips and scene.klein_visible_strips < _MAX_IMAGES:
                if scene.klein_visible_strips > 3:
                    row.operator("object.klein_hide_strip", text="", icon="REMOVE").strip_index = i
                row.operator("object.klein_add_strip", text="", icon="ADD")
            elif i == scene.klein_visible_strips and scene.klein_visible_strips > 3:
                row.operator("object.klein_hide_strip", text="", icon="REMOVE").strip_index = i
        return True

    def draw_model_options_ui(self, col, context):
        """Weights and text-image options, between the model selector and Batch Count."""
        scene = context.scene
        sub = col.column(align=True)
        sub.prop(scene, "fibo2_weights", text="Weights")
        sub.prop(scene, "fibo2_render_text", text="Render Text Images")
        font = sub.column(align=True)
        font.active = scene.fibo2_render_text
        font.prop(scene, "fibo2_font_path", text="Extra Font")
        if self.EDIT_ONLY:
            sub.label(text="Editing only: add an input or Ref. image", icon="INFO")
        sub.label(text="Non-commercial use only (CC BY-NC 4.0)", icon="INFO")

    # ------------------------------------------------------------------ generate

    def _images(self, inputs: ModelInputs, scene) -> list:
        from PIL import Image as _PILImage
        images = [inputs.image.convert("RGB")] if inputs.image is not None else []
        for n in range(1, 10):
            path = getattr(scene, f"klein_strip_{n}_path", "")
            if not path:
                continue
            try:
                images.append(_PILImage.open(path).convert("RGB"))
            except Exception as exc:
                print(f"Fibo 2: reference '{path}' could not be opened: {exc}")
        if len(images) > _MAX_IMAGES:
            print(f"Fibo 2: {len(images)} images, keeping the first {_MAX_IMAGES}")
            images = images[:_MAX_IMAGES]
        for i, img in enumerate(images):
            pixels = img.width * img.height
            if pixels > _MAX_REF_PIXELS:
                scale = (_MAX_REF_PIXELS / pixels) ** 0.5
                images[i] = img.resize((max(_SIZE_STEP, round(img.width * scale)),
                                        max(_SIZE_STEP, round(img.height * scale))),
                                       _PILImage.Resampling.LANCZOS)
        return images

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch
        from PIL import Image as _PILImage

        width = max(_SIZE_STEP, int(inputs.width) // _SIZE_STEP * _SIZE_STEP)
        height = max(_SIZE_STEP, int(inputs.height) // _SIZE_STEP * _SIZE_STEP)
        prompt = inputs.prompt or ""
        caption = _json_caption(prompt)
        if caption is not None:
            prompt = json.dumps(caption, ensure_ascii=False, separators=(",", ":"))
        elif prompt.lstrip().startswith("{"):
            print("Fibo 2: the prompt looks like JSON but does not parse — sending it as plain text")

        kwargs = dict(
            prompt=prompt, height=height, width=width,
            num_inference_steps=inputs.steps, guidance_scale=inputs.guidance,
            render_text_images=bool(getattr(scene, "fibo2_render_text", True)),
            callback_on_step_end=self.step_callback(inputs),
        )
        if inputs.seed != 0:
            kwargs["generator"] = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(inputs.seed)
        if inputs.neg_prompt and inputs.guidance > 1.0:
            kwargs["negative_prompt"] = inputs.neg_prompt
        if kwargs["render_text_images"]:
            # The user's font first, then the fonts the model was trained with.
            fonts = []
            font = (getattr(scene, "fibo2_font_path", "") or "").strip()
            if font:
                from ...utils.helpers import bpy
                fonts.append(bpy.path.abspath(font))
            fonts += _repo_fonts(self.MODEL_ID, dict(cache_dir=prefs.hf_cache_dir or None,
                                                     local_files_only=prefs.local_files_only))
            if fonts:
                kwargs["fonts"] = fonts

        images = self._images(inputs, scene)
        if caption is not None and "edit_instruction" in caption and not images:
            raise RuntimeError("Fibo 2: an edit caption needs an input strip or a Ref. image.")
        if self.EDIT_ONLY and not images:
            raise RuntimeError("Fibo 2 Edit Turbo is editing only: add an input strip or a Ref. image "
                               "(use Bria Fibo 2.0 for text-to-image).")

        self.set_phase(inputs, "Generating")
        if inputs.mode == "inpaint":
            if inputs.inpaint_mask is None or inputs.image is None:
                raise RuntimeError("Inpaint mask not available. Check inpaint_selected_strip.")
            if len(images) > 1:
                print(f"Fibo 2: masked edit takes exactly one image; ignoring {len(images) - 1} reference(s)")
            kwargs["image"] = inputs.image.convert("RGB").resize((width, height), _PILImage.Resampling.LANCZOS)
            kwargs["mask"] = inputs.inpaint_mask.convert("L").resize((width, height), _PILImage.Resampling.NEAREST)
        elif images:
            kwargs["image"] = images if len(images) > 1 else images[0]
        print(f"Fibo 2: {len(images)} image(s), mode={inputs.mode}, {width}x{height}, "
              f"{inputs.steps} steps, guidance {inputs.guidance}")

        result = pipe_obj["pipe"](**kwargs).images[0]
        if result.size != (inputs.width, inputs.height):
            result = result.resize((inputs.width, inputs.height), _PILImage.Resampling.LANCZOS)
        return result.convert("RGB")


class BriaFibo2DevPlugin(_BriaFibo2Base):
    MODEL_ID     = "ml-resources/Fibo-2.0-dev"
    DISPLAY_NAME = "Bria Fibo 2.0"
    DESCRIPTION  = (
        "Bria Fibo 2.0 (8B): text-to-image from JSON or plain captions, multi-image editing "
        "(<image_1>…<image_4>), masked editing and text rendering. Negative prompt + guidance"
    )
    INPUTS       = (InputSpec.PROMPT | InputSpec.NEG_PROMPT | InputSpec.IMAGE | InputSpec.LORA)
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.NEG_PROMPT, UISection.IMAGE_STRIP,
        UISection.RESOLUTION, UISection.FRAMES, UISection.STEPS, UISection.GUIDANCE,
        UISection.SEED, UISection.LORA,
    ]
    PARAMS       = ParamSpec(steps=50, guidance=5.0)


class BriaFibo2EditTurboPlugin(_BriaFibo2Base):
    MODEL_ID     = "ml-resources/Fibo-2.0-edit-turbo"
    DISPLAY_NAME = "Bria Fibo 2.0 Edit Turbo"
    DESCRIPTION  = (
        "Bria Fibo 2.0 Edit Turbo: 4-step distilled image editing (editing only), up to 4 images "
        "(<image_1>…<image_4>), masked editing. No guidance"
    )
    INPUTS       = InputSpec.PROMPT | InputSpec.IMAGE | InputSpec.LORA
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.IMAGE_STRIP,
        UISection.RESOLUTION, UISection.FRAMES, UISection.STEPS,
        UISection.SEED, UISection.LORA,
    ]
    PARAMS       = ParamSpec(steps=4, guidance=1.0)
    EDIT_ONLY    = True
