"""Text-to-image and img2img via Z-Image (Tongyi-MAI/Z-Image and Z-Image-Turbo)."""

import os as _os
try:
    import huggingface_hub as _hfhub
    if not hasattr(_hfhub, "is_offline_mode"):
        def _is_offline_mode():
            return _os.environ.get("HF_HUB_OFFLINE", "0") == "1"
        _hfhub.is_offline_mode = _is_offline_mode
except Exception:
    pass

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram


class _ZImageBase(ModelPlugin):
    MODEL_TYPE = "image"
    REQUIRED_PACKAGES = ["torch", "diffusers"]
    supports_inpaint  = False

    INPUTS           = InputSpec.PROMPT | InputSpec.NEG_PROMPT | InputSpec.IMAGE
    UI_SECTIONS = [
        UISection.PROMPT, UISection.NEG_PROMPT, UISection.IMAGE_STRIP,
        UISection.RESOLUTION, UISection.FRAMES, UISection.STEPS, UISection.GUIDANCE,
        UISection.IMAGE_STRENGTH, UISection.SEED,
    ]

    _TURBO_TRANSFORMER = "linoyts/beyond-reality-z-image-diffusers"

    def _build_pipe(self, model_id, prefs, turbo=False):
        import torch
        try:
            from diffusers import ZImagePipeline
        except ImportError as _e:
            if "huggingface-hub" in str(_e):
                raise RuntimeError(
                    "Z-Image requires huggingface-hub>=1.5.0 but an older version is installed. "
                    "Upgrade with: pip install \"huggingface_hub>=1.5.0\" -U"
                ) from _e
            raise

        _cache_dir = prefs.hf_cache_dir or None
        if turbo:
            from diffusers import ZImageTransformer2DModel
            transformer = ZImageTransformer2DModel.from_pretrained(
                self._TURBO_TRANSFORMER, torch_dtype=torch.bfloat16, cache_dir=_cache_dir,
            )
            pipe = ZImagePipeline.from_pretrained(
                model_id, transformer=transformer, torch_dtype=torch.bfloat16, cache_dir=_cache_dir,
            )
        else:
            pipe = ZImagePipeline.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False, cache_dir=_cache_dir,
            )

        if gfx_device == "mps":
            pipe.to("mps")
        elif low_vram():
            pipe.enable_sequential_cpu_offload() if turbo else pipe.enable_model_cpu_offload()
            if not turbo:
                pipe.vae.enable_tiling()
        else:
            pipe.to("cuda")
        return pipe

    def _build_img2img(self, model_id, prefs, turbo=False):
        import torch
        try:
            from diffusers import ZImageImg2ImgPipeline
        except ImportError as _e:
            if "huggingface-hub" in str(_e):
                raise RuntimeError(
                    "Z-Image requires huggingface-hub>=1.5.0 but an older version is installed. "
                    "Upgrade with: pip install \"huggingface_hub>=1.5.0\" -U"
                ) from _e
            raise

        _cache_dir = prefs.hf_cache_dir or None
        if turbo:
            from diffusers import ZImageTransformer2DModel
            transformer = ZImageTransformer2DModel.from_pretrained(
                self._TURBO_TRANSFORMER, torch_dtype=torch.bfloat16, cache_dir=_cache_dir,
            )
            conv = ZImageImg2ImgPipeline.from_pretrained(
                model_id, transformer=transformer, torch_dtype=torch.bfloat16, cache_dir=_cache_dir,
            )
        else:
            conv = ZImageImg2ImgPipeline.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False, cache_dir=_cache_dir,
            )

        if gfx_device == "mps":
            conv.to("mps")
        elif low_vram():
            conv.enable_sequential_cpu_offload() if turbo else conv.enable_model_cpu_offload()
            if not turbo:
                conv.vae.enable_tiling()
        else:
            conv.to("cuda")
        return conv

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch

        seed = inputs.seed
        generator = (
            torch.Generator("cuda").manual_seed(seed)
            if torch.cuda.is_available() and seed != 0 else None
        )
        self.set_phase(inputs, "Generating")
        cb = self.step_callback(inputs)
        if inputs.mode == "img2img" and inputs.image is not None:
            conv = pipe_obj["converter"]
            return conv(
                prompt=inputs.prompt,
                negative_prompt=inputs.neg_prompt,
                image=inputs.image,
                strength=1.0 - inputs.strength,
                num_inference_steps=inputs.steps,
                guidance_scale=inputs.guidance,
                generator=generator,
                callback_on_step_end=cb,
            ).images[0]
        else:
            pipe = pipe_obj["pipe"]
            return pipe(
                prompt=inputs.prompt,
                negative_prompt=inputs.neg_prompt,
                num_inference_steps=inputs.steps,
                guidance_scale=inputs.guidance,
                height=inputs.height,
                width=inputs.width,
                generator=generator,
                callback_on_step_end=cb,
            ).images[0]


class ZImagePlugin(_ZImageBase):
    MODEL_ID     = "Tongyi-MAI/Z-Image"
    DISPLAY_NAME = "Image: Z-Image"
    DESCRIPTION  = "Text-to-image and img2img via Z-Image"
    PARAMS       = ParamSpec(steps=30, guidance=7.0)

    def load(self, prefs, scene, **kw):
        mode = kw.get("mode", "txt2img")
        if mode == "img2img":
            conv = self._build_img2img(self.MODEL_ID, prefs, turbo=False)
            return {"pipe": None, "converter": conv, "refiner": None, "preprocessor": None}
        pipe = self._build_pipe(self.MODEL_ID, prefs, turbo=False)
        return {"pipe": pipe, "converter": None, "refiner": None, "preprocessor": None}


class ZImageTurboPlugin(_ZImageBase):
    MODEL_ID     = "Tongyi-MAI/Z-Image-Turbo"
    DISPLAY_NAME = "Image: Z-Image Turbo (fast)"
    DESCRIPTION  = "Fast text-to-image and img2img via Z-Image Turbo"
    PARAMS       = ParamSpec(steps=8, guidance=0.0)

    def load(self, prefs, scene, **kw):
        mode = kw.get("mode", "txt2img")
        if mode == "img2img":
            conv = self._build_img2img(self.MODEL_ID, prefs, turbo=True)
            return {"pipe": None, "converter": conv, "refiner": None, "preprocessor": None}
        pipe = self._build_pipe(self.MODEL_ID, prefs, turbo=True)
        return {"pipe": pipe, "converter": None, "refiner": None, "preprocessor": None}
