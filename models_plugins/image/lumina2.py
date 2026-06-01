"""Text-to-image via Lumina-Image-2.0 (Alpha-VLLM/Lumina-Image-2.0)."""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram


class Lumina2Plugin(ModelPlugin):
    MODEL_ID     = "Alpha-VLLM/Lumina-Image-2.0"
    DISPLAY_NAME = "Image: Lumina-Image 2.0"
    MODEL_TYPE   = "image"
    DESCRIPTION  = "High-quality text-to-image via Lumina-Image 2.0"

    INPUTS       = InputSpec.PROMPT | InputSpec.NEG_PROMPT
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.NEG_PROMPT,
        UISection.RESOLUTION, UISection.FRAMES, UISection.STEPS, UISection.GUIDANCE, UISection.SEED,
    ]
    PARAMS            = ParamSpec(steps=30, guidance=4.0)
    REQUIRED_PACKAGES = ["torch", "diffusers"]
    supports_inpaint  = False
    supports_img2img  = False

    def load(self, prefs, scene, **kw):
        import torch
        from diffusers import Lumina2Pipeline

        _cache_dir = prefs.hf_cache_dir or None
        print("Loading Lumina-Image-2.0…")
        pipe = Lumina2Pipeline.from_pretrained(self.MODEL_ID, torch_dtype=torch.bfloat16, cache_dir=_cache_dir)
        if gfx_device == "mps":
            pipe.to("mps")
        elif low_vram():
            pipe.enable_model_cpu_offload()
            pipe.vae.enable_tiling()
        else:
            pipe.enable_model_cpu_offload()
        return {"pipe": pipe, "converter": None, "refiner": None, "preprocessor": None}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch

        pipe = pipe_obj["pipe"]
        seed = inputs.seed
        generator = (
            torch.Generator("cuda").manual_seed(seed)
            if torch.cuda.is_available() and seed != 0
            else (torch.Generator(device=gfx_device).manual_seed(seed) if seed != 0 else None)
        )
        self.set_phase(inputs, "Generating")
        return pipe(
            prompt=inputs.prompt,
            negative_prompt=inputs.neg_prompt,
            num_inference_steps=inputs.steps,
            guidance_scale=inputs.guidance,
            height=inputs.height,
            width=inputs.width,
            cfg_trunc_ratio=0.25,
            cfg_normalization=True,
            generator=generator,
            callback_on_step_end=self.step_callback(inputs),
        ).images[0]
