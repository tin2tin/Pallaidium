"""Text-to-image via Segmind Vega (segmind/Segmind-Vega)."""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram


class SegmindVegaPlugin(ModelPlugin):
    MODEL_ID     = "segmind/Segmind-Vega"
    DISPLAY_NAME = "Image: Segmind Vega (fast)"
    MODEL_TYPE   = "image"
    DESCRIPTION  = "Fast text-to-image via Segmind Vega + VegaRT LCM"

    INPUTS       = InputSpec.PROMPT | InputSpec.NEG_PROMPT
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.NEG_PROMPT,
        UISection.RESOLUTION, UISection.STEPS, UISection.GUIDANCE, UISection.SEED,
        UISection.ENHANCE,
    ]
    PARAMS       = ParamSpec(steps=4, guidance=0.0)
    REQUIRED_PACKAGES = ["torch", "diffusers"]

    def load(self, prefs, scene, **kw):
        import torch
        from diffusers import AutoPipelineForText2Image

        print("Loading Segmind Vega…")
        pipe = AutoPipelineForText2Image.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.float16,
            variant="fp16",
            local_files_only=prefs.local_files_only,
        )
        pipe.load_lora_weights("segmind/Segmind-VegaRT")
        if gfx_device == "mps":
            pipe.to("mps")
        elif low_vram():
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(gfx_device)
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
        return pipe(
            prompt=inputs.prompt,
            negative_prompt=inputs.neg_prompt,
            num_inference_steps=inputs.steps,
            guidance_scale=0.0,
            height=inputs.height,
            width=inputs.width,
            generator=generator,
        ).images[0]
