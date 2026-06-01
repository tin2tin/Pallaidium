"""SDXL Canny ControlNet (diffusers/controlnet-canny-sdxl-1.0-small)."""

import cv2
import numpy as np

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram


class SDXLCannyPlugin(ModelPlugin):
    MODEL_ID     = "diffusers/controlnet-canny-sdxl-1.0-small"
    DISPLAY_NAME = "Image: SDXL Canny ControlNet"
    MODEL_TYPE   = "image"
    DESCRIPTION  = "Edge-guided image generation via SDXL Canny ControlNet"

    INPUTS       = InputSpec.PROMPT | InputSpec.IMAGE
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.IMAGE_STRIP,
        UISection.RESOLUTION, UISection.STEPS, UISection.IMAGE_STRENGTH, UISection.SEED,
        UISection.ENHANCE,
    ]
    PARAMS       = ParamSpec(steps=30)
    REQUIRED_PACKAGES      = ["torch", "diffusers", "cv2"]
    supports_inpaint       = False
    supports_img2img       = False
    requires_input_strip   = True

    def load(self, prefs, scene, **kw):
        import torch
        from diffusers import (
            ControlNetModel,
            StableDiffusionXLControlNetPipeline,
            AutoencoderKL,
        )

        print("Loading SDXL Canny ControlNet…")
        controlnet = ControlNetModel.from_pretrained(
            self.MODEL_ID, torch_dtype=torch.float16, variant="fp16",
            local_files_only=prefs.local_files_only,
        )
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
        )
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet, vae=vae,
            torch_dtype=torch.float16, variant="fp16",
        )
        from ...utils.helpers import NoWatermark
        pipe.watermark = NoWatermark()

        if kw.get("use_lcm"):
            from diffusers import LCMScheduler
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")

        if gfx_device == "mps":
            pipe.to("mps")
        elif low_vram():
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(gfx_device)
        return {"pipe": pipe, "converter": None, "refiner": None, "preprocessor": None}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch

        pipe  = pipe_obj["pipe"]
        image = inputs.image
        if image is None:
            raise ValueError("SDXL Canny requires an input image.")

        seed = inputs.seed
        generator = (
            torch.Generator("cuda").manual_seed(seed)
            if torch.cuda.is_available() and seed != 0 else None
        )

        arr = np.array(image)
        edges = cv2.Canny(arr, 100, 200)
        canny = np.concatenate([edges[:, :, None]] * 3, axis=2)
        from PIL import Image
        canny_image = Image.fromarray(canny)

        return pipe(
            prompt=inputs.prompt,
            num_inference_steps=inputs.steps,
            controlnet_conditioning_scale=1.0 - inputs.strength,
            image=canny_image,
            generator=generator,
        ).images[0]
