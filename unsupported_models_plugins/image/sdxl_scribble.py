"""SDXL Scribble ControlNet (xinsir/controlnet-scribble-sdxl-1.0)."""

import random as _random

import cv2
import numpy as np

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram


class SDXLScribblePlugin(ModelPlugin):
    MODEL_ID     = "xinsir/controlnet-scribble-sdxl-1.0"
    DISPLAY_NAME = "Image: SDXL Scribble ControlNet"
    MODEL_TYPE   = "image"
    DESCRIPTION  = "Sketch-guided generation via SDXL Scribble ControlNet"

    INPUTS       = InputSpec.PROMPT | InputSpec.NEG_PROMPT | InputSpec.IMAGE
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.NEG_PROMPT, UISection.IMAGE_STRIP,
        UISection.RESOLUTION, UISection.STEPS, UISection.GUIDANCE, UISection.SEED,
        UISection.SCRIBBLE_TOGGLE, UISection.ENHANCE,
    ]
    PARAMS       = ParamSpec(steps=30, guidance=9.0)
    REQUIRED_PACKAGES          = ["torch", "diffusers", "controlnet_aux", "cv2"]
    supports_inpaint           = False
    supports_img2img           = False
    requires_input_strip       = True
    uses_standard_input_strip  = False

    def load(self, prefs, scene, **kw):
        import torch
        from controlnet_aux import HEDdetector
        from diffusers import (
            ControlNetModel,
            StableDiffusionXLControlNetPipeline,
            EulerAncestralDiscreteScheduler,
            AutoencoderKL,
        )

        print("Loading SDXL Scribble ControlNet…")
        processor = HEDdetector.from_pretrained("lllyasviel/Annotators")
        controlnet = ControlNetModel.from_pretrained(
            self.MODEL_ID, torch_dtype=torch.float16,
            local_files_only=prefs.local_files_only,
        )
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
        )
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet, vae=vae,
            torch_dtype=torch.float16,
            local_files_only=prefs.local_files_only,
        )
        if kw.get("use_lcm"):
            from diffusers import LCMScheduler
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
        else:
            eulera = EulerAncestralDiscreteScheduler.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler"
            )
            pipe.scheduler = eulera

        if gfx_device == "mps":
            pipe.to("mps")
        elif low_vram():
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(gfx_device)
        return {"pipe": pipe, "converter": None, "refiner": None, "preprocessor": processor}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch
        from PIL import Image

        pipe      = pipe_obj["pipe"]
        processor = pipe_obj["preprocessor"]
        image     = inputs.image
        if image is None:
            raise ValueError("SDXL Scribble requires an input image.")

        seed = inputs.seed
        generator = (
            torch.Generator("cuda").manual_seed(seed)
            if torch.cuda.is_available() and seed != 0 else None
        )
        arr = np.array(image)

        use_raw = getattr(scene, "use_scribble_image", False)
        if not use_raw:
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            gray = cv2.bitwise_not(gray)
            gray = cv2.GaussianBlur(gray, (0, 0), 3)
            rand_val = int(round(_random.uniform(0.01, 0.10), 2) * 255)
            gray[gray > rand_val] = 255
            gray[gray < 255] = 0
            processed = Image.fromarray(gray)
            processed = processor(processed, scribble=True)
        else:
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            gray = cv2.bitwise_not(gray)
            processed = processor(gray, scribble=True)

        return pipe(
            prompt=inputs.prompt,
            negative_prompt=inputs.neg_prompt,
            image=processed,
            num_inference_steps=inputs.steps,
            guidance_scale=inputs.guidance,
            controlnet_conditioning_scale=1.0,
            height=inputs.height,
            width=inputs.width,
            generator=generator,
        ).images[0]
