"""SDXL OpenPose ControlNet (xinsir/controlnet-openpose-sdxl-1.0)."""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram


class SDXLOpenPosePlugin(ModelPlugin):
    MODEL_ID     = "xinsir/controlnet-openpose-sdxl-1.0"
    DISPLAY_NAME = "Image: SDXL OpenPose ControlNet"
    MODEL_TYPE   = "image"
    DESCRIPTION  = "Pose-guided generation via SDXL OpenPose ControlNet"

    INPUTS       = InputSpec.PROMPT | InputSpec.NEG_PROMPT | InputSpec.IMAGE
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.NEG_PROMPT, UISection.IMAGE_STRIP,
        UISection.RESOLUTION, UISection.STEPS, UISection.SEED,
        UISection.POSE_TOGGLE, UISection.ENHANCE,
    ]
    PARAMS       = ParamSpec(steps=30)
    REQUIRED_PACKAGES          = ["torch", "diffusers", "controlnet_aux"]
    supports_inpaint           = False
    supports_img2img           = False
    requires_input_strip       = True
    uses_standard_input_strip  = False

    def load(self, prefs, scene, **kw):
        import torch
        from diffusers import (
            ControlNetModel,
            StableDiffusionXLControlNetPipeline,
            AutoencoderKL,
            EulerAncestralDiscreteScheduler,
        )
        from controlnet_aux import OpenposeDetector

        print("Loading SDXL OpenPose ControlNet…")
        eulera_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler"
        )
        controlnet = ControlNetModel.from_pretrained(
            self.MODEL_ID, torch_dtype=torch.float16
        )
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
        )
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet, vae=vae,
            torch_dtype=torch.float16,
            scheduler=eulera_scheduler,
        )
        processor = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

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
        return {"pipe": pipe, "converter": None, "refiner": None, "preprocessor": processor}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch
        import numpy as np

        pipe      = pipe_obj["pipe"]
        processor = pipe_obj["preprocessor"]
        image     = inputs.image
        if image is None:
            raise ValueError("SDXL OpenPose requires an input image.")

        seed = inputs.seed
        generator = (
            torch.Generator("cuda").manual_seed(seed)
            if torch.cuda.is_available() and seed != 0 else None
        )
        image = image.resize((inputs.width, inputs.height))

        if not getattr(scene, "openpose_use_bones", False):
            image = processor(np.array(image), hand_and_face=True)

        return pipe(
            prompt=inputs.prompt,
            negative_prompt=inputs.neg_prompt,
            image=image,
            controlnet_conditioning_scale=1.0,
            num_inference_steps=inputs.steps,
            generator=generator,
        ).images[0]
