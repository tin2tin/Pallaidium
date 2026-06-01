"""Text-to-video and img2vid via LTX-Video (NF4 quantized, LTXConditionPipeline)."""

import shutil
from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram, solve_path, clean_filename


class LTXVideoPlugin(ModelPlugin):
    MODEL_ID     = "Lightricks/LTX-Video"
    DISPLAY_NAME = "Video: LTX-Video"
    MODEL_TYPE   = "video"
    DESCRIPTION  = "Text-to-video and img2vid via LTX-Video (NF4 quantized)"

    INPUTS       = InputSpec.PROMPT | InputSpec.NEG_PROMPT | InputSpec.IMAGE
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.NEG_PROMPT, UISection.VIDEO_STRIP,
        UISection.RESOLUTION, UISection.FRAMES, UISection.STEPS, UISection.GUIDANCE, UISection.SEED,
    ]
    PARAMS       = ParamSpec(width=768, height=512, frames=97, steps=40, guidance=3.5)
    REQUIRED_PACKAGES = ["torch", "diffusers"]

    def load(self, prefs, scene, **kw):
        import torch
        from diffusers import (
            LTXConditionPipeline, LTXVideoTransformer3DModel, BitsAndBytesConfig,
        )

        print(f"Loading {self.MODEL_ID}…")

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        transformer = LTXVideoTransformer3DModel.from_pretrained(
            "a-r-r-o-w/LTX-Video-0.9.7-diffusers",
            quantization_config=nf4_config,
            torch_dtype=torch.bfloat16,
            subfolder="transformer",
        )
        pipe = LTXConditionPipeline.from_pretrained(
            "a-r-r-o-w/LTX-Video-0.9.7-diffusers",
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )

        if gfx_device == "mps":
            pipe.to("mps")
        else:
            pipe.vae.enable_tiling()
            pipe.enable_model_cpu_offload()

        return {"pipe": pipe, "refiner": None, "last_model_card": self.MODEL_ID}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch
        from diffusers.utils import export_to_video

        pipe = pipe_obj["pipe"]
        seed = inputs.seed
        generator = (
            torch.Generator("cuda").manual_seed(seed)
            if torch.cuda.is_available() and seed != 0 else None
        )

        if inputs.image is not None:
            image_input = inputs.image
        elif inputs.video_path:
            from diffusers.utils import load_video
            image_input = load_video(inputs.video_path)
        else:
            image_input = None

        if image_input is not None:
            video_frames = pipe(
                image=image_input,
                prompt=inputs.prompt,
                negative_prompt=inputs.neg_prompt,
                num_inference_steps=inputs.steps,
                guidance_scale=inputs.guidance,
                height=inputs.height,
                width=inputs.width,
                num_frames=inputs.frames,
                generator=generator,
                max_sequence_length=512,
                decode_timestep=0.05,
                image_cond_noise_scale=0.025,
            ).frames[0]
        else:
            video_frames = pipe(
                prompt=inputs.prompt,
                negative_prompt=inputs.neg_prompt,
                num_inference_steps=inputs.steps,
                guidance_scale=inputs.guidance,
                height=inputs.height,
                width=inputs.width,
                num_frames=inputs.frames,
                generator=generator,
                max_sequence_length=512,
                decode_timestep=0.05,
                image_cond_noise_scale=0.025,
            ).frames[0]

        import bpy
        render = bpy.context.scene.render
        fps = round(render.fps / render.fps_base, 3)
        src_path = export_to_video(video_frames, fps=fps)
        dst_path = solve_path(clean_filename(str(seed) + "_" + inputs.prompt) + ".mp4")
        shutil.move(src_path, dst_path)
        return dst_path
