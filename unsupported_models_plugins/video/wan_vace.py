"""Text-to-video and img2vid via Wan2.1-VACE-1.3B (4-bit quantized VAE + WanVACEPipeline)."""

import shutil
from PIL import Image
from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram, solve_path, clean_filename, load_first_frame


class WanVACEPlugin(ModelPlugin):
    MODEL_ID     = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
    DISPLAY_NAME = "Video: Wan2.1 VACE 1.3B"
    MODEL_TYPE   = "video"
    DESCRIPTION  = "Text-to-video and img2vid via Wan2.1-VACE-1.3B (4-bit NF4 quantized)"

    INPUTS       = InputSpec.PROMPT | InputSpec.NEG_PROMPT | InputSpec.IMAGE
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.NEG_PROMPT, UISection.VIDEO_STRIP,
        UISection.RESOLUTION, UISection.FRAMES, UISection.STEPS, UISection.GUIDANCE, UISection.SEED,
    ]
    PARAMS       = ParamSpec(width=832, height=480, frames=81, steps=40, guidance=5.0)
    REQUIRED_PACKAGES = ["torch", "diffusers"]

    def load(self, prefs, scene, **kw):
        import torch
        from diffusers import AutoencoderKLWan, WanVACEPipeline
        from diffusers.quantizers import PipelineQuantizationConfig
        from diffusers.schedulers import UniPCMultistepScheduler

        print(f"Loading {self.MODEL_ID}…")

        quant_config = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
            components_to_quantize=["transformer"],
        )
        vae = AutoencoderKLWan.from_pretrained(
            self.MODEL_ID, subfolder="vae", torch_dtype=torch.float32,
        )
        pipe = WanVACEPipeline.from_pretrained(
            self.MODEL_ID, vae=vae, quantization_config=quant_config,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)

        if gfx_device == "mps":
            pipe.to("mps")
        elif low_vram():
            pipe.enable_model_cpu_offload()
        else:
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

        image = inputs.image
        if image is None and inputs.video_path:
            image = load_first_frame(inputs.video_path)

        w, h = inputs.width, inputs.height

        if image is not None:
            img = image.resize((w, h))
            frames = [img]
            frames.extend([Image.new("RGB", (w, h), (128, 128, 128))] * (inputs.frames - 1))
            mask_black = Image.new("L", (w, h), 0)
            mask_white = Image.new("L", (w, h), 255)
            mask = [mask_black] + [mask_white] * (inputs.frames - 1)

            video_frames = pipe(
                video=frames,
                mask=mask,
                prompt=inputs.prompt,
                negative_prompt=inputs.neg_prompt,
                num_inference_steps=inputs.steps,
                guidance_scale=inputs.guidance,
                height=h,
                width=w,
                generator=generator,
                max_sequence_length=512,
            ).frames[0]
        else:
            video_frames = pipe(
                prompt=inputs.prompt,
                negative_prompt=inputs.neg_prompt,
                num_inference_steps=inputs.steps,
                guidance_scale=inputs.guidance,
                height=h,
                width=w,
                num_frames=inputs.frames,
                generator=generator,
                max_sequence_length=256,
            ).frames[0]

        import bpy
        render = bpy.context.scene.render
        fps = round(render.fps / render.fps_base, 3)
        src_path = export_to_video(video_frames, fps=fps)
        dst_path = solve_path(clean_filename(str(seed) + "_" + inputs.prompt) + ".mp4")
        shutil.move(src_path, dst_path)
        return dst_path
