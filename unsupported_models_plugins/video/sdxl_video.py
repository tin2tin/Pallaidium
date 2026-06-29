"""Frame-by-frame video processing via SDXL img2img (stabilityai/stable-diffusion-xl-base-1.0)."""

import shutil
import numpy as np
from PIL import Image
from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import (
    gfx_device, low_vram, solve_path, clean_filename, NoWatermark,
    process_video, process_image,
)


def _closest_div8(n: int) -> int:
    return max(8, (n // 8) * 8)


class SDXLVideoPlugin(ModelPlugin):
    MODEL_ID     = "stable-diffusion-xl/frame2frame"
    DISPLAY_NAME = "SDXL Frame-by-Frame"
    MODEL_TYPE   = "video"
    DESCRIPTION  = "Apply SDXL img2img to every frame of an input video or image strip"

    INPUTS       = InputSpec.PROMPT | InputSpec.NEG_PROMPT | InputSpec.IMAGE | InputSpec.LORA
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.NEG_PROMPT, UISection.VIDEO_STRIP,
        UISection.RESOLUTION, UISection.FRAMES, UISection.STEPS, UISection.GUIDANCE,
        UISection.IMAGE_STRENGTH, UISection.SEED, UISection.LORA,
    ]
    PARAMS       = ParamSpec(width=1024, height=576, steps=20, guidance=2.8)
    REQUIRED_PACKAGES = ["torch", "diffusers", "torchvision"]

    def load(self, prefs, scene, **kw):
        import torch
        from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderKL

        enabled_items = kw.get("enabled_items", [])
        print(f"Loading {self.MODEL_ID} (frame-by-frame)…")

        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16,
        )
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16, variant="fp16", vae=vae,
        )
        pipe.watermark = NoWatermark()

        if enabled_items:
            from ...utils.helpers import clean_filename as _cf, bpy
            lora_folder = getattr(bpy.context.scene, "lora_folder", "")
            names, weights = [], []
            for item in enabled_items:
                name = _cf(item.name).replace(".", "")
                names.append(name)
                weights.append(item.weight_value)
                pipe.load_lora_weights(
                    bpy.path.abspath(lora_folder),
                    weight_name=item.name + ".safetensors",
                    adapter_name=name,
                )
            pipe.set_adapters(names, adapter_weights=weights)

        if gfx_device == "mps":
            pipe.to("mps")
        elif low_vram():
            pipe.enable_sequential_cpu_offload()
            pipe.vae.enable_tiling()
        else:
            pipe.enable_model_cpu_offload()

        return {"pipe": pipe, "refiner": pipe, "last_model_card": self.MODEL_ID}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch
        from torchvision import transforms
        from diffusers.utils import export_to_video

        refiner = pipe_obj["refiner"]
        seed = inputs.seed
        generator = (
            torch.Generator("cuda").manual_seed(seed)
            if torch.cuda.is_available() and seed != 0 else None
        )

        pil_to_tensor = transforms.ToTensor()

        if inputs.video_path:
            tmp_out = solve_path("temp_images")
            frames = process_video(inputs.video_path, tmp_out)
        elif inputs.image is not None:
            import bpy
            frames = process_image(bpy.path.abspath(scene.image_path), inputs.frames)
        else:
            raise RuntimeError("SDXL frame-by-frame requires a video or image input.")

        video_frames = []
        for frame_idx, frame in enumerate(frames):
            if frame is None or not isinstance(frame, Image.Image):
                continue
            w, h = frame.size
            if w == 0 or h == 0:
                continue
            new_w = _closest_div8(w)
            new_h = _closest_div8(h)
            if (new_w, new_h) != (w, h):
                frame = frame.resize((new_w, new_h), Image.Resampling.LANCZOS)
            frame = transforms.functional.invert(frame)
            t = pil_to_tensor(frame).float()
            if t.numel() == 0:
                continue
            if t.ndim == 3:
                t = t.unsqueeze(0)
            try:
                image = refiner(
                    inputs.prompt,
                    image=t,
                    strength=1.0 - inputs.strength,
                    num_inference_steps=inputs.steps,
                    guidance_scale=inputs.guidance,
                    generator=generator,
                ).images[0]
                if image is not None and isinstance(image, Image.Image):
                    video_frames.append(image)
            except Exception as e:
                print(f"Frame {frame_idx} error: {e}")

        if not video_frames:
            raise RuntimeError("No frames were generated.")

        video_frames_np = np.array(video_frames)
        import bpy
        render = bpy.context.scene.render
        fps = round(render.fps / render.fps_base, 3)
        src_path = export_to_video(list(video_frames_np), fps=fps)
        dst_path = solve_path(clean_filename(str(seed) + "_" + inputs.prompt) + ".mp4")
        shutil.move(src_path, dst_path)
        return dst_path
