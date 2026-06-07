"""Text-to-video and img2vid via SkyReels V1 (HunyuanVideo backbone, int4 quantized)."""

import shutil
from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram, solve_path, clean_filename, load_first_frame


def _ensure_skyreel(prompt: str) -> str:
    if not prompt.startswith("FPS-24,"):
        return "FPS-24, " + prompt
    return prompt


class SkyReelsPlugin(ModelPlugin):
    MODEL_ID     = "Skywork/SkyReels-V1-Hunyuan-T2V"
    DISPLAY_NAME = "Video: SkyReels V1"
    MODEL_TYPE   = "video"
    DESCRIPTION  = "Text-to-video and img2vid via SkyReels V1 (HunyuanVideo int4)"

    INPUTS       = InputSpec.PROMPT | InputSpec.NEG_PROMPT | InputSpec.IMAGE
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.NEG_PROMPT, UISection.VIDEO_STRIP,
        UISection.RESOLUTION, UISection.FRAMES, UISection.STEPS, UISection.GUIDANCE, UISection.SEED,
    ]
    PARAMS       = ParamSpec(width=848, height=480, frames=97, steps=50, guidance=1.0)
    REQUIRED_PACKAGES = ["torch", "diffusers"]

    def load(self, prefs, scene, **kw):
        import torch
        from diffusers import HunyuanVideoTransformer3DModel

        _cache_dir = prefs.hf_cache_dir or None
        mode = kw.get("mode", "txt2vid")
        print(f"Loading {self.MODEL_ID} ({mode})…")

        _lfo = prefs.local_files_only
        if mode in ("img2vid", "vid2vid"):
            from diffusers import HunyuanSkyreelsImageToVideoPipeline
            model_id = "hunyuanvideo-community/HunyuanVideo"
            transformer = HunyuanVideoTransformer3DModel.from_pretrained(
                "newgenai79/SkyReels-V1-Hunyuan-I2V-int4",
                subfolder="transformer", torch_dtype=torch.bfloat16, cache_dir=_cache_dir,
                local_files_only=_lfo,
            )
            pipe = HunyuanSkyreelsImageToVideoPipeline.from_pretrained(
                model_id, transformer=transformer, torch_dtype=torch.float16, cache_dir=_cache_dir,
                local_files_only=_lfo,
            )
        else:
            from diffusers import HunyuanVideoPipeline
            transformer = HunyuanVideoTransformer3DModel.from_pretrained(
                "newgenai79/SkyReels-V1-Hunyuan-T2V-int4",
                subfolder="transformer", torch_dtype=torch.bfloat16, cache_dir=_cache_dir,
                local_files_only=_lfo,
            )
            transformer.enable_layerwise_casting(
                storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16
            )
            pipe = HunyuanVideoPipeline.from_pretrained(
                "newgenai79/HunyuanVideo-int4", transformer=transformer, torch_dtype=torch.float16,
                cache_dir=_cache_dir, local_files_only=_lfo,
            )

        if gfx_device == "mps":
            pipe.to("mps")
        elif low_vram():
            pipe.vae.enable_tiling()
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

        prompt = _ensure_skyreel(inputs.prompt)

        image = inputs.image
        if image is None and inputs.video_path:
            image = load_first_frame(inputs.video_path)

        self.set_phase(inputs, "Generating")
        if image is not None:
            video_frames = pipe(
                image=image,
                prompt=prompt,
                negative_prompt=inputs.neg_prompt,
                num_inference_steps=inputs.steps,
                guidance_scale=inputs.guidance,
                height=inputs.height,
                width=inputs.width,
                num_frames=inputs.frames,
                generator=generator,
                max_sequence_length=512,
                callback_on_step_end=self.step_callback(inputs),
            ).frames[0]
        else:
            video_frames = pipe(
                prompt=prompt,
                negative_prompt=inputs.neg_prompt,
                num_inference_steps=inputs.steps,
                guidance_scale=inputs.guidance,
                height=inputs.height,
                width=inputs.width,
                num_frames=inputs.frames,
                generator=generator,
                max_sequence_length=512,
                callback_on_step_end=self.step_callback(inputs),
            ).frames[0]

        self.set_phase(inputs, "Saving")
        import bpy
        render = bpy.context.scene.render
        fps = round(render.fps / render.fps_base, 3)
        src_path = export_to_video(video_frames, fps=fps)
        dst_path = solve_path(clean_filename(str(seed) + "_" + inputs.prompt) + ".mp4")
        shutil.move(src_path, dst_path)
        return dst_path
