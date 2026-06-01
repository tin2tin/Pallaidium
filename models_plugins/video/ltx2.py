"""Img2vid via LTX-2 19b distilled (rootonchair) — 3-stage: latent → upsample → decode."""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram, solve_path, clean_filename, load_first_frame


class LTX2Plugin(ModelPlugin):
    MODEL_ID     = "rootonchair/LTX-2-19b-distilled"
    DISPLAY_NAME = "Video: LTX-2 19b (distilled)"
    MODEL_TYPE   = "video"
    DESCRIPTION  = "High-quality img2vid via LTX-2 19b distilled with latent upsampling"

    INPUTS       = InputSpec.PROMPT | InputSpec.NEG_PROMPT | InputSpec.IMAGE | InputSpec.LORA
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.NEG_PROMPT, UISection.VIDEO_STRIP,
        UISection.RESOLUTION, UISection.FRAMES, UISection.SEED, UISection.LORA,
    ]
    PARAMS       = ParamSpec(width=512, height=320, frames=97, steps=8, guidance=1.0)
    REQUIRED_PACKAGES = ["torch", "diffusers"]

    def load(self, prefs, scene, **kw):
        import torch
        from diffusers.pipelines.ltx2 import LTX2ImageToVideoPipeline

        _cache_dir = prefs.hf_cache_dir or None
        print(f"Loading {self.MODEL_ID}…")
        pipe = LTX2ImageToVideoPipeline.from_pretrained(self.MODEL_ID, torch_dtype=torch.bfloat16, cache_dir=_cache_dir)

        if gfx_device == "mps":
            pipe.to("mps")
        else:
            pipe.vae.enable_tiling()
            pipe.enable_sequential_cpu_offload(device=gfx_device)

        return {"pipe": pipe, "refiner": None, "last_model_card": self.MODEL_ID}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch
        import bpy
        from diffusers.pipelines.ltx2 import LTX2LatentUpsamplePipeline
        from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
        from diffusers.pipelines.ltx2.export_utils import encode_video

        pipe = pipe_obj["pipe"]
        seed = inputs.seed
        generator = (
            torch.Generator("cuda").manual_seed(seed)
            if torch.cuda.is_available() and seed != 0 else None
        )

        render = bpy.context.scene.render
        fps = round(render.fps / render.fps_base, 3)

        w = (inputs.width  // 32) * 32
        h = (inputs.height // 32) * 32
        target = inputs.frames
        num_frames = max(9, ((target - 1) // 8) * 8 + 1)

        image = inputs.image
        if image is None and inputs.video_path:
            image = load_first_frame(inputs.video_path)

        # Stage 1
        self.set_phase(inputs, "Stage 1: image → latents")
        video_latent, audio_latent = pipe(
            image=image,
            prompt=inputs.prompt,
            negative_prompt=inputs.neg_prompt,
            width=w,
            height=h,
            num_frames=num_frames,
            frame_rate=fps,
            max_sequence_length=512,
            num_inference_steps=8,
            sigmas=DISTILLED_SIGMA_VALUES,
            guidance_scale=1.0,
            generator=generator,
            output_type="latent",
            return_dict=False,
        )

        # Stage 1.5 — latent upsample
        self.set_phase(inputs, "Stage 1.5: latent upsampling")
        _cache_dir = prefs.hf_cache_dir or None
        latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
            self.MODEL_ID, subfolder="latent_upsampler", torch_dtype=torch.bfloat16,
            cache_dir=_cache_dir,
        )
        upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=latent_upsampler)
        upsample_pipe.enable_model_cpu_offload(device=gfx_device)
        upscaled = upsample_pipe(latents=video_latent, output_type="latent", return_dict=False)[0]

        # Stage 2 — decode at 2×
        self.set_phase(inputs, "Stage 2: decode")
        video, audio = pipe(
            image=image,
            latents=upscaled,
            audio_latents=audio_latent,
            prompt=inputs.prompt,
            negative_prompt=inputs.neg_prompt,
            width=w * 2,
            height=h * 2,
            num_frames=num_frames,
            num_inference_steps=3,
            noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
            sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
            generator=generator,
            guidance_scale=1.0,
            output_type="np",
            return_dict=False,
            callback_on_step_end=self.step_callback(inputs),
        )

        self.set_phase(inputs, "Saving")
        video_tensor = torch.from_numpy((video * 255).round().astype("uint8"))
        dst_path = solve_path(clean_filename(str(seed) + "_" + inputs.prompt) + ".mp4")
        encode_video(
            video_tensor[0],
            fps=fps,
            audio=audio[0].float().cpu(),
            audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
            output_path=dst_path,
        )
        return dst_path
