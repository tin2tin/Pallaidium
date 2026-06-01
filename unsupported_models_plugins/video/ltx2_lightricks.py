"""Img2vid and txt2vid via Lightricks/LTX-2 (4-bit quantized, with restoration post-process)."""

import gc
import numpy as np
from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, solve_path, clean_filename, load_first_frame


class LTX2LightricksPlugin(ModelPlugin):
    MODEL_ID     = "Lightricks/LTX-2"
    DISPLAY_NAME = "Video: LTX-2 Lightricks"
    MODEL_TYPE   = "video"
    DESCRIPTION  = "High-quality txt2vid and img2vid via Lightricks/LTX-2 (4-bit quantized)"

    INPUTS       = InputSpec.PROMPT | InputSpec.NEG_PROMPT | InputSpec.IMAGE | InputSpec.LORA
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.NEG_PROMPT, UISection.VIDEO_STRIP,
        UISection.RESOLUTION, UISection.FRAMES, UISection.SEED, UISection.LORA,
    ]
    PARAMS       = ParamSpec(width=512, height=320, frames=97, steps=8, guidance=1.0)
    REQUIRED_PACKAGES = ["torch", "diffusers", "transformers"]

    def load(self, prefs, scene, **kw):
        # Lightricks/LTX-2 loads all components inline during generate() to manage GPU memory.
        # Returning a no-op cache so the framework doesn't try to reload each batch.
        return {"pipe": None, "refiner": None, "last_model_card": self.MODEL_ID}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch
        import cv2
        import bpy
        from diffusers import LTX2ImageToVideoPipeline, LTX2Pipeline, LTX2LatentUpsamplePipeline, LTX2VideoTransformer3DModel
        from diffusers.pipelines.ltx2.export_utils import encode_video
        from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
        from transformers import Gemma3ForConditionalGeneration

        MODEL_PATH = self.MODEL_ID
        torch_dtype = torch.bfloat16
        device = "cuda"
        seed = inputs.seed

        generator = torch.Generator("cpu").manual_seed(seed)

        render = bpy.context.scene.render
        fps = round(render.fps / render.fps_base, 3)

        w = (inputs.width  // 32) * 32
        h = (inputs.height // 32) * 32
        target = inputs.frames
        num_frames = max(9, ((target - 1) // 8) * 8 + 1)

        def _cleanup():
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        image = inputs.image
        if image is None and inputs.video_path:
            image = load_first_frame(inputs.video_path)

        text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
            "OzzyGT/LTX-2-bnb-4bit-text-encoder", dtype=torch_dtype, device_map="cpu",
        )
        transformer = LTX2VideoTransformer3DModel.from_pretrained(
            "OzzyGT/LTX-2-bnb-4bit-transformer-distilled", torch_dtype=torch_dtype, device_map="cpu",
        )

        if image is not None:
            pipe = LTX2ImageToVideoPipeline.from_pretrained(
                MODEL_PATH, transformer=transformer, text_encoder=text_encoder, torch_dtype=torch_dtype,
            )
        else:
            pipe = LTX2Pipeline.from_pretrained(
                MODEL_PATH, transformer=transformer, text_encoder=text_encoder, torch_dtype=torch_dtype,
            )

        pipe.vae.enable_tiling(
            tile_sample_min_height=256, tile_sample_min_width=256, tile_sample_min_num_frames=16,
            tile_sample_stride_height=192, tile_sample_stride_width=192, tile_sample_stride_num_frames=8,
        )
        pipe.vae.use_framewise_encoding = True
        pipe.vae.use_framewise_decoding = True
        pipe.enable_model_cpu_offload()
        _cleanup()

        print("LTX-2 Lightricks: Stage 1")
        call_kw = dict(
            prompt=inputs.prompt,
            width=w, height=h, num_frames=num_frames, frame_rate=fps,
            num_inference_steps=8, sigmas=DISTILLED_SIGMA_VALUES,
            guidance_scale=1.0, generator=generator, output_type="latent", return_dict=False,
        )
        if image is not None:
            call_kw["image"] = image

        video_latent, audio_latent = pipe(**call_kw)

        print("LTX-2 Lightricks: Latent upsampling")
        latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
            "rootonchair/LTX-2-19b-distilled", subfolder="latent_upsampler", torch_dtype=torch_dtype,
        )
        upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=latent_upsampler)
        upsample_pipe.enable_model_cpu_offload(device=device)
        upscaled_latent = upsample_pipe(latents=video_latent, output_type="latent", return_dict=False)[0]

        latent_upsampler.to("cpu")
        del video_latent, upsample_pipe, latent_upsampler
        _cleanup()

        print("LTX-2 Lightricks: Stage 2 (high-res decode)")
        stage2_kw = dict(
            latents=upscaled_latent, audio_latents=audio_latent,
            prompt=inputs.prompt, negative_prompt=inputs.neg_prompt,
            width=w * 2, height=h * 2, num_frames=num_frames,
            num_inference_steps=3, noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
            sigmas=STAGE_2_DISTILLED_SIGMA_VALUES, generator=generator,
            guidance_scale=1.0, output_type="np", return_dict=False,
        )
        if image is not None:
            stage2_kw["image"] = image

        video, audio = pipe(**stage2_kw)

        # Restoration post-process
        video_np = (video[0] * 255).round().astype("uint8")
        processed = []
        for frame in video_np:
            frame_f = frame.astype(np.float32) / 255.0
            mask = frame_f > 0.85
            frame_f[mask] = 0.85 + (frame_f[mask] - 0.85) * 0.5
            restored = np.clip(frame_f * 255, 0, 255).astype(np.uint8)
            denoised = cv2.bilateralFilter(restored, d=5, sigmaColor=30, sigmaSpace=30)
            ff = denoised.astype(np.float32) / 255.0
            blur_s = cv2.GaussianBlur(ff, (0, 0), 0.6)
            micro = ff - blur_s
            micro_b = ff + micro * 0.8
            blur_l = cv2.GaussianBlur(ff, (0, 0), 2.0)
            struct = ff - blur_l
            final = np.clip(micro_b + struct * 0.4, 0, 1)
            processed.append((final * 255).astype(np.uint8))

        video_final = torch.from_numpy(np.stack(processed))
        dst_path = solve_path(clean_filename(str(seed) + "_" + inputs.prompt) + ".mp4")
        encode_video(
            video_final,
            fps=fps,
            audio=audio[0].float().cpu(),
            audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
            output_path=dst_path,
        )
        return dst_path
