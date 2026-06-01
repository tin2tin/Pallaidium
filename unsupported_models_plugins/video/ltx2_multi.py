"""Multi-input (image + audio) video generation via LTX-2 (custom audio-to-video pipeline)."""

import gc
from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, solve_path, clean_filename, load_first_frame


class LTX2MultiPlugin(ModelPlugin):
    MODEL_ID     = "LTX-2 Multi-Input File"
    DISPLAY_NAME = "Video: LTX-2 (Multimodal)"
    MODEL_TYPE   = "video"
    DESCRIPTION  = "LTX-2 audio-to-video pipeline supporting image + audio inputs"

    INPUTS       = InputSpec.PROMPT | InputSpec.NEG_PROMPT | InputSpec.IMAGE | InputSpec.LORA | InputSpec.AUDIO_REF
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.NEG_PROMPT, UISection.VIDEO_STRIP,
        UISection.RESOLUTION, UISection.FRAMES, UISection.SEED, UISection.LORA,
    ]
    PARAMS       = ParamSpec(width=512, height=320, frames=121, steps=8, guidance=1.0)
    REQUIRED_PACKAGES = ["torch", "diffusers", "transformers"]

    def load(self, prefs, scene, **kw):
        # All loading happens inside generate() to manage multi-stage GPU memory.
        return {"pipe": None, "refiner": None, "last_model_card": self.MODEL_ID}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch
        import bpy
        from PIL import Image
        from diffusers import (
            AutoencoderKLLTX2Video,
            LTX2LatentUpsamplePipeline,
            LTX2Pipeline,
            LTX2VideoTransformer3DModel,
        )
        from diffusers.pipelines.ltx2.export_utils import encode_video
        from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        from transformers import Gemma3ForConditionalGeneration

        try:
            from sdnq.common import use_torch_compile as triton_is_available
            from sdnq.loader import apply_sdnq_options_to_model
        except ImportError:
            triton_is_available = False
            apply_sdnq_options_to_model = None

        MODEL_PATH = "Lightricks/LTX-2"
        torch_dtype = torch.bfloat16
        onload_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        offload_device = torch.device("cpu")
        fps = 24.0
        seed = inputs.seed
        generator = (
            torch.Generator("cuda").manual_seed(seed)
            if torch.cuda.is_available() and seed != 0 else None
        )

        def _cleanup():
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        w = inputs.width
        h = inputs.height

        # Resolve image
        image = inputs.image
        if image is None and inputs.video_path:
            image = load_first_frame(inputs.video_path)

        # Resolve audio
        sound_path = inputs.audio_ref

        # Frame count (must follow 8n+1 rule)
        if sound_path:
            try:
                import soundfile as sf
                info = sf.info(sound_path)
                dur_s = info.frames / info.samplerate
            except Exception:
                dur_s = inputs.frames / fps
            raw = dur_s * fps
            num_frames = int(((raw + 7) // 8) * 8) + 1
        else:
            target = inputs.frames
            num_frames = max(9, ((target - 1) // 8) * 8 + 1)

        if image is None:
            image = Image.new("RGB", (w, h), (0, 0, 0))

        _cleanup()

        # --- TEXT ENCODING ---
        print("LTX-2 Multi: Text encoding")
        text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
            "OzzyGT/LTX-2-bnb-8bit-text-encoder", dtype=torch_dtype,
        )
        embeds_pipe = LTX2Pipeline.from_pretrained(
            MODEL_PATH,
            text_encoder=text_encoder,
            transformer=None, vae=None, audio_vae=None, vocoder=None,
            scheduler=None, connectors=None,
            torch_dtype=torch_dtype,
        )
        embeds_pipe.enable_sequential_cpu_offload()
        with torch.inference_mode():
            prompt_embeds, prompt_attention_mask, _, _ = embeds_pipe.encode_prompt(
                inputs.prompt, inputs.neg_prompt, do_classifier_free_guidance=False
            )
        prompt_embeds = prompt_embeds.detach().to(offload_device, copy=True)
        prompt_attention_mask = prompt_attention_mask.detach().to(offload_device, copy=True)
        del embeds_pipe, text_encoder
        _cleanup()

        # --- STAGE 1 ---
        print("LTX-2 Multi: Stage 1")
        transformer = LTX2VideoTransformer3DModel.from_pretrained(
            "OzzyGT/LTX_2_SDNQ_4bit_dynamic_distilled_transformer",
            torch_dtype=torch_dtype, device_map="cpu",
        )
        if triton_is_available and torch.cuda.is_available() and apply_sdnq_options_to_model:
            transformer = apply_sdnq_options_to_model(transformer, use_quantized_matmul=True)

        pipe = LTX2Pipeline.from_pretrained(
            "rootonchair/LTX-2-19b-distilled",
            custom_pipeline="multimodalart/ltx2-audio-to-video",
            transformer=transformer,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

        # Apply user LoRAs
        enabled_items = kw.get("enabled_items", [])
        if enabled_items:
            from ...utils.helpers import clean_filename, bpy
            lora_folder = getattr(bpy.context.scene, "lora_folder", "")
            names, weights = [], []
            for item in enabled_items:
                name = clean_filename(item.name).replace(".", "")
                names.append(name)
                weights.append(item.weight_value)
                pipe.load_lora_weights(
                    bpy.path.abspath(lora_folder),
                    weight_name=item.name + ".safetensors",
                    adapter_name=name,
                )
            pipe.set_adapters(names, adapter_weights=weights)

        pipe.enable_group_offload(
            onload_device=onload_device, offload_device=offload_device,
            offload_type="leaf_level", low_cpu_mem_usage=True,
        )

        stage1_kw = dict(
            prompt_embeds=prompt_embeds.to(onload_device),
            prompt_attention_mask=prompt_attention_mask.to(onload_device),
            width=w, height=h, num_frames=num_frames, frame_rate=fps,
            num_inference_steps=8, sigmas=DISTILLED_SIGMA_VALUES,
            guidance_scale=1.0, generator=generator,
            output_type="latent", return_dict=False,
        )
        if sound_path:
            stage1_kw["audio"] = sound_path
        if image is not None:
            stage1_kw["image"] = image

        with torch.inference_mode():
            outputs = pipe(**stage1_kw)

        if isinstance(outputs, tuple):
            video_latent = outputs[0]
            audio_latent = outputs[1] if len(outputs) > 1 else None
        else:
            video_latent = outputs
            audio_latent = None

        video_latent = video_latent.detach().to(offload_device, copy=True)
        if audio_latent is not None:
            audio_latent = audio_latent.detach().to(offload_device, copy=True)
        del pipe, transformer
        _cleanup()

        # --- LATENT UPSAMPLE ---
        print("LTX-2 Multi: Latent upsampling")
        latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
            "rootonchair/LTX-2-19b-distilled", subfolder="latent_upsampler", torch_dtype=torch_dtype,
        ).to(onload_device)
        vae = AutoencoderKLLTX2Video.from_pretrained(
            MODEL_PATH, subfolder="vae", torch_dtype=torch_dtype,
        ).to(onload_device)
        upscale_pipe = LTX2LatentUpsamplePipeline(vae=vae, latent_upsampler=latent_upsampler)
        upscale_pipe.enable_model_cpu_offload(device=onload_device)
        with torch.inference_mode():
            up_latent = upscale_pipe(
                latents=video_latent, output_type="latent", return_dict=False,
            )[0]
        up_latent = up_latent.detach().to(offload_device, copy=True)
        del upscale_pipe, latent_upsampler, vae, video_latent
        _cleanup()

        # --- STAGE 2 (REFINE) ---
        print("LTX-2 Multi: Stage 2 refinement")
        transformer = LTX2VideoTransformer3DModel.from_pretrained(
            "OzzyGT/LTX_2_SDNQ_4bit_dynamic_distilled_transformer",
            torch_dtype=torch_dtype, device_map="cpu",
        )
        if triton_is_available and torch.cuda.is_available() and apply_sdnq_options_to_model:
            transformer = apply_sdnq_options_to_model(transformer, use_quantized_matmul=True)

        refine_pipe = LTX2Pipeline.from_pretrained(
            MODEL_PATH, transformer=transformer, text_encoder=None, torch_dtype=torch_dtype,
        )
        refine_pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            refine_pipe.scheduler.config, use_dynamic_shifting=False, shift_terminal=None,
        )
        adapter_name = clean_filename("ltx-2-19b-ic-lora-detailer.safetensors").replace(".", "")
        try:
            refine_pipe.load_lora_weights(
                "Lightricks/LTX-2-19b-IC-LoRA-Detailer",
                weight_name="ltx-2-19b-ic-lora-detailer.safetensors",
                adapter_name=adapter_name,
            )
            refine_pipe.set_adapters(adapter_name)
        except Exception as e:
            print(f"IC-LoRA load failed (continuing): {e}")

        refine_pipe.enable_group_offload(
            onload_device=onload_device, offload_device=offload_device,
            offload_type="leaf_level", low_cpu_mem_usage=True,
        )
        refine_kw = dict(
            latents=up_latent.to(onload_device),
            prompt_embeds=prompt_embeds.to(onload_device),
            prompt_attention_mask=prompt_attention_mask.to(onload_device),
            width=w * 2, height=h * 2, num_frames=num_frames,
            num_inference_steps=3, sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
            noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
            guidance_scale=1.0, generator=generator,
            output_type="latent", return_dict=False,
        )
        if audio_latent is not None:
            refine_kw["audio_latents"] = audio_latent.to(onload_device)
        with torch.inference_mode():
            outputs = refine_pipe(**refine_kw)
        if isinstance(outputs, tuple):
            final_v = outputs[0]
            final_a = outputs[1] if len(outputs) > 1 else None
        else:
            final_v = outputs
            final_a = None
        del refine_pipe, transformer, up_latent, audio_latent
        _cleanup()

        # --- DECODE ---
        print("LTX-2 Multi: Decode")
        decode_pipe = LTX2Pipeline.from_pretrained(
            MODEL_PATH, text_encoder=None, transformer=None,
            scheduler=None, connectors=None, torch_dtype=torch_dtype,
        )
        decode_pipe.to(onload_device)
        decode_pipe.vae.enable_tiling(
            tile_sample_min_height=256, tile_sample_min_width=256, tile_sample_min_num_frames=16,
            tile_sample_stride_height=192, tile_sample_stride_width=192, tile_sample_stride_num_frames=8,
        )
        decode_pipe.vae.use_framewise_encoding = True
        decode_pipe.vae.use_framewise_decoding = True
        decode_pipe.enable_model_cpu_offload()

        with torch.inference_mode():
            video = decode_pipe.vae.decode(
                final_v.to(onload_device, dtype=decode_pipe.vae.dtype), None, return_dict=False,
            )[0]
            video = decode_pipe.video_processor.postprocess_video(video, output_type="np")
            audio_out = None
            if final_a is not None:
                mel = decode_pipe.audio_vae.decode(
                    final_a.to(onload_device, dtype=decode_pipe.audio_vae.dtype), return_dict=False,
                )[0]
                audio_out = decode_pipe.vocoder(mel)
        del decode_pipe
        _cleanup()

        video_tensor = torch.from_numpy((video * 255).round().astype("uint8"))
        dst_path = solve_path(clean_filename(str(seed) + "_" + inputs.prompt) + ".mp4")
        if audio_out is not None:
            encode_video(
                video_tensor[0], fps=fps,
                audio=audio_out[0].float().cpu(), audio_sample_rate=24000,
                output_path=dst_path,
            )
        else:
            encode_video(video_tensor[0], fps=fps, output_path=dst_path)

        return dst_path
