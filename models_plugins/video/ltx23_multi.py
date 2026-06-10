"""Two-stage distilled text/image/audio-to-video via LTX-2.3 (OzzyGT/LTX-2.3-Distilled, SDNQ 4-bit)."""

import os
import gc
import ctypes

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, solve_path, clean_filename, load_first_frame


def vae_temporal_decode_streaming(vae, latents_cpu, *, decode_device, temb=None):
    import torch
    """Streaming temporal decode — faster than spatial tiling on ≥16 GB cards."""
    tile_latent_min    = vae.tile_sample_min_num_frames // vae.temporal_compression_ratio
    n_latent_frames    = latents_cpu.shape[2]
    n_sample_frames    = (n_latent_frames - 1) * vae.temporal_compression_ratio + 1
    latent_stride      = vae.tile_sample_stride_num_frames // vae.temporal_compression_ratio
    sample_stride      = vae.tile_sample_stride_num_frames   # sample-space stride (8× latent_stride)
    blend_n            = vae.tile_sample_min_num_frames - vae.tile_sample_stride_num_frames

    result_tiles, prev_tile =[], None
    for i in range(0, n_latent_frames, latent_stride):
        tile_cpu = latents_cpu[:, :, i : i + tile_latent_min + 1, :, :]
        tile = tile_cpu.to(device=decode_device, dtype=vae.dtype, non_blocking=True)
        saved = vae.use_framewise_decoding
        vae.use_framewise_decoding = False
        decoded = vae.decode(tile, temb=temb, return_dict=False)[0]
        vae.use_framewise_decoding = saved
        row = decoded.cpu()
        if i > 0:
            row = row[:, :, :-1, :, :]
        if prev_tile is None:
            result_tiles.append(row[:, :, : sample_stride + 1, :, :])
        else:
            stitched = vae.blend_t(prev_tile, row, blend_n)
            result_tiles.append(stitched[:, :, :sample_stride, :, :])
        prev_tile = row
        del tile, decoded
    return torch.cat(result_tiles, dim=2)[:, :, :n_sample_frames]


class LTX2_3MultiPlugin(ModelPlugin):
    MODEL_ID     = "LTX-2.3 Multi-Input File"
    DISPLAY_NAME = "Video: LTX-2.3 (Multimodal)"
    MODEL_TYPE   = "video"
    DESCRIPTION  = "Two-stage distilled LTX-2.3 (SDNQ 4-bit) — text/image/audio-to-video with audio output"

    # All multimodal inputs are defined but optional
    INPUTS       = InputSpec.PROMPT | InputSpec.NEG_PROMPT | InputSpec.IMAGE | InputSpec.LORA | InputSpec.AUDIO_REF
    UI_SECTIONS  =[
        UISection.PROMPT, UISection.NEG_PROMPT, UISection.VIDEO_STRIP,
        UISection.RESOLUTION, UISection.FRAMES, UISection.SEED, UISection.LORA,
    ]
    # Standard ParamSpec without unsupported UI fields
    PARAMS            = ParamSpec(width=1920, height=896, frames=121, steps=8, guidance=1.0)
    REQUIRED_PACKAGES =["torch", "torchaudio", "soundfile", "av", "diffusers", "transformers", "sdnq"]
    supports_inpaint  = False

    def load(self, prefs, scene, **kw):
        return {"pipe": None, "refiner": None, "last_model_card": self.MODEL_ID}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs) -> str:
        import torch
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch._dynamo.config.disable = True

        # print("=" * 60)
        # print("[LTX23Multi] generate() received inputs:")
        # print(f"  inputs.prompt     = {inputs.prompt!r}")
        # print(f"  inputs.neg_prompt = {inputs.neg_prompt!r}")
        # print(f"  inputs.image      = {'<PIL Image>' if inputs.image is not None else None}")
        # print(f"  inputs.video_path = {inputs.video_path!r}")
        # print(f"  inputs.audio_ref  = {inputs.audio_ref!r}")
        # print(f"  inputs.width      = {inputs.width}, inputs.height = {inputs.height}")
        # print(f"  inputs.frames     = {inputs.frames}, inputs.seed = {inputs.seed}")
        # print("=" * 60)

        from diffusers import LTX2VideoTransformer3DModel
        from diffusers.pipelines.ltx2.export_utils import encode_video
        from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        from transformers import Gemma3ForConditionalGeneration

        try:
            from ._pipeline_ltx2_multimodal import LTX2MultiModalPipeline, LTX2AudioCondition, LTX2ImageCondition, load_audio
        except ImportError:
            from _pipeline_ltx2_multimodal import LTX2MultiModalPipeline, LTX2AudioCondition, LTX2ImageCondition, load_audio

        _cache_dir     = prefs.hf_cache_dir or None
        _lfo           = prefs.local_files_only
        MODEL_PATH     = "OzzyGT/LTX-2.3-Distilled"
        SDNQ_PATH      = "OzzyGT/LTX-2.3-Distilled-sdnq-dynamic-int4"
        UPSAMPLER_PATH = "OzzyGT/LTX-2.3-upsampler-x2"

        torch_dtype    = torch.bfloat16
        onload_device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        offload_device = torch.device("cpu")
        fps            = 24.0

        seed = inputs.seed or torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device="cpu").manual_seed(seed)

        def _flush():
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass

        # ── Stage Resolution Match Fix ──────────────────────────────────────
        stage1_w = max(32, round((inputs.width / 2) / 32) * 32)
        stage1_h = max(32, round((inputs.height / 2) / 32) * 32)
        w = stage1_w * 2
        h = stage1_h * 2

        # ── Resolve Image & Audio Inputs (With Priority Overrides) ──────────
        image_input = inputs.image
        
        # Robustly find the video track mapping
        vid_path = None
        for attr in["video_path", "video", "video_ref"]:
            val = getattr(inputs, attr, None)
            if val and isinstance(val, str) and os.path.exists(val):
                vid_path = val
                break
                
        # Robustly find the dedicated Sound Strip mapping
        explicit_audio = None
        for attr in["audio_path", "audio", "audio_ref", "sound", "sound_path"]:
            val = getattr(inputs, attr, None)
            if val and isinstance(val, str) and os.path.exists(val):
                explicit_audio = val
                break

        sound_path = explicit_audio

        # Probe the Video Strip file for modalities if provided
        if vid_path:
            try:
                import av
                with av.open(vid_path) as container:
                    has_video = any(s.type == 'video' for s in container.streams)
                    has_audio = any(s.type == 'audio' for s in container.streams)
                    
                    if has_video and image_input is None:
                        image_input = load_first_frame(vid_path)
                        # print("[DEBUG] Extracted video track for Image Condition.")

                    if has_audio:
                        if explicit_audio:
                            # print(f"[DEBUG] Video contains audio, but overriding with explicitly provided Sound Strip audio.")
                            pass
                        else:
                            sound_path = vid_path
                            # print("[DEBUG] Extracted audio track from video for Audio Condition.")
            except Exception as e:
                # print(f"[DEBUG] Could not probe video_path with av ({e}). Falling back to simple frame extraction.")
                if image_input is None:
                    try:
                        image_input = load_first_frame(vid_path)
                    except Exception:
                        pass

        # print("="*50)
        # print("[DEBUG] LTX-2.3 Multimodal Media Detection:")
        # print(f"  -> Image Input Provided: {'YES' if image_input is not None else 'NO'}")
        # print(f"  -> Audio Input Provided: {'YES' if sound_path is not None else 'NO'}")
        # if sound_path:
        #     print(f"  -> Active Audio Path: {sound_path}")
        # print("="*50)

        # ── Frame Count Calculation ─────────────────────────────────────────
        if sound_path:
            dur_s = None
            try:
                # Try reading with soundfile (works for wav/flac/ogg)
                import soundfile as sf
                info = sf.info(sound_path)
                dur_s = info.frames / info.samplerate
            except Exception as e1:
                try:
                    # Fallback to PyAV (works for MP4/WebM video containers)
                    import av
                    with av.open(sound_path) as container:
                        audio_stream = next((s for s in container.streams if s.type == 'audio'), None)
                        if audio_stream and audio_stream.duration:
                            dur_s = float(audio_stream.duration * audio_stream.time_base)
                except Exception as e2:
                    pass

            if dur_s is None:
                dur_s = inputs.frames / fps

            if inputs.frames > 0:
                # inputs.frames is authoritative (resolved from the strip at queue time).
                # Use it directly for LTX 8-frame alignment so rounding in the audio
                # duration probe can't produce a mismatched frame count.
                num_frames = max(9, ((inputs.frames - 1) // 8) * 8 + 1)
                # Warn only when audio is genuinely much longer (trimming likely failed).
                _audio_frames = int(((dur_s * fps + 7) // 8) * 8) + 1
                if _audio_frames > num_frames + 8:
                    print(f"[LTX23Multi] WARN audio={_audio_frames} fr >> requested={num_frames} fr "
                          f"(likely untrimmed) — clamping dur_s")
                    dur_s = num_frames / fps
                else:
                    dur_s = num_frames / fps
            else:
                raw = dur_s * fps
                num_frames = max(9, int(((raw + 7) // 8) * 8) + 1)
                dur_s = num_frames / fps
        else:
            target = inputs.frames
            num_frames = max(9, ((target - 1) // 8) * 8 + 1)
            dur_s = num_frames / fps
            
        # print(f"[DEBUG] Final Target Number of frames: {num_frames}")
        _flush()

        # ── Parse Image Conditions (FLF / last-frame-only / single-frame) ─────
        # print("[LTX23Multi] ── Image condition debug ──────────────────────────────")
        # print(f"[LTX23Multi]   inputs.image     = {'<PIL ' + str(getattr(inputs.image, 'size', '?')) + '>' if inputs.image is not None else None}")
        # print(f"[LTX23Multi]   inputs.last_image= {'<PIL ' + str(getattr(getattr(inputs, 'last_image', None), 'size', '?')) + '>' if getattr(inputs, 'last_image', None) is not None else None}")
        # print(f"[LTX23Multi]   inputs.video_path= {getattr(inputs, 'video_path', None)!r}")
        # print(f"[LTX23Multi]   inputs.audio_ref = {getattr(inputs, 'audio_ref', None)!r}")

        image_conditions = None
        last_input = getattr(inputs, "last_image", None)

        if last_input is not None:
            if isinstance(last_input, str):
                from diffusers.utils import load_image
                last_input = load_image(last_input).convert("RGB")
            elif hasattr(last_input, "convert"):
                last_input = last_input.convert("RGB")
            # print(f"[LTX23Multi]   last_input PIL size={last_input.size}")

        if image_input is not None:
            if isinstance(image_input, str):
                from diffusers.utils import load_image
                image_input = load_image(image_input).convert("RGB")
            elif hasattr(image_input, "convert"):
                image_input = image_input.convert("RGB")
            # print(f"[LTX23Multi]   image_input PIL size={image_input.size}")

        _middle_paths = getattr(inputs, "middle_images_paths", [])

        if image_input is not None and last_input is not None and _middle_paths:
            # Mode MA: multi-anchor — first + N intermediate anchors + last
            from diffusers.utils import load_image as _load_image
            image_conditions = [
                LTX2ImageCondition(image=image_input, frame=0, strength=1.0),
            ]
            for _mp, _frac in _middle_paths:
                _frame_idx = round(_frac * (num_frames - 1))
                _frame_idx = max(1, min(num_frames - 2, _frame_idx))
                try:
                    _mid_pil = _load_image(_mp).convert("RGB").resize((inputs.width, inputs.height))
                    image_conditions.append(LTX2ImageCondition(image=_mid_pil, frame=_frame_idx, strength=1.0))
                except Exception as _e:
                    print(f"[LTX23Multi] WARNING: skipping middle anchor {_mp!r}: {_e}")
            image_conditions.append(LTX2ImageCondition(image=last_input, frame=-1, strength=1.0))
            print(f"[LTX23Multi] MODE: MULTI-ANCHOR — {len(image_conditions)} conditions, num_frames={num_frames}")
        elif image_input is not None and last_input is not None:
            # Mode A: FLF — hard anchor image 1 at start, soft keyframe image 2 at end
            image_conditions = [
                LTX2ImageCondition(image=image_input, frame=0,  strength=1.0),
                LTX2ImageCondition(image=last_input,  frame=-1, strength=1.0),
            ]
            # print(f"[LTX23Multi]   MODE: FLF — img1@0 + img2@{num_frames-1} — num_frames={num_frames}")
        elif last_input is not None:
            # Mode B: last-frame-only — soft keyframe image 2 at end
            image_conditions = [LTX2ImageCondition(image=last_input, frame=-1, strength=1.0)]
            # print(f"[LTX23Multi]   MODE: last-frame-only — img2@{num_frames-1} — num_frames={num_frames}")
        elif image_input is not None:
            # Existing: single first-frame condition
            image_conditions = [LTX2ImageCondition(image=image_input, frame=0, strength=1.0)]
            # print("[LTX23Multi]   MODE: single first-frame (frame=0)")
        # else:
            # print("[LTX23Multi]   MODE: no image conditions (text-to-video)")
        # print(f"[LTX23Multi]   image_conditions count={len(image_conditions) if image_conditions else 0}")
        # print("[LTX23Multi] ────────────────────────────────────────────────────────")

        # ── Step 0: Text encoding ───────────────────────────────────────────
        self.set_phase(inputs, "Text encoding")
        text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
            SDNQ_PATH, subfolder="text_encoder", torch_dtype=torch_dtype, cache_dir=_cache_dir,
            local_files_only=_lfo,
        )
        embeds_pipe = LTX2MultiModalPipeline.from_pretrained(
            MODEL_PATH,
            text_encoder=text_encoder,
            transformer=None, vae=None, audio_vae=None, vocoder=None,
            scheduler=None,
            torch_dtype=torch_dtype, cache_dir=_cache_dir, local_files_only=_lfo,
        )
        embeds_pipe.to(onload_device)
        with torch.inference_mode():
            prompt_embeds, prompt_attention_mask, _, _ = embeds_pipe.encode_prompt(
                prompt=inputs.prompt,
                negative_prompt=inputs.neg_prompt,
                do_classifier_free_guidance=False,
            )
        prompt_embeds          = prompt_embeds.detach().to(offload_device, copy=True)
        prompt_attention_mask  = prompt_attention_mask.detach().to(offload_device, copy=True)
        del embeds_pipe, text_encoder
        _flush()

        # ── Stage 1: Base generation ────────────────────────────────────────
        self.set_phase(inputs, f"Stage 1: generating {stage1_w}×{stage1_h}")
        # diffusers doesn't know the "sdnq" quant_method — use sdnq's own loader.
        import os as _os
        from huggingface_hub import snapshot_download as _snap
        from sdnq.loader import load_sdnq_model as _load_sdnq
        _sdnq_transformer_path = _os.path.join(
            _snap(SDNQ_PATH, cache_dir=_cache_dir, local_files_only=_lfo), "transformer"
        )
        transformer = _load_sdnq(
            model_path=_sdnq_transformer_path, model_cls=LTX2VideoTransformer3DModel,
            dtype=torch_dtype, device="cpu",
        )
        pipe = LTX2MultiModalPipeline.from_pretrained(
            MODEL_PATH,
            transformer=transformer,
            text_encoder=None, tokenizer=None,
            torch_dtype=torch_dtype, cache_dir=_cache_dir, local_files_only=_lfo,
        )
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            pipe.scheduler.config, use_dynamic_shifting=False, shift_terminal=None,
        )

        # Apply user LoRAs to Stage 1 — must be before enable_group_offload.
        # generate() has no **kw so read directly from the scene.
        from ...utils.helpers import bpy as _bpy
        _lora_folder   = _bpy.path.abspath(getattr(scene, "lora_folder", ""))
        _enabled_loras = [item for item in getattr(scene, "lora_files", []) if item.enabled]
        _lora_names, _lora_weights = [], []
        if _enabled_loras and _lora_folder:
            print(f"LTX-2.3 Stage 1: loading {len(_enabled_loras)} LoRA(s) from {_lora_folder}")
            for _item in _enabled_loras:
                _name = clean_filename(_item.name).replace(".", "")
                try:
                    pipe.load_lora_weights(
                        _lora_folder,
                        weight_name=_item.name + ".safetensors",
                        adapter_name=_name,
                    )
                except Exception as _e:
                    print(f"  LoRA '{_item.name}': load error — {_e}")
                    continue
                # load_lora_weights succeeds silently even when no keys match;
                # check that the adapter was actually registered before using it.
                _loaded = {a for _v in pipe.get_list_adapters().values() for a in _v}
                if _name in _loaded:
                    _lora_names.append(_name)
                    _w = getattr(_item, "weight_value", 1.0)
                    _lora_weights.append(_w)
                    print(f"  LoRA '{_item.name}': loaded (weight={_w})")
                else:
                    print(f"  LoRA '{_item.name}': no matching keys for LTX-2.3, skipped.")
            if _lora_names:
                pipe.set_adapters(_lora_names, adapter_weights=_lora_weights)
                print(f"  Active LoRAs: {_lora_names}")
            else:
                print("  No compatible LoRAs applied.")

        # Parse Audio Conditions
        audio_conditions = None
        # print(f"[DEBUG] Audio VAE check: sound_path={sound_path!r}, "
        #       f"has audio_vae attr={hasattr(pipe, 'audio_vae')}, "
        #       f"audio_vae value={getattr(pipe, 'audio_vae', 'ATTR_MISSING')!r}")
        if sound_path and hasattr(pipe, "audio_vae") and pipe.audio_vae:
            target_sr = pipe.audio_vae.config.sample_rate
            try:
                waveform = load_audio(sound_path, target_sample_rate=target_sr, seconds=dur_s)
                audio_conditions =[LTX2AudioCondition(audio=waveform, strength=1.0)]
                # print(f"[DEBUG] Audio Condition successfully loaded and resampled to {target_sr}Hz. Waveform shape: {waveform.shape}")
            except Exception as e:
                # print(f"[DEBUG] WARNING: Failed to load audio condition in pipeline: {e}")
                import traceback; traceback.print_exc()
        elif sound_path:
            pass  # print(f"[DEBUG] WARNING: Audio path provided but audio_vae is None/missing — audio condition NOT applied!")

        pipe.enable_group_offload(
            onload_device=onload_device,
            offload_type="leaf_level",
            use_stream=False,
            low_cpu_mem_usage=False,
        )

        stage1_kw = dict(
            prompt_embeds=prompt_embeds.to(onload_device, dtype=torch_dtype),
            prompt_attention_mask=prompt_attention_mask.to(onload_device),
            width=stage1_w, height=stage1_h,
            num_frames=num_frames, frame_rate=fps,
            num_inference_steps=8, sigmas=DISTILLED_SIGMA_VALUES,
            guidance_scale=1.0, generator=generator,
            output_type="latent", return_dict=False,
            use_cross_timestep=True, # Critical for Audio+Image cross-attention mapping
            callback_on_step_end=self.step_callback(inputs),
        )
        
        if image_conditions is not None:
            stage1_kw["image_conditions"] = image_conditions
        if audio_conditions is not None:
            stage1_kw["audio_conditions"] = audio_conditions

        # Gentle Guidance overrides. Excludes destructive audio modalities overrides
        if image_conditions is not None and audio_conditions is not None:
            # print("[DEBUG] Image + Audio detected: Applying gentle alignment STG guidance.")
            stage1_kw["stg_scale"] = 1.0
            stage1_kw["spatio_temporal_guidance_blocks"] = [28]
            stage1_kw["guidance_rescale"] = 0.7

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch_dtype):
            outputs = pipe(**stage1_kw)

        # Robust extraction for multimodal pipelines
        if isinstance(outputs, (tuple, list)):
            video_latent = outputs[0].detach().to(offload_device, copy=True)
            audio_latent = outputs[1].detach().to(offload_device, copy=True) if len(outputs) > 1 and outputs[1] is not None else None
        else:
            video_latent = outputs.detach().to(offload_device, copy=True)
            audio_latent = None
            
        # print(f"[DEBUG] Stage 1 Results:")
        # print(f"  -> Video Latent Generated: {'YES' if video_latent is not None else 'NO'}")
        # print(f"  -> Audio Latent Generated: {'YES' if audio_latent is not None else 'NO (Failed or ignored by pipeline)'}")

        del pipe, transformer
        _flush()

        # ── Latent upsampling (2×) ──────────────────────────────────────────
        self.set_phase(inputs, "Stage 1.5: latent upsampling ×2")
        upsampler = LTX2LatentUpsamplerModel.from_pretrained(
            UPSAMPLER_PATH, torch_dtype=torch_dtype, cache_dir=_cache_dir, local_files_only=_lfo,
        ).to(onload_device)
        with torch.inference_mode():
            up_latent = upsampler(video_latent.to(onload_device, dtype=torch_dtype))
        up_latent = up_latent.detach().to(offload_device, copy=True)
        del upsampler, video_latent
        _flush()

        # ── Stage 2: Refinement ─────────────────────────────────────────────
        self.set_phase(inputs, f"Stage 2: refinement {w}×{h}")
        transformer2 = _load_sdnq(
            model_path=_sdnq_transformer_path, model_cls=LTX2VideoTransformer3DModel,
            dtype=torch_dtype, device="cpu",
        )
        refine_pipe = LTX2MultiModalPipeline.from_pretrained(
            MODEL_PATH,
            transformer=transformer2,
            text_encoder=None, tokenizer=None,
            torch_dtype=torch_dtype, cache_dir=_cache_dir, local_files_only=_lfo,
        )
        refine_pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            refine_pipe.scheduler.config, use_dynamic_shifting=False, shift_terminal=None,
        )

        # Apply same user LoRAs to Stage 2 transformer before group offload
        if _enabled_loras and _lora_folder:
            print(f"LTX-2.3 Stage 2: loading {len(_enabled_loras)} LoRA(s)")
            _r_names, _r_weights = [], []
            for _item in _enabled_loras:
                _name = clean_filename(_item.name).replace(".", "")
                try:
                    refine_pipe.load_lora_weights(
                        _lora_folder,
                        weight_name=_item.name + ".safetensors",
                        adapter_name=_name,
                    )
                except Exception as _e:
                    print(f"  LoRA '{_item.name}': load error — {_e}")
                    continue
                _loaded = {a for _v in refine_pipe.get_list_adapters().values() for a in _v}
                if _name in _loaded:
                    _r_names.append(_name)
                    _w = getattr(_item, "weight_value", 1.0)
                    _r_weights.append(_w)
                    print(f"  LoRA '{_item.name}': loaded (weight={_w})")
                else:
                    print(f"  LoRA '{_item.name}': no matching keys for LTX-2.3, skipped.")
            if _r_names:
                refine_pipe.set_adapters(_r_names, adapter_weights=_r_weights)
                print(f"  Active LoRAs: {_r_names}")
            else:
                print("  No compatible LoRAs applied.")

        refine_pipe.enable_group_offload(
            onload_device=onload_device,
            offload_type="leaf_level",
            use_stream=False,
            low_cpu_mem_usage=False,
        )

        refine_kw = dict(
            prompt_embeds=prompt_embeds.to(onload_device, dtype=torch_dtype),
            prompt_attention_mask=prompt_attention_mask.to(onload_device),
            latents=up_latent.to(onload_device, dtype=torch_dtype),
            width=w, height=h, num_frames=num_frames,
            num_inference_steps=3,
            noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
            sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
            guidance_scale=1.0, generator=generator,
            output_type="latent", return_dict=False,
            use_cross_timestep=True,
            callback_on_step_end=self.step_callback(inputs),
        )
        
        if image_conditions is not None:
            refine_kw["image_conditions"] = image_conditions
            
        if audio_conditions is not None:
            refine_kw["audio_conditions"] = audio_conditions
            
        if audio_latent is not None:
            refine_kw["audio_latents"] = audio_latent.to(onload_device, dtype=torch_dtype)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch_dtype):
            outputs2 = refine_pipe(**refine_kw)

        if isinstance(outputs2, (tuple, list)):
            final_v = outputs2[0].detach().to(offload_device, copy=True)
            if len(outputs2) > 1 and outputs2[1] is not None:
                final_a = outputs2[1].detach().to(offload_device, copy=True)
                # print("[DEBUG] Stage 2 Results: Audio Latent was refined.")
            else:
                final_a = audio_latent
                # print("[DEBUG] Stage 2 Results: No refined audio returned. Falling back to Stage 1 audio latent.")
        else:
            final_v = outputs2.detach().to(offload_device, copy=True)
            final_a = audio_latent
            # print("[DEBUG] Stage 2 Results: Pipeline returned singular output. Falling back to Stage 1 audio latent.")
            
        del refine_pipe, transformer2, up_latent, prompt_embeds, prompt_attention_mask
        _flush()

        # ── Decode ──────────────────────────────────────────────────────────
        self.set_phase(inputs, "Decoding")
        decode_pipe = LTX2MultiModalPipeline.from_pretrained(
            MODEL_PATH,
            transformer=None, text_encoder=None, tokenizer=None, scheduler=None,
            torch_dtype=torch_dtype, cache_dir=_cache_dir, local_files_only=_lfo,
        )
        vae = decode_pipe.vae.to(onload_device)
        with torch.inference_mode():
            video = vae_temporal_decode_streaming(vae, final_v.to("cpu"), decode_device=onload_device)
            video = decode_pipe.video_processor.postprocess_video(video, output_type="np")

        audio_out      = None
        audio_sr       = 24000
        
        if final_a is not None and hasattr(decode_pipe, "audio_vae") and decode_pipe.audio_vae:
            audio_vae = decode_pipe.audio_vae.to(onload_device)
            # BigVGAN v2 has 108 sequential convs; bfloat16 accumulation compounds
            # to 40-90% spectral degradation. torch.autocast(float32) does NOT upcast
            # existing bfloat16 tensors — explicit .float() on both weights and mel is required.
            vocoder   = decode_pipe.vocoder.float().to(onload_device)
            audio_sr  = getattr(vocoder.config, "output_sampling_rate", 24000)
            # print(f"[DEBUG] Audio VAE Decode Triggered! Latent shape: {final_a.shape}")
            with torch.inference_mode():
                mel = audio_vae.decode(final_a.to(onload_device, dtype=audio_vae.dtype), return_dict=False)[0]
                # print(f"[DEBUG] Decoded mel shape: {mel.shape}, dtype: {mel.dtype}")
                audio_out = vocoder(mel.float()).cpu()
            # print(f"[DEBUG] Vocoder output shape: {audio_out.shape}, dtype: {audio_out.dtype}")
            # print(f"[DEBUG] Audio amplitude stats: min={audio_out.min():.4f}, max={audio_out.max():.4f}, "
            #       f"mean={audio_out.mean():.4f}, std={audio_out.std():.4f}")
            # _write_audio clips to [-1,1] before int16 conversion; if values are out of range, audio
            # sounds distorted. Normalize if needed.
            peak = audio_out.abs().max()
            if peak > 1.0:
                # print(f"[DEBUG] WARNING: Audio peak {peak:.4f} > 1.0 — normalizing to prevent clipping distortion.")
                audio_out = audio_out / peak
            del audio_vae, vocoder
        else:
            pass
            # _avae = getattr(decode_pipe, "audio_vae", "ATTR_MISSING")
            # print(f"[DEBUG] Audio decode skipped. final_a valid={final_a is not None}, "
            #       f"has audio_vae attr={hasattr(decode_pipe, 'audio_vae')}, "
            #       f"audio_vae value={_avae!r}")

        del decode_pipe, vae, final_v, final_a
        _flush()

        # ── Save ────────────────────────────────────────────────────────────
        self.set_phase(inputs, "Saving")
        dst_path = solve_path(clean_filename(str(seed) + "_" + inputs.prompt[:40]) + ".mp4")

        # When an input audio was provided, mux it directly as the output audio track.
        # The model's audio VAE generates new audio conditioned on the input but does
        # not reproduce speech faithfully; passing the rendered source through preserves it.
        _use_audio    = None
        _use_audio_sr = 24000

        if sound_path:
            try:
                import torchaudio
                _wav, _sr = torchaudio.load(sound_path)   # [C, T]
                _target_n = int(round(dur_s * _sr))
                if _wav.shape[-1] > _target_n:
                    _wav = _wav[..., :_target_n]
                elif _wav.shape[-1] < _target_n:
                    _wav = torch.nn.functional.pad(_wav, (0, _target_n - _wav.shape[-1]))
                # encode_video expects [T, 2] stereo (samples-first, 2-channel)
                _mono = _wav.mean(0).float()               # [T]
                _use_audio    = _mono.unsqueeze(-1).expand(-1, 2).contiguous()  # [T, 2]
                _use_audio_sr = int(_sr)
                print(f"[LTX23Multi] Muxing input audio: {_sr} Hz, "
                      f"{_wav.shape[-1]} samples → {dur_s:.2f}s")
            except Exception as _ae:
                print(f"[LTX23Multi] Input audio mux failed ({_ae}), "
                      f"falling back to model audio.")
                if audio_out is not None:
                    _use_audio    = audio_out[0].float().cpu()
                    _use_audio_sr = audio_sr
        elif audio_out is not None:
            _use_audio    = audio_out[0].float().cpu()
            _use_audio_sr = audio_sr

        if _use_audio is not None:
            encode_video(
                torch.from_numpy((video[0] * 255).round().astype("uint8")),
                fps=fps, audio=_use_audio,
                audio_sample_rate=_use_audio_sr, output_path=dst_path,
            )
        else:
            encode_video(
                torch.from_numpy((video[0] * 255).round().astype("uint8")),
                fps=fps, output_path=dst_path,
            )
            
        print(f"LTX-2.3 saved: {dst_path}")
        return dst_path