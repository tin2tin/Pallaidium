"""LTX-2.3 IC-LoRA — reference-video/audio style transfer via IC-LoRA token conditioning.

Workflow:
  Input strip  → META strip containing:
                   1st MOVIE child  = image-condition source (first frame used)
                   2nd MOVIE child  = IC-LoRA control reference video (pre-trimmed at queue time)
                   1st SOUND child  = driving audio condition for inference (audio_conditions)
                   2nd SOUND child  = IC-LoRA control reference audio (control_audio, optional)
  OR single-file:
                 ltx23ic_control_strip scene prop → MOVIE strip used as control_video

IMAGE / AUDIO_REF inputs provide the driving first-frame image and the target audio condition.
A MOVIE with embedded audio can serve as both image source and audio driver (auto-detected via av).

Fallback IC-LoRA: if no LoRA is loaded, Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control is
auto-downloaded so generation still proceeds.
"""

import os
import gc
import ctypes

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, solve_path, clean_filename, load_first_frame


def vae_temporal_decode_streaming(vae, latents_cpu, *, decode_device, temb=None):
    import torch
    tile_latent_min    = vae.tile_sample_min_num_frames // vae.temporal_compression_ratio
    n_latent_frames    = latents_cpu.shape[2]
    n_sample_frames    = (n_latent_frames - 1) * vae.temporal_compression_ratio + 1
    latent_stride      = vae.tile_sample_stride_num_frames // vae.temporal_compression_ratio
    sample_stride      = vae.tile_sample_stride_num_frames
    blend_n            = vae.tile_sample_min_num_frames - vae.tile_sample_stride_num_frames

    result_tiles, prev_tile = [], None
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


_IC_LORA_FALLBACK = "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control"


class LTX2_3MultiICLoRAPlugin(ModelPlugin):
    MODEL_ID     = "LTX-2.3 IC-LoRA"
    DISPLAY_NAME = "Video: LTX-2.3 (IC-LoRA)"
    MODEL_TYPE   = "video"
    DESCRIPTION  = (
        "IC-LoRA video/audio style transfer. Use a META strip whose 1st MOVIE child is the "
        "image-condition source and 2nd MOVIE child is the IC-LoRA control reference. "
        "Available IC-LoRA models: Union-Control (default), HDR, LipDub, Outpaint."
    )

    INPUTS       = InputSpec.PROMPT | InputSpec.NEG_PROMPT | InputSpec.IMAGE | InputSpec.LORA | InputSpec.AUDIO_REF
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.NEG_PROMPT, UISection.VIDEO_STRIP,
        UISection.RESOLUTION, UISection.FRAMES, UISection.SEED, UISection.LORA,
    ]
    PARAMS            = ParamSpec(width=1920, height=896, frames=121, steps=8, guidance=1.0)
    REQUIRED_PACKAGES = ["torch", "torchaudio", "soundfile", "av", "diffusers", "transformers", "sdnq"]
    supports_inpaint  = False

    def load(self, prefs, scene, **kw):
        return {"pipe": None, "refiner": None, "last_model_card": self.MODEL_ID}

    def draw_custom_ui(self, col, context) -> bool:
        scene = context.scene
        if scene.sequence_editor:
            col.prop_search(
                scene, "ltx23ic_control_strip",
                scene.sequence_editor, "strips",
                text="IC-LoRA Ref Strip",
                icon="SEQ_STRIP_META",
            )
        else:
            col.prop(scene, "ltx23ic_control_strip", text="IC-LoRA Ref Strip")
        col.prop(scene, "ltx23ic_control_strength")
        col.prop(scene, "ltx23ic_control_downscale")
        col.prop(scene, "ltx23ic_control_audio_str")
        col.prop(scene, "ltx23ic_identity_guidance")
        return False

    @staticmethod
    def _apply_loras(pipe, lora_folder, enabled_loras):
        names, weights = [], []
        for item in enabled_loras:
            name = clean_filename(item.name).replace(".", "")
            try:
                pipe.load_lora_weights(
                    lora_folder,
                    weight_name=item.name + ".safetensors",
                    adapter_name=name,
                )
            except Exception as e:
                print(f"  LoRA '{item.name}': load error — {e}")
                continue
            loaded = {a for v in pipe.get_list_adapters().values() for a in v}
            if name in loaded:
                names.append(name)
                weights.append(getattr(item, "weight_value", 1.0))
                print(f"  LoRA '{item.name}': loaded (weight={weights[-1]})")
            else:
                print(f"  LoRA '{item.name}': no matching keys for LTX-2.3, skipped.")
        if names:
            pipe.set_adapters(names, adapter_weights=weights)
            print(f"  Active LoRAs: {names}")
        else:
            print("  No compatible LoRAs applied.")
        return names

    @staticmethod
    def _ensure_ic_lora(pipe, lora_folder, enabled_loras, cache_dir, lfo):
        """Load user LoRAs; if none match, auto-download the Union-Control fallback."""
        names = []
        if enabled_loras and lora_folder:
            names = LTX2_3MultiICLoRAPlugin._apply_loras(pipe, lora_folder, enabled_loras)

        if not names:
            loaded = {a for v in pipe.get_list_adapters().values() for a in v}
            if not loaded:
                print(f"[LTX23ICLoRA] No LoRA loaded — auto-loading fallback: {_IC_LORA_FALLBACK}")
                try:
                    pipe.load_lora_weights(
                        _IC_LORA_FALLBACK,
                        cache_dir=cache_dir,
                        local_files_only=lfo,
                    )
                    pipe.set_adapters(["ic_lora_fallback"], adapter_weights=[1.0])
                    print("[LTX23ICLoRA] Fallback IC-LoRA loaded.")
                    names = ["ic_lora_fallback"]
                except Exception as _e:
                    print(f"[LTX23ICLoRA] WARNING: Fallback IC-LoRA load failed ({_e}). Proceeding without IC-LoRA.")
        return names

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs) -> str:
        import torch
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch._dynamo.config.disable = True

        from diffusers import LTX2VideoTransformer3DModel
        from diffusers.pipelines.ltx2.export_utils import encode_video
        from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        from transformers import Gemma3ForConditionalGeneration

        try:
            from ._pipeline_ltx2_multimodal import (
                LTX2MultiModalPipeline, LTX2AudioCondition, LTX2ImageCondition,
                load_audio, load_video,
            )
        except ImportError:
            from _pipeline_ltx2_multimodal import (
                LTX2MultiModalPipeline, LTX2AudioCondition, LTX2ImageCondition,
                load_audio, load_video,
            )

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

        # ── IC-LoRA control params from scene_proxy ─────────────────────────
        _ctrl_video_path  = getattr(scene, "ltx23ic_control_video_path", "")
        _ctrl_audio_path  = getattr(scene, "ltx23ic_control_audio_path", "")
        _ctrl_strength    = float(getattr(scene, "ltx23ic_control_strength",  1.0))
        _ctrl_downscale   = int(getattr(scene,   "ltx23ic_control_downscale", 1))
        _ctrl_audio_str   = float(getattr(scene, "ltx23ic_control_audio_str", 1.0))
        _identity_guid    = float(getattr(scene, "ltx23ic_identity_guidance",  0.0))
        _audio_start_time = float(getattr(scene, "ltx23m_audio_start_time",   0.0))

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

        # ── Resolve Image & Audio Inputs ────────────────────────────────────
        image_input = inputs.image

        vid_path = None
        for attr in ["video_path", "video", "video_ref"]:
            val = getattr(inputs, attr, None)
            if val and isinstance(val, str) and os.path.exists(val):
                vid_path = val
                break

        explicit_audio = None
        for attr in ["audio_path", "audio", "audio_ref", "sound", "sound_path"]:
            val = getattr(inputs, attr, None)
            if val and isinstance(val, str) and os.path.exists(val):
                explicit_audio = val
                break

        sound_path = explicit_audio

        if vid_path:
            try:
                import av
                with av.open(vid_path) as container:
                    has_video = any(s.type == 'video' for s in container.streams)
                    has_audio = any(s.type == 'audio' for s in container.streams)
                    if has_video and image_input is None:
                        image_input = load_first_frame(vid_path)
                    if has_audio and not explicit_audio:
                        sound_path = vid_path
            except Exception:
                if image_input is None:
                    try:
                        image_input = load_first_frame(vid_path)
                    except Exception:
                        pass

        # ── Frame Count Calculation ─────────────────────────────────────────
        if sound_path:
            dur_s = None
            try:
                import soundfile as sf
                info = sf.info(sound_path)
                dur_s = info.frames / info.samplerate
            except Exception:
                try:
                    import av
                    with av.open(sound_path) as container:
                        audio_stream = next((s for s in container.streams if s.type == 'audio'), None)
                        if audio_stream and audio_stream.duration:
                            dur_s = float(audio_stream.duration * audio_stream.time_base)
                except Exception:
                    pass

            if dur_s is None:
                dur_s = inputs.frames / fps

            if inputs.frames > 0:
                num_frames = max(9, ((inputs.frames - 1) // 8) * 8 + 1)
                _audio_frames = int(((dur_s * fps + 7) // 8) * 8) + 1
                if _audio_frames > num_frames + 8:
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

        _flush()

        # ── Parse Image Conditions ──────────────────────────────────────────
        image_conditions = None
        last_input = getattr(inputs, "last_image", None)

        if last_input is not None:
            if isinstance(last_input, str):
                from diffusers.utils import load_image
                last_input = load_image(last_input).convert("RGB")
            elif hasattr(last_input, "convert"):
                last_input = last_input.convert("RGB")

        if image_input is not None:
            if isinstance(image_input, str):
                from diffusers.utils import load_image
                image_input = load_image(image_input).convert("RGB")
            elif hasattr(image_input, "convert"):
                image_input = image_input.convert("RGB")

        _middle_paths = getattr(inputs, "middle_images_paths", [])

        if image_input is not None and last_input is not None and _middle_paths:
            from diffusers.utils import load_image as _load_image
            image_conditions = [LTX2ImageCondition(image=image_input, frame=0, strength=1.0)]
            for _mp, _frac in _middle_paths:
                _frame_idx = max(1, min(num_frames - 2, round(_frac * (num_frames - 1))))
                try:
                    _mid_pil = _load_image(_mp).convert("RGB").resize((inputs.width, inputs.height))
                    image_conditions.append(LTX2ImageCondition(image=_mid_pil, frame=_frame_idx, strength=1.0))
                except Exception as _e:
                    print(f"[LTX23ICLoRA] WARNING: skipping middle anchor {_mp!r}: {_e}")
            image_conditions.append(LTX2ImageCondition(image=last_input, frame=-1, strength=1.0))
        elif image_input is not None and last_input is not None:
            image_conditions = [
                LTX2ImageCondition(image=image_input, frame=0,  strength=1.0),
                LTX2ImageCondition(image=last_input,  frame=-1, strength=1.0),
            ]
        elif last_input is not None:
            image_conditions = [LTX2ImageCondition(image=last_input, frame=-1, strength=1.0)]
        elif image_input is not None:
            image_conditions = [LTX2ImageCondition(image=image_input, frame=0, strength=1.0)]

        # ── Load control reference material ────────────────────────────────
        # Paths were pre-trimmed at queue time via render_meta_child_to_path()
        control_video_frames = None
        if _ctrl_video_path and os.path.isfile(_ctrl_video_path):
            try:
                control_video_frames = load_video(_ctrl_video_path)
                print(f"[LTX23ICLoRA] Control video loaded: {len(control_video_frames)} frames from {_ctrl_video_path!r}")
            except Exception as _e:
                print(f"[LTX23ICLoRA] WARNING: failed to load control video ({_e})")

        # control_audio_wave loaded after pipe is built (needs audio_vae sample_rate)
        _ctrl_audio_wave = None

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
        prompt_embeds         = prompt_embeds.detach().to(offload_device, copy=True)
        prompt_attention_mask = prompt_attention_mask.detach().to(offload_device, copy=True)
        del embeds_pipe, text_encoder
        _flush()

        # ── Stage 1 ─────────────────────────────────────────────────────────
        self.set_phase(inputs, f"Stage 1: generating {stage1_w}×{stage1_h}")
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

        from ...utils.helpers import bpy as _bpy
        _lora_folder   = _bpy.path.abspath(getattr(scene, "lora_folder", ""))
        _enabled_loras = [item for item in getattr(scene, "lora_files", []) if item.enabled]

        if _enabled_loras and _lora_folder:
            print(f"LTX-2.3 IC-LoRA Stage 1: loading {len(_enabled_loras)} LoRA(s)")
        self._ensure_ic_lora(pipe, _lora_folder, _enabled_loras, _cache_dir, _lfo)

        # Load control audio (needs audio_vae sample rate)
        if _ctrl_audio_path and os.path.isfile(_ctrl_audio_path):
            if hasattr(pipe, "audio_vae") and pipe.audio_vae:
                target_sr = pipe.audio_vae.config.sample_rate
                try:
                    _ctrl_audio_wave = load_audio(_ctrl_audio_path, target_sample_rate=target_sr, seconds=dur_s)
                    print(f"[LTX23ICLoRA] Control audio loaded from {_ctrl_audio_path!r}")
                except Exception as _e:
                    print(f"[LTX23ICLoRA] WARNING: failed to load control audio ({_e})")

        # Parse driving audio conditions (target audio)
        audio_conditions = None
        if sound_path and hasattr(pipe, "audio_vae") and pipe.audio_vae:
            target_sr = pipe.audio_vae.config.sample_rate
            try:
                waveform = load_audio(sound_path, target_sample_rate=target_sr, seconds=dur_s)
                audio_conditions = [LTX2AudioCondition(
                    audio=waveform, strength=1.0, start_time=_audio_start_time,
                )]
            except Exception:
                import traceback; traceback.print_exc()

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
            use_cross_timestep=True,
            identity_guidance_scale=_identity_guid,
            callback_on_step_end=self.step_callback(inputs),
        )

        if image_conditions is not None:
            stage1_kw["image_conditions"] = image_conditions
        if audio_conditions is not None:
            stage1_kw["audio_conditions"] = audio_conditions
        if control_video_frames is not None:
            stage1_kw["control_video"]            = control_video_frames
            stage1_kw["control_strength"]         = _ctrl_strength
            stage1_kw["control_downscale_factor"] = _ctrl_downscale
        if _ctrl_audio_wave is not None:
            stage1_kw["control_audio"]         = _ctrl_audio_wave
            stage1_kw["control_audio_strength"] = _ctrl_audio_str

        # Auto-STG when image + audio driving conditions are present
        if image_conditions is not None and audio_conditions is not None:
            stage1_kw["stg_scale"] = 1.0
            stage1_kw["spatio_temporal_guidance_blocks"] = [28]
            stage1_kw["guidance_rescale"] = 0.7

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch_dtype):
            outputs = pipe(**stage1_kw)

        if isinstance(outputs, (tuple, list)):
            video_latent = outputs[0].detach().to(offload_device, copy=True)
            audio_latent = outputs[1].detach().to(offload_device, copy=True) if len(outputs) > 1 and outputs[1] is not None else None
        else:
            video_latent = outputs.detach().to(offload_device, copy=True)
            audio_latent = None

        del pipe, transformer
        _flush()

        # ── Latent upsampling ───────────────────────────────────────────────
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

        if _enabled_loras and _lora_folder:
            print(f"LTX-2.3 IC-LoRA Stage 2: loading {len(_enabled_loras)} LoRA(s)")
        self._ensure_ic_lora(refine_pipe, _lora_folder, _enabled_loras, _cache_dir, _lfo)

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
            identity_guidance_scale=_identity_guid,
            callback_on_step_end=self.step_callback(inputs),
        )

        if image_conditions is not None:
            refine_kw["image_conditions"] = image_conditions
        if audio_conditions is not None:
            refine_kw["audio_conditions"] = audio_conditions
        if audio_latent is not None:
            refine_kw["audio_latents"] = audio_latent.to(onload_device, dtype=torch_dtype)
        if control_video_frames is not None:
            refine_kw["control_video"]            = control_video_frames
            refine_kw["control_strength"]         = _ctrl_strength
            refine_kw["control_downscale_factor"] = _ctrl_downscale
        if _ctrl_audio_wave is not None:
            refine_kw["control_audio"]         = _ctrl_audio_wave
            refine_kw["control_audio_strength"] = _ctrl_audio_str

        if image_conditions is not None and audio_conditions is not None:
            refine_kw["stg_scale"] = 1.0
            refine_kw["spatio_temporal_guidance_blocks"] = [28]
            refine_kw["guidance_rescale"] = 0.7

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch_dtype):
            outputs2 = refine_pipe(**refine_kw)

        if isinstance(outputs2, (tuple, list)):
            final_v = outputs2[0].detach().to(offload_device, copy=True)
            final_a = outputs2[1].detach().to(offload_device, copy=True) if len(outputs2) > 1 and outputs2[1] is not None else audio_latent
        else:
            final_v = outputs2.detach().to(offload_device, copy=True)
            final_a = audio_latent

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

        audio_out = None
        audio_sr  = 24000

        if final_a is not None and hasattr(decode_pipe, "audio_vae") and decode_pipe.audio_vae:
            audio_vae = decode_pipe.audio_vae.to(onload_device)
            vocoder   = decode_pipe.vocoder.float().to(onload_device)
            audio_sr  = getattr(vocoder.config, "output_sampling_rate", 24000)
            with torch.inference_mode():
                mel = audio_vae.decode(final_a.to(onload_device, dtype=audio_vae.dtype), return_dict=False)[0]
                audio_out = vocoder(mel.float()).cpu()
            peak = audio_out.abs().max()
            if peak > 1.0:
                audio_out = audio_out / peak
            del audio_vae, vocoder

        del decode_pipe, vae, final_v, final_a
        _flush()

        # ── Save ────────────────────────────────────────────────────────────
        self.set_phase(inputs, "Saving")
        dst_path = solve_path(clean_filename(str(seed) + "_" + inputs.prompt[:40]) + ".mp4")

        _use_audio    = None
        _use_audio_sr = 24000

        if sound_path:
            try:
                import torchaudio
                _wav, _sr = torchaudio.load(sound_path)
                _target_n = int(round(dur_s * _sr))
                if _wav.shape[-1] > _target_n:
                    _wav = _wav[..., :_target_n]
                elif _wav.shape[-1] < _target_n:
                    _wav = torch.nn.functional.pad(_wav, (0, _target_n - _wav.shape[-1]))
                _mono = _wav.mean(0).float()
                _use_audio    = _mono.unsqueeze(-1).expand(-1, 2).contiguous()
                _use_audio_sr = int(_sr)
                print(f"[LTX23ICLoRA] Muxing input audio: {_sr} Hz → {dur_s:.2f}s")
            except Exception as _ae:
                print(f"[LTX23ICLoRA] Input audio mux failed ({_ae}), falling back to model audio.")
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

        print(f"LTX-2.3 IC-LoRA saved: {dst_path}")
        return dst_path
