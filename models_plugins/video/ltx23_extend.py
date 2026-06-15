"""Clip extension via LTX-2.3 (OzzyGT/LTX-2.3-Distilled, SDNQ) — lock a source clip and generate a continuation.

Adapted from ltx23_multi.py. Instead of using only the first frame of the input
video strip as an image condition, the whole source clip is carried over as a hard
``LTX2VideoCondition`` locked at latent index 0, and the total ``num_frames`` is
extended by "Extend (s)" so the model paints the new tail after the source.

Audio: the source clip's audio (or a separately-picked SOUND strip) is locked from
t=0 via ``audio_conditions``; the model generates the continuation audio.
"""

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


class LTX2_3ExtendPlugin(ModelPlugin):
    MODEL_ID     = "LTX-2.3 Extend"
    DISPLAY_NAME = "Video: LTX-2.3 (Extend)"
    MODEL_TYPE   = "video"
    DESCRIPTION  = "Extend a video clip with LTX-2.3 (SDNQ) — locks the source clip and generates a continuation with audio"

    # Audio comes from the picked SOUND strip (ltx23ext_audio_strip) or the source
    # clip's embedded track — not from the general ref_audio_path (AUDIO_REF).
    INPUTS       = InputSpec.PROMPT | InputSpec.NEG_PROMPT | InputSpec.IMAGE | InputSpec.LORA
    # No UISection.FRAMES — total length is source frames + "Extend (frames)" (ltx23ext_extend_frames),
    # so the standard generate_movie_frames control is unused and hidden.
    UI_SECTIONS  =[
        UISection.PROMPT, UISection.NEG_PROMPT, UISection.VIDEO_STRIP,
        UISection.RESOLUTION, UISection.SEED, UISection.LORA, #UISection.STEPS, UISection.GUIDANCE,
    ]
    # Standard ParamSpec without unsupported UI fields
    PARAMS            = ParamSpec(width=1280, height=704, frames=121, steps=8, guidance=1.0)
    REQUIRED_PACKAGES =["torch", "torchaudio", "soundfile", "av", "diffusers", "transformers", "sdnq"]
    supports_inpaint  = False
    uses_strip_power  = False  # "Strip Power" (image_power) has no effect in Extend mode — hide it

    def draw_custom_ui(self, col, context) -> bool:
        scene = context.scene
        # SOUND-strip picker (prop_search dropdown + eyedropper) — drives the
        # extended video's audio, overriding the source clip's embedded track.
        if scene.sequence_editor is not None:
            row = col.row(align=True)
            row.prop_search(
                scene, "ltx23ext_audio_strip",
                scene.sequence_editor, "strips",
                text="Audio Strip", icon="SEQ_STRIP_DUPLICATE",
            )
            row.operator("sequencer.strip_picker", text="", icon="EYEDROPPER").action = "ltx23ext_audio_select"
        # Extension controls.
        col.prop(scene, "ltx23ext_extend_frames")
        col.prop(scene, "ltx23ext_video_strength")
        return False

    def load(self, prefs, scene, **kw):
        return {"pipe": None, "refiner": None, "last_model_card": self.MODEL_ID}

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
                LTX2VideoCondition, load_audio, load_video,
            )
        except ImportError:
            from _pipeline_ltx2_multimodal import (
                LTX2MultiModalPipeline, LTX2AudioCondition, LTX2ImageCondition,
                LTX2VideoCondition, load_audio, load_video,
            )

        _cache_dir     = prefs.hf_cache_dir or None
        _lfo           = prefs.local_files_only
        MODEL_PATH     = "OzzyGT/LTX-2.3-Distilled-1.1-sdnq-dynamic-int8"
        SDNQ_PATH      = "OzzyGT/LTX-2.3-Distilled-1.1-sdnq-dynamic-int8"
        UPSAMPLER_PATH = "OzzyGT/LTX-2.3-upsampler-x2"

        torch_dtype    = torch.bfloat16
        onload_device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        offload_device = torch.device("cpu")
        fps            = 24.0

        seed = inputs.seed or torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        _modality_scale = getattr(scene, "ltx23m_modality_scale", 1.5)

        # ── Extension params ────────────────────────────────────────────────
        _extend_frames  = int(getattr(scene, "ltx23ext_extend_frames", 96))
        _video_strength = float(getattr(scene, "ltx23ext_video_strength", 1.0))
        _ext_audio_path = getattr(scene, "ltx23ext_audio_path", "")

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
        # Extend: a picked SOUND strip (ltx23ext_audio_strip → path) takes top priority.
        if _ext_audio_path and isinstance(_ext_audio_path, str) and os.path.exists(_ext_audio_path):
            explicit_audio = _ext_audio_path
            print(f"[LTX23Extend] Using picked audio strip: {_ext_audio_path!r}")
        # NOTE: ref_audio_path / inputs.audio_ref is intentionally NOT consulted here —
        # audio comes only from the picked SOUND strip or the source clip's embedded track.
        if explicit_audio is None:
            for attr in["audio_path", "audio", "sound", "sound_path"]:
                val = getattr(inputs, attr, None)
                if val and isinstance(val, str) and os.path.exists(val):
                    explicit_audio = val
                    break

        sound_path = explicit_audio

        # ── Load the source clip frames (carried over, locked at index 0) ────
        source_frames = None
        src_n = 0
        if vid_path:
            try:
                source_frames = load_video(vid_path)
                src_n = len(source_frames)
                print(f"[LTX23Extend] Source clip loaded: {src_n} frames from {vid_path!r}")
            except Exception as e:
                print(f"[LTX23Extend] WARNING: failed to load source video ({e}).")
                source_frames = None
                src_n = 0
            # If the source clip carries audio and no explicit audio was supplied,
            # use the source clip's embedded track for the locked prefix.
            if sound_path is None:
                try:
                    import av
                    with av.open(vid_path) as container:
                        if any(s.type == 'audio' for s in container.streams):
                            sound_path = vid_path
                            print("[LTX23Extend] Using source clip's embedded audio.")
                except Exception:
                    pass

        if src_n > 0 and src_n / fps > 6.0:
            print(f"[LTX23Extend] WARNING: source clip is {src_n / fps:.1f}s "
                  f"({src_n} frames) — long sources increase VRAM use and may OOM.")

        # ── Frame Count Calculation ─────────────────────────────────────────
        # Extend: total output = source frames + extension, aligned to 8n+1.
        extend_n = max(8, int(_extend_frames))
        if src_n > 0:
            num_frames = max(9, ((src_n + extend_n - 1) // 8) * 8 + 1)
            print(f"[LTX23Extend] Extending {src_n} src + {extend_n} new "
                  f"→ {num_frames} fr total ({num_frames / fps:.1f}s)")
        else:
            # No source clip — degrade to plain text/image-to-video using the strip length.
            target = inputs.frames if inputs.frames > 0 else int(extend_n)
            num_frames = max(9, ((target - 1) // 8) * 8 + 1)
            print(f"[LTX23Extend] WARNING: no source clip selected — generating "
                  f"{num_frames} fr ({num_frames / fps:.1f}s) without extension.")

        # dur_s is the FULL output duration; audio is loaded/trimmed to this length.
        dur_s = num_frames / fps

        # Extend: disabled — the multimodal audio-duration-authoritative branch
        # assumed output == strip length, which no longer holds when extending.
        # dur_s is fixed above from num_frames; we only probe audio for logging.
        # if sound_path:
        #     ... (original soundfile/av duration probe + inputs.frames clamp logic) ...
        _flush()

        # ── Build video_conditions (source clip locked at index 0) ──────────
        video_conditions = None
        if source_frames:
            video_conditions = [LTX2VideoCondition(frames=source_frames, index=0, strength=_video_strength)]
            print(f"[LTX23Extend] video_conditions: {src_n} frames @ index 0, strength={_video_strength}")

        # Extend: disabled — image-condition modes (FLF / multi-anchor / single
        # first-frame) are replaced by the locked source clip (video_conditions).
        # Kept for reference so they can be revived if needed.
        image_conditions = None
        # last_input = getattr(inputs, "last_image", None)
        # if last_input is not None:
        #     if isinstance(last_input, str):
        #         from diffusers.utils import load_image
        #         last_input = load_image(last_input).convert("RGB")
        #     elif hasattr(last_input, "convert"):
        #         last_input = last_input.convert("RGB")
        # if image_input is not None:
        #     if isinstance(image_input, str):
        #         from diffusers.utils import load_image
        #         image_input = load_image(image_input).convert("RGB")
        #     elif hasattr(image_input, "convert"):
        #         image_input = image_input.convert("RGB")
        # _middle_paths = getattr(inputs, "middle_images_paths", [])
        # if image_input is not None and last_input is not None and _middle_paths:
        #     # Mode MA: multi-anchor
        #     from diffusers.utils import load_image as _load_image
        #     image_conditions = [LTX2ImageCondition(image=image_input, frame=0, strength=1.0)]
        #     for _mp, _frac in _middle_paths:
        #         _frame_idx = round(_frac * (num_frames - 1))
        #         _frame_idx = max(1, min(num_frames - 2, _frame_idx))
        #         try:
        #             _mid_pil = _load_image(_mp).convert("RGB").resize((inputs.width, inputs.height))
        #             image_conditions.append(LTX2ImageCondition(image=_mid_pil, frame=_frame_idx, strength=1.0))
        #         except Exception as _e:
        #             print(f"[LTX23Extend] WARNING: skipping middle anchor {_mp!r}: {_e}")
        #     image_conditions.append(LTX2ImageCondition(image=last_input, frame=-1, strength=1.0))
        # elif image_input is not None and last_input is not None:
        #     # Mode A: FLF
        #     image_conditions = [
        #         LTX2ImageCondition(image=image_input, frame=0,  strength=1.0),
        #         LTX2ImageCondition(image=last_input,  frame=-1, strength=1.0),
        #     ]
        # elif last_input is not None:
        #     # Mode B: last-frame-only
        #     image_conditions = [LTX2ImageCondition(image=last_input, frame=-1, strength=1.0)]
        # elif image_input is not None:
        #     # Single first-frame condition
        #     image_conditions = [LTX2ImageCondition(image=image_input, frame=0, strength=1.0)]

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
        # Stream Gemma3 leaf-by-leaf instead of residing the whole encoder on GPU.
        # On low-VRAM cards (≤10 GB) a full .to(onload_device) makes text encoding the
        # peak-memory step and OOMs at torch.stack(hidden_states) — especially on a
        # second job where prior fragmentation leaves less headroom.
        embeds_pipe.enable_group_offload(
            onload_device=onload_device,
            offload_type="leaf_level",
            use_stream=True,
            low_cpu_mem_usage=True,
        )
        with torch.inference_mode():
            prompt_embeds, prompt_attention_mask, _, _ = embeds_pipe.encode_prompt(
                prompt=inputs.prompt,
                negative_prompt=inputs.neg_prompt,
                do_classifier_free_guidance=False,
                device=onload_device,
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

        # The source-clip video_conditions are VAE-encoded inside prepare_latents.
        # For a multi-frame clip at full resolution that single conv3d can need tens
        # of GB and OOMs (esp. in Stage 2 at 2× res). Tile the VAE encode spatially
        # and temporally so it stays within VRAM. Safe here: Stage 1/2 use
        # output_type="latent" and never decode through this VAE.
        try:
            pipe.vae.enable_tiling(
                tile_sample_min_height=256, tile_sample_min_width=256,
                tile_sample_min_num_frames=16,
                tile_sample_stride_height=192, tile_sample_stride_width=192,
                tile_sample_stride_num_frames=12,
            )
            pipe.vae.use_framewise_decoding = True
        except Exception as _e:
            print(f"[LTX23Extend] VAE encode tiling enable failed (Stage 1): {_e}")

        # Apply user LoRAs to Stage 1 — must be before enable_group_offload.
        # generate() has no **kw so read directly from the scene.
        from ...utils.helpers import bpy as _bpy
        _lora_folder   = _bpy.path.abspath(getattr(scene, "lora_folder", ""))
        _enabled_loras = [item for item in getattr(scene, "lora_files", []) if item.enabled]
        _lora_names, _lora_weights = [], []
        if _enabled_loras and _lora_folder:
            import warnings as _warnings
            print(f"LTX-2.3 Stage 1: loading {len(_enabled_loras)} LoRA(s) from {_lora_folder}")
            for _item in _enabled_loras:
                _name = clean_filename(_item.name).replace(".", "")
                try:
                    # peft warns "Already found a `peft_config` attribute" every time a
                    # 2nd adapter is added — benign here, each LoRA gets a distinct
                    # adapter_name and set_adapters() blends them by weight below.
                    with _warnings.catch_warnings():
                        _warnings.filterwarnings(
                            "ignore",
                            message="Already found a `peft_config` attribute",
                        )
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
        if sound_path and hasattr(pipe, "audio_vae") and pipe.audio_vae:
            target_sr = pipe.audio_vae.config.sample_rate
            try:
                waveform = load_audio(sound_path, target_sample_rate=target_sr, seconds=dur_s)
                audio_conditions =[LTX2AudioCondition(audio=waveform, strength=1.0)]
            except Exception as e:
                import traceback; traceback.print_exc()
        elif sound_path:
            pass

        pipe.enable_group_offload(
            onload_device=onload_device,
            offload_type="leaf_level",
            use_stream=True,
            low_cpu_mem_usage=True,
        )

        stage1_kw = dict(
            prompt_embeds=prompt_embeds.to(onload_device, dtype=torch_dtype),
            prompt_attention_mask=prompt_attention_mask.to(onload_device),
            width=stage1_w, height=stage1_h,
            num_frames=num_frames, frame_rate=fps,
            num_inference_steps=8, sigmas=DISTILLED_SIGMA_VALUES,
            guidance_scale=1.0, generator=generator,
            control_downscale_factor=1, control_strength=1.0,
            output_type="latent", return_dict=False,
            use_cross_timestep=True,
            callback_on_step_end=self.step_callback(inputs),
        )

        if video_conditions is not None:
            stage1_kw["video_conditions"] = video_conditions
        if image_conditions is not None:
            stage1_kw["image_conditions"] = image_conditions
        if audio_conditions is not None:
            stage1_kw["audio_conditions"] = audio_conditions

        # With guidance_scale=1.0 (distilled), audio_cfg_delta=0 always.
        # modality_scale>1 triggers do_modality_isolation_guidance — the lever that
        # amplifies the locked clip's / audio's effect at distilled scale.
        if (video_conditions is not None or audio_conditions is not None) and _modality_scale != 1.0:
            stage1_kw["modality_scale"] = _modality_scale
        if image_conditions is not None and audio_conditions is not None:
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

        # Tile the VAE encode of the source-clip video_conditions (full-res here → the
        # step that just OOM'd). See Stage 1 note above.
        try:
            refine_pipe.vae.enable_tiling(
                tile_sample_min_height=256, tile_sample_min_width=256,
                tile_sample_min_num_frames=16,
                tile_sample_stride_height=192, tile_sample_stride_width=192,
                tile_sample_stride_num_frames=12,
            )
            refine_pipe.vae.use_framewise_decoding = True
        except Exception as _e:
            print(f"[LTX23Extend] VAE encode tiling enable failed (Stage 2): {_e}")

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
            use_stream=True,
            low_cpu_mem_usage=True,
        )

        refine_kw = dict(
            prompt_embeds=prompt_embeds.to(onload_device, dtype=torch_dtype),
            prompt_attention_mask=prompt_attention_mask.to(onload_device),
            latents=up_latent.to(onload_device, dtype=torch_dtype),
            width=w, height=h, num_frames=num_frames, frame_rate=fps,
            num_inference_steps=3,
            noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
            sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
            guidance_scale=1.0, generator=generator,
            output_type="latent", return_dict=False,
            use_cross_timestep=True,
            callback_on_step_end=self.step_callback(inputs),
        )

        if video_conditions is not None:
            refine_kw["video_conditions"] = video_conditions
        if image_conditions is not None:
            refine_kw["image_conditions"] = image_conditions

        if audio_conditions is not None:
            refine_kw["audio_conditions"] = audio_conditions
            if _modality_scale != 1.0:
                refine_kw["modality_scale"] = _modality_scale
        elif video_conditions is not None and _modality_scale != 1.0:
            refine_kw["modality_scale"] = _modality_scale

        # NOTE: no STG / guidance_rescale in Stage 2. Use pipeline defaults.

        # Audio into Stage 2:
        #  - With input audio (audio_conditions set): re-lock the SOURCE audio via
        #    audio_conditions (already added above) and do NOT pass audio_latents —
        #    passing pre-encoded latents makes the pipeline ignore audio_conditions, so
        #    Stage 2 would drift off the real speech and worsen lip sync.
        #  - Without input audio: pass the Stage-1 generated audio latent so Stage 2
        #    stays coherent with the audio the video was generated against.
        if audio_conditions is None and audio_latent is not None:
            refine_kw["audio_latents"] = audio_latent.to(onload_device, dtype=torch_dtype)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch_dtype):
            outputs2 = refine_pipe(**refine_kw)

        if isinstance(outputs2, (tuple, list)):
            final_v = outputs2[0].detach().to(offload_device, copy=True)
            if len(outputs2) > 1 and outputs2[1] is not None:
                final_a = outputs2[1].detach().to(offload_device, copy=True)
            else:
                final_a = audio_latent
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

        audio_out      = None
        audio_sr       = 24000

        if final_a is not None and hasattr(decode_pipe, "audio_vae") and decode_pipe.audio_vae:
            audio_vae = decode_pipe.audio_vae.to(onload_device)
            # BigVGAN v2 has 108 sequential convs; bfloat16 accumulation compounds
            # to 40-90% spectral degradation. torch.autocast(float32) does NOT upcast
            # existing bfloat16 tensors — explicit .float() on both weights and mel is required.
            vocoder   = decode_pipe.vocoder.float().to(onload_device)
            audio_sr  = getattr(vocoder.config, "output_sampling_rate", 24000)
            with torch.inference_mode():
                mel = audio_vae.decode(final_a.to(onload_device, dtype=audio_vae.dtype), return_dict=False)[0]
                audio_out = vocoder(mel.float()).cpu()
            # _write_audio clips to [-1,1] before int16 conversion; if values are out of range, audio
            # sounds distorted. Normalize if needed.
            peak = audio_out.abs().max()
            if peak > 1.0:
                audio_out = audio_out / peak
            del audio_vae, vocoder
        else:
            pass

        del decode_pipe, vae, final_v, final_a
        _flush()

        # ── Save ────────────────────────────────────────────────────────────
        self.set_phase(inputs, "Saving")
        dst_path = solve_path(clean_filename(str(seed) + "_" + inputs.prompt[:40]) + ".mp4")

        # Extend: the input audio (source clip or picked strip) only covers the
        # LOCKED prefix, not the generated tail. The model-generated audio_out spans
        # the FULL output duration and stays coherent with the continuation, so prefer
        # it for the muxed track. If the model produced no audio, fall back to muxing
        # the (shorter, silence-padded) input audio.
        _use_audio    = None
        _use_audio_sr = 24000

        if audio_out is not None:
            _use_audio    = audio_out[0].float().cpu()
            _use_audio_sr = audio_sr
        elif sound_path:
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
                print(f"[LTX23Extend] Muxing input audio (prefix only, padded): {_sr} Hz → {dur_s:.2f}s")
            except Exception as _ae:
                print(f"[LTX23Extend] Input audio mux failed ({_ae}).")

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

        print(f"LTX-2.3 Extend saved: {dst_path}")
        return dst_path
