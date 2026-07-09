"""Clip extension via LTX-2.3 — with selectable stage mode.

Copy of ltx23_extend.py with a stage-mode enum:
  FULL  — Stage 1 + upsample + Stage 2 (default, identical to the original)
  STEP1 — Stage 1 only → half-res preview clip
  STEP2 — VAE-encode input video → upsample + Stage 2 → full-res refined clip
"""

import os
import gc
import ctypes

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, solve_path, clean_filename, load_first_frame

try:
    from ._ltx23_control_shared import align_video_frames
except ImportError:
    from _ltx23_control_shared import align_video_frames


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


class LTX2_3ExtendStagedPlugin(ModelPlugin):
    MODEL_ID     = "LTX-2.3 Extend Staged"
    DISPLAY_NAME = "LTX-2.3 Extend (Staged)"
    MODEL_TYPE   = "video"
    DESCRIPTION  = "Extend a video clip with LTX-2.3 — with selectable stage mode (Step 1 / Step 2 / Full)"

    INPUTS       = InputSpec.PROMPT | InputSpec.NEG_PROMPT | InputSpec.IMAGE | InputSpec.LORA
    UI_SECTIONS  =[
        UISection.PROMPT, UISection.NEG_PROMPT, UISection.VIDEO_STRIP,
        UISection.RESOLUTION, UISection.SEED, UISection.LORA,
    ]
    PARAMS            = ParamSpec(steps=8, guidance=1.0)
    REQUIRED_PACKAGES =["torch", "torchaudio", "soundfile", "av", "diffusers", "transformers", "sdnq"]
    supports_inpaint  = False
    uses_strip_power  = False

    def draw_custom_ui(self, col, context) -> bool:
        scene = context.scene
        # The audio strip lives in the scene shown in the VSE
        # (context.sequencer_scene in Blender 5.x), which can differ from the
        # active scene. List + store it there so the picker and queue agree.
        vse_scene = getattr(context, "sequencer_scene", None) or context.scene
        if vse_scene.sequence_editor is not None:
            row = col.row(align=True)
            row.prop_search(
                vse_scene, "ltx23ext_audio_strip",
                vse_scene.sequence_editor, "strips",
                text="Audio Strip", icon="SEQ_STRIP_DUPLICATE",
            )
            row.operator("sequencer.strip_picker", text="", icon="EYEDROPPER").action = "ltx23ext_audio_select"
        col.prop(scene, "ltx23ext_extend_frames")
        col.prop(scene, "ltx23ext_video_strength")
        col.prop(scene, "ltx23m_modality_scale")
        return False

    def draw_post_seed_ui(self, col, context):
        col.prop(context.scene, "ltx23_stage_mode")

    # Async (worker-thread) generation re-enabled: the two real render-queue
    # crashes are fixed elsewhere — the audio-mixdown race
    # (run_sound_mixdown_sync in helpers.py) and the second-run CUDA OOM
    # (per-job clear_cuda_cache in queue_ops.py). Off-thread SDNQ loading never
    # reproduced a torch_cpu fault on its own. Running generate() on the worker
    # keeps the UI responsive and lets the download/processing progress bars
    # update live. Flip back to True if the access-violation ever returns.
    requires_main_thread_for_generate = False

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

        from diffusers.pipelines.ltx2.pipeline_ltx2_condition import retrieve_latents

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
        _stage_mode = getattr(scene, "ltx23_stage_mode", "FULL")

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

        vid_path = None
        for attr in["video_path", "video", "video_ref"]:
            val = getattr(inputs, attr, None)
            if val and isinstance(val, str) and os.path.exists(val):
                vid_path = val
                break

        explicit_audio = None
        if _ext_audio_path and isinstance(_ext_audio_path, str) and os.path.exists(_ext_audio_path):
            explicit_audio = _ext_audio_path
            print(f"[LTX23ExtendStaged] Using picked audio strip: {_ext_audio_path!r}")
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
                source_frames = align_video_frames(source_frames, tag="LTX23ExtendStaged")
                src_n = len(source_frames)
                print(f"[LTX23ExtendStaged] Source clip loaded: {src_n} frames from {vid_path!r}")
            except Exception as e:
                print(f"[LTX23ExtendStaged] WARNING: failed to load source video ({e}).")
                source_frames = None
                src_n = 0
            if sound_path is None:
                try:
                    import av
                    with av.open(vid_path) as container:
                        if any(s.type == 'audio' for s in container.streams):
                            sound_path = vid_path
                            print("[LTX23ExtendStaged] Using source clip's embedded audio.")
                except Exception:
                    pass

        if src_n > 0 and src_n / fps > 6.0:
            print(f"[LTX23ExtendStaged] WARNING: source clip is {src_n / fps:.1f}s "
                  f"({src_n} frames) — long sources increase VRAM use and may OOM.")

        # ── STEP2: validate input video ─────────────────────────────────────
        if _stage_mode == "STEP2" and not vid_path:
            raise RuntimeError(
                "Step 2 mode requires an input video strip. Select a MOVIE strip "
                "and try again."
            )

        # ── Frame Count Calculation ─────────────────────────────────────────
        if _stage_mode == "STEP2":
            # In STEP2 we refine the input video as-is, not extend
            target = inputs.frames if inputs.frames > 0 else src_n
            num_frames = max(9, ((target - 1) // 8) * 8 + 1)
            dur_s = num_frames / fps
        else:
            extend_n = max(8, int(_extend_frames))
            if src_n > 0:
                num_frames = max(9, ((src_n + extend_n - 1) // 8) * 8 + 1)
                print(f"[LTX23ExtendStaged] Extending {src_n} src + {extend_n} new "
                      f"→ {num_frames} fr total ({num_frames / fps:.1f}s)")
            else:
                target = inputs.frames if inputs.frames > 0 else int(extend_n)
                num_frames = max(9, ((target - 1) // 8) * 8 + 1)
                print(f"[LTX23ExtendStaged] WARNING: no source clip selected — generating "
                      f"{num_frames} fr ({num_frames / fps:.1f}s) without extension.")
            dur_s = num_frames / fps
        _flush()

        # ── Build video_conditions (source clip locked at index 0) ──────────
        video_conditions = None
        if source_frames and _stage_mode != "STEP2":
            video_conditions = [LTX2VideoCondition(frames=source_frames, index=0, strength=_video_strength)]
            print(f"[LTX23ExtendStaged] video_conditions: {src_n} frames @ index 0, strength={_video_strength}")

        image_conditions = None

        # ── Step 0: Text encoding ───────────────────────────────────────────
        self.set_phase(inputs, "Text encoding")
        from ...utils.helpers import suppress_text_encoder_warnings
        with suppress_text_encoder_warnings():
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

        # ── Shared: SDNQ transformer path ──────────────────────────────────
        import os as _os
        from huggingface_hub import snapshot_download as _snap
        from sdnq.loader import load_sdnq_model as _load_sdnq
        _sdnq_transformer_path = _os.path.join(
            _snap(SDNQ_PATH, cache_dir=_cache_dir, local_files_only=_lfo), "transformer"
        )

        # ── Shared: LoRA config ─────────────────────────────────────────────
        from ...utils.helpers import bpy as _bpy
        _lora_folder   = _bpy.path.abspath(getattr(scene, "lora_folder", ""))
        _enabled_loras = [item for item in getattr(scene, "lora_files", []) if item.enabled]

        audio_latent = None
        audio_conditions = None

        # ====================================================================
        # STEP2: VAE-encode input video instead of running Stage 1
        # ====================================================================
        if _stage_mode == "STEP2":
            self.set_phase(inputs, "Encoding input video")
            encode_pipe = LTX2MultiModalPipeline.from_pretrained(
                MODEL_PATH,
                transformer=None, text_encoder=None, tokenizer=None, scheduler=None,
                torch_dtype=torch_dtype, cache_dir=_cache_dir, local_files_only=_lfo,
            )
            vae = encode_pipe.vae
            try:
                vae.disable_tiling()
            except Exception:
                vae.use_tiling = False
            vae = vae.to(onload_device)

            frames = load_video(vid_path)
            video_proc = encode_pipe.video_processor
            pixels = video_proc.preprocess_video(frames, stage1_h, stage1_w, resize_mode="crop")
            cur_f = pixels.size(2)
            if cur_f < num_frames:
                pad = torch.zeros(pixels.size(0), pixels.size(1), num_frames - cur_f,
                                  pixels.size(3), pixels.size(4), dtype=pixels.dtype)
                pixels = torch.cat([pixels, pad], dim=2)
            elif cur_f > num_frames:
                pixels = pixels[:, :, :num_frames]
            _nf = pixels.size(2)
            _target_8n1 = max(9, ((_nf - 1 + 7) // 8) * 8 + 1)
            if _target_8n1 > _nf:
                pixels = torch.nn.functional.pad(pixels, (0, 0, 0, 0, 0, _target_8n1 - _nf))
            elif _nf > _target_8n1:
                pixels = pixels[:, :, :_target_8n1]
            print(f"[LTX23ExtendStaged] STEP2: encoding {_nf} frames (aligned to {pixels.size(2)})")
            pixels = pixels.to(dtype=vae.dtype, device=onload_device)
            with torch.inference_mode():
                video_latent = retrieve_latents(vae.encode(pixels), generator=generator, sample_mode="argmax")
            _target_latent_f = (num_frames - 1) // 8 + 1
            if video_latent.size(2) > _target_latent_f:
                video_latent = video_latent[:, :, :_target_latent_f, :, :]
            video_latent = video_latent.detach().to(offload_device, copy=True)

            del encode_pipe, vae, pixels, frames
            _flush()
            print(f"[LTX23ExtendStaged] STEP2: encoded input video → latent {tuple(video_latent.shape)}")

        # ====================================================================
        # FULL / STEP1: Stage 1 generation
        # ====================================================================
        if _stage_mode != "STEP2":
            self.set_phase(inputs, f"Stage 1: generating {stage1_w}×{stage1_h}")
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

            try:
                pipe.vae.enable_tiling(
                    tile_sample_min_height=256, tile_sample_min_width=256,
                    tile_sample_min_num_frames=16,
                    tile_sample_stride_height=192, tile_sample_stride_width=192,
                    tile_sample_stride_num_frames=12,
                )
                pipe.vae.tile_sample_min_num_frames = 16
                pipe.vae.tile_sample_stride_num_frames = 12
                pipe.vae.use_framewise_decoding = True
            except Exception as _e:
                print(f"[LTX23ExtendStaged] VAE encode tiling enable failed (Stage 1): {_e}")

            _lora_names, _lora_weights = [], []
            if _enabled_loras and _lora_folder:
                import warnings as _warnings
                print(f"LTX-2.3 Stage 1: loading {len(_enabled_loras)} LoRA(s) from {_lora_folder}")
                for _item in _enabled_loras:
                    _name = clean_filename(_item.name).replace(".", "")
                    try:
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

            if sound_path and hasattr(pipe, "audio_vae") and pipe.audio_vae:
                target_sr = pipe.audio_vae.config.sample_rate
                try:
                    waveform = load_audio(sound_path, target_sample_rate=target_sr, seconds=dur_s)
                    audio_conditions =[LTX2AudioCondition(audio=waveform, strength=1.0)]
                except Exception as e:
                    import traceback; traceback.print_exc()

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

            if (video_conditions is not None or audio_conditions is not None) and _modality_scale != 1.0:
                stage1_kw["modality_scale"] = _modality_scale
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

        # ====================================================================
        # STEP1: skip upsample + Stage 2, jump straight to decode
        # ====================================================================
        if _stage_mode == "STEP1":
            final_v = video_latent
            final_a = audio_latent
            print(f"[LTX23ExtendStaged] STEP1: skipping upsample + Stage 2")

        # ====================================================================
        # FULL / STEP2: Latent upsampling (2×) + Stage 2 refinement
        # ====================================================================
        if _stage_mode != "STEP1":
            self.set_phase(inputs, "Stage 1.5: latent upsampling ×2")
            upsampler = LTX2LatentUpsamplerModel.from_pretrained(
                UPSAMPLER_PATH, torch_dtype=torch_dtype, cache_dir=_cache_dir, local_files_only=_lfo,
            ).to(onload_device)
            with torch.inference_mode():
                up_latent = upsampler(video_latent.to(onload_device, dtype=torch_dtype))
            up_latent = up_latent.detach().to(offload_device, copy=True)
            del upsampler, video_latent
            _flush()

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

            try:
                refine_pipe.vae.enable_tiling(
                    tile_sample_min_height=256, tile_sample_min_width=256,
                    tile_sample_min_num_frames=16,
                    tile_sample_stride_height=192, tile_sample_stride_width=192,
                    tile_sample_stride_num_frames=12,
                )
                refine_pipe.vae.tile_sample_min_num_frames = 16
                refine_pipe.vae.tile_sample_stride_num_frames = 12
                refine_pipe.vae.use_framewise_decoding = True
            except Exception as _e:
                print(f"[LTX23ExtendStaged] VAE encode tiling enable failed (Stage 2): {_e}")

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

            # Re-load the SOURCE input audio here so Stage 2 conditions on a
            # FRESH condition object rather than reusing the Stage-1 one (which
            # was built before `del pipe` / `_flush()` and may reference an
            # offloaded/freed device). Matches the same fix in ltx23_multi.py /
            # ltx23_multi_ic_lora.py — without this Stage 2 silently ran on the
            # stale Stage-1 audio_conditions instead of the source file.
            if sound_path and hasattr(refine_pipe, "audio_vae") and refine_pipe.audio_vae:
                target_sr = refine_pipe.audio_vae.config.sample_rate
                try:
                    waveform = load_audio(sound_path, target_sample_rate=target_sr, seconds=dur_s)
                    audio_conditions = [LTX2AudioCondition(audio=waveform, strength=1.0)]
                    print(f"[LTX23ExtendStaged] Stage 2: source audio conditions (re)loaded from {os.path.basename(sound_path)}")
                except Exception:
                    import traceback; traceback.print_exc()

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

            # In FULL mode, pass video/image/audio conditions to Stage 2
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
                _mono = _wav.mean(0).float()               # [T]
                _use_audio    = _mono.unsqueeze(-1).expand(-1, 2).contiguous()  # [T, 2]
                _use_audio_sr = int(_sr)
                print(f"[LTX23ExtendStaged] Muxing input audio (prefix only, padded): {_sr} Hz → {dur_s:.2f}s")
            except Exception as _ae:
                print(f"[LTX23ExtendStaged] Input audio mux failed ({_ae}).")

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

        print(f"LTX-2.3 Extend Staged ({_stage_mode}) saved: {dst_path}")
        return dst_path
