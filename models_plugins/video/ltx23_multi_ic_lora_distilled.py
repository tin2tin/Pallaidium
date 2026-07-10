"""LTX-2.3 IC-LoRA Distilled — reference-video/audio style transfer, SDNQ two-stage recipe.

Copy of ltx23_multi_ic_lora.py with the inference converted to the official
diffusers-recipes two-stage SDNQ distilled script
(asomoza/diffusers-recipes models/ltx2_3/scripts/ltx23_two_stages_sdnq_distilled.py):
  - Base pipeline components from OzzyGT/LTX-2.3-Distilled
  - SDNQ transformers loaded via plain from_pretrained (SDNQConfig auto-registration)
    with selectable 4-/8-bit per stage and for the text encoder
  - Selectable upscale factor (2x or 1.5x) with matching upsampler repo
  - audio_guidance_scale / guidance_rescale / audio_guidance_rescale plumbed through
  - Streaming temporal decode with short-latent fast path, or on-GPU tiling decode
  - Decode reuses the Stage-2 pipe's VAE/audio-VAE/vocoder instead of reloading

All original features are kept: stage-mode enum (FULL / STEP1 / STEP2), IC-LoRA
control video/audio, 3DREAL image reference, first/middle/last image anchors,
fresh Stage-2 source-audio conditions (lip-sync fix), identity guidance, STG,
original-audio muxing, phases and step callbacks.
"""

import os
import gc
import ctypes

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, solve_path, clean_filename, load_first_frame

try:
    from ._ltx23_control_shared import ensure_ic_lora, resolve_control_inputs, print_input_summary, align_video_frames
except ImportError:
    from _ltx23_control_shared import ensure_ic_lora, resolve_control_inputs, print_input_summary, align_video_frames


def vae_temporal_decode_streaming(vae, latents_cpu, *, decode_device, temb=None):
    import torch
    tile_latent_min = vae.tile_sample_min_num_frames // vae.temporal_compression_ratio
    if latents_cpu.shape[2] <= tile_latent_min:
        latents = latents_cpu.to(device=decode_device, dtype=vae.dtype, non_blocking=True)
        return vae.decode(latents, temb=temb, return_dict=False)[0].cpu()

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


class LTX2_3MultiICLoRADistilledPlugin(ModelPlugin):
    MODEL_ID     = "LTX-2.3 IC-LoRA Distilled"
    DISPLAY_NAME = "LTX-2.3 IC-LoRA (SDNQ Distilled)"
    MODEL_TYPE   = "video"
    DESCRIPTION  = (
        "IC-LoRA video/audio style transfer on the official two-stage SDNQ distilled recipe "
        "(4-bit transformers, no CFG, 8+3 steps) with selectable stage mode (Step 1 / Step 2 / Full). "
        "Select a MOVIE/SCENE strip as the main input and pick a Ref Strip: a MOVIE → IC-LoRA "
        "control video; an IMAGE → 3DREAL appearance reference (frame 0), with the main input "
        "video driving control_video (matches fal/LTX-2.3-3DREAL: video_url + image_url)."
    )

    INPUTS       = InputSpec.PROMPT | InputSpec.NEG_PROMPT | InputSpec.IMAGE | InputSpec.LORA | InputSpec.AUDIO_REF
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.NEG_PROMPT, UISection.VIDEO_STRIP,
        UISection.RESOLUTION, UISection.FRAMES, UISection.SEED, UISection.LORA,
    ]
    PARAMS            = ParamSpec(steps=8, guidance=1.0)
    REQUIRED_PACKAGES = ["torch", "torchaudio", "soundfile", "av", "diffusers", "transformers", "sdnq"]
    supports_inpaint  = False
    supports_input_downscale = True

    # Recipe configuration (ltx23_two_stages_sdnq_distilled.py) ---------------
    # The recipe uses the unquantized "OzzyGT/LTX-2.3-Distilled" here, but its
    # bf16 connectors are big enough that group offloading's pin_memory()
    # (cudaHostAlloc) fails on Windows once Blender's working set is high —
    # "CUDA error: out of memory" on the first Stage-1 forward. The SDNQ int8
    # shell has quantized connectors and is the exact repo the original
    # ic_lora plugin runs reliably; only the transformer/text-encoder repos
    # below follow the distilled recipe.
    MODEL_PATH           = "OzzyGT/LTX-2.3-sdnq-dynamic-int8"
    SDNQ_4BIT_MODEL_PATH = "OzzyGT/LTX-2.3-Distilled-sdnq-dynamic-int4"
    SDNQ_8BIT_MODEL_PATH = "OzzyGT/LTX-2.3-Distilled-1.1-sdnq-dynamic-int8"
    STAGE_1_SDNQ_BITS      = 8     # 4 or 8
    STAGE_2_SDNQ_BITS      = 8     # 4 or 8
    SDNQ_TEXT_ENCODER_BITS = 4     # None = unquantized from MODEL_PATH, 4 or 8 = SDNQ
    UPSCALE_FACTOR         = 2     # 2 or 1.5
    DECODE_MODE            = "streaming"  # "streaming" = low VRAM, "tiling" = on-GPU spatial tiling
    LOW_CPU_MEM_USAGE      = True

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

    def draw_custom_ui(self, col, context) -> bool:
        scene = context.scene
        # The IC-LoRA ref strip lives in the scene shown in the VSE, which in
        # Blender 5.x can differ from the active scene. List + store it there so
        # the dropdown shows the strips you actually see and the picker agrees.
        vse_scene = getattr(context, "sequencer_scene", None) or context.scene
        row = col.row(align=True)
        row.prop(scene, "ref_audio_path", text="Audio Ref.")
        row.operator("sequencer.open_audio_filebrowser", text="", icon="FILEBROWSER")
        col.prop(scene, "ltx23ic_input_downscale_pct")
        if vse_scene.sequence_editor:
            row = col.row(align=True)
            row.prop_search(
                vse_scene, "ltx23ic_control_strip",
                vse_scene.sequence_editor, "strips",
                text="Ref Strip (vid/img)",
                icon="SEQ_STRIP_META",
            )
            row.operator("sequencer.strip_picker", text="", icon="EYEDROPPER").action = "ltx23ic_control_select"
        else:
            col.prop(vse_scene, "ltx23ic_control_strip", text="IC-LoRA Ref Strip")
        col.prop(scene, "ltx23ic_control_strength")
        col.prop(scene, "ltx23ic_control_downscale")
        col.prop(scene, "ltx23ic_control_audio_str")
        col.prop(scene, "ltx23ic_identity_guidance")
        col.prop(scene, "ltx23m_modality_scale")
        col.prop(scene, "ltx23m_image_strength")
        return False

    def draw_post_seed_ui(self, col, context):
        col.prop(context.scene, "ltx23_stage_mode")

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs) -> str:
        import torch
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch._dynamo.config.disable = True

        from sdnq import SDNQConfig  # noqa: F401 — registers the SDNQ weight loader
        from diffusers import LTX2VideoTransformer3DModel
        from diffusers.pipelines.ltx2.export_utils import encode_video
        from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        from diffusers.video_processor import VideoProcessor
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

        from diffusers.pipelines.ltx2.pipeline_ltx2_condition import retrieve_latents

        from ...utils.helpers import ensure_group_offload_pin_fallback
        ensure_group_offload_pin_fallback()

        _cache_dir     = prefs.hf_cache_dir or None
        _lfo           = prefs.local_files_only

        MODEL_PATH     = self.MODEL_PATH
        _stage1_sdnq_path = self.SDNQ_4BIT_MODEL_PATH if self.STAGE_1_SDNQ_BITS == 4 else self.SDNQ_8BIT_MODEL_PATH
        _stage2_sdnq_path = self.SDNQ_4BIT_MODEL_PATH if self.STAGE_2_SDNQ_BITS == 4 else self.SDNQ_8BIT_MODEL_PATH
        UPSCALE_FACTOR = self.UPSCALE_FACTOR
        UPSAMPLER_PATH = "OzzyGT/LTX-2.3-upsampler-x2" if UPSCALE_FACTOR == 2 else "OzzyGT/LTX-2.3-upsampler-x1.5"

        torch_dtype    = torch.bfloat16
        onload_device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        offload_device = torch.device("cpu")
        fps            = 24.0

        # Stage 1 pipeline args (distilled: fewer steps, no CFG)
        guidance_scale        = 1.0
        audio_guidance_scale  = 1.0
        guidance_rescale      = 0.0
        audio_guidance_rescale = 0.0

        seed = inputs.seed or torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        _stage_mode = getattr(scene, "ltx23_stage_mode", "FULL")

        # ── IC-LoRA control params from scene_proxy ─────────────────────────
        _ctrl_video_path  = getattr(scene, "ltx23ic_control_video_path", "")
        _ctrl_audio_path  = getattr(scene, "ltx23ic_control_audio_path", "")
        _ref_image_path   = getattr(scene, "ltx23ic_ref_image_path",     "")
        _ctrl_strength    = float(getattr(scene, "ltx23ic_control_strength",  1.0))
        _ctrl_downscale   = int(getattr(scene,   "ltx23ic_control_downscale", 1))
        _ctrl_audio_str   = float(getattr(scene, "ltx23ic_control_audio_str", 1.0))
        _identity_guid    = float(getattr(scene, "ltx23ic_identity_guidance",  0.0))
        _modality_scale   = float(getattr(scene, "ltx23m_modality_scale", 1.5))
        _image_strength   = float(getattr(scene, "ltx23m_image_strength", 1.0))

        def _flush():
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # Also release cached page-locked host memory where a binding
                # exists. Group offloading pins large host buffers
                # (cudaHostAlloc); freed ones sit in torch's host-allocator
                # cache, and on a later job pin_memory() can then fail with
                # "CUDA error: out of memory" even though VRAM is free.
                for _n in ("_host_emptyCache", "_cuda_hostEmptyCache"):
                    _fn = getattr(torch._C, _n, None)
                    if _fn is not None:
                        try:
                            _fn()
                        except Exception:
                            pass
                        break
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass

        # Group offloading must run with use_stream=False: streamed mode pins
        # every group into page-locked host RAM (cudaHostAlloc) on each
        # onload, and on Windows with Blender's working set high this fails
        # mid-forward — as "CUDA error: out of memory" or "resource already
        # mapped" — even when a probe pin succeeded moments earlier, and even
        # when failed pins fall back to pageable async copies (those then die
        # at the next kernel launch). Blocking copies are slower but never
        # touch pinned memory.

        # ── Stage Resolution Match Fix ──────────────────────────────────────
        # Stage 1 dims must survive the upscale as multiples of 32: align to 32
        # for 2x, to 64 for 1.5x (64 * 1.5 = 96 = 3 * 32).
        _align = 32 if UPSCALE_FACTOR == 2 else 64
        stage1_w = max(_align, round((inputs.width / UPSCALE_FACTOR) / _align) * _align)
        stage1_h = max(_align, round((inputs.height / UPSCALE_FACTOR) / _align) * _align)
        w = int(round(stage1_w * UPSCALE_FACTOR))
        h = int(round(stage1_h * UPSCALE_FACTOR))

        # ── Resolve Image & Audio Inputs ────────────────────────────────────
        image_input = inputs.image

        vid_path = None
        for attr in ["video_path", "video", "video_ref"]:
            val = getattr(inputs, attr, None)
            if val and isinstance(val, str) and os.path.exists(val):
                vid_path = val
                break

        # 3DREAL mode: the "ref strip" dropdown holds a still image. Use it as
        # the frame-0 appearance reference and let the MAIN input video drive
        # control_video (matches fal/LTX-2.3-3DREAL: video_url + image_url).
        image_input, _video_is_control, control_video_frames = resolve_control_inputs(
            image_input, vid_path, _ref_image_path, _ctrl_video_path, load_first_frame, load_video,
        )
        control_video_frames = align_video_frames(control_video_frames, tag="LTX23ICLoRADistilled")

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
                    if has_video and image_input is None and not _video_is_control:
                        image_input = load_first_frame(vid_path)
                    if has_audio and not explicit_audio:
                        sound_path = vid_path
            except Exception:
                if image_input is None and not _video_is_control:
                    try:
                        image_input = load_first_frame(vid_path)
                    except Exception:
                        pass

        # ── STEP2: validate input video ─────────────────────────────────────
        if _stage_mode == "STEP2" and not vid_path:
            raise RuntimeError(
                "Step 2 mode requires an input video strip. Select a MOVIE strip "
                "and try again."
            )

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
                    print(f"[LTX23ICLoRADistilled] WARN audio={_audio_frames} fr >> strip={num_frames} fr "
                          f"— clamping to strip duration")
                    dur_s = num_frames / fps
                elif _audio_frames < num_frames - 8:
                    print(f"[LTX23ICLoRADistilled] WARN audio={_audio_frames} fr << strip={num_frames} fr "
                          f"— audio shorter than strip, will be padded with silence")
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

        if inputs.frames > 0 and num_frames != inputs.frames:
            print(f"[LTX23ICLoRADistilled] Duration adjusted for 8n+1 alignment: "
                  f"requested {inputs.frames} fr → {num_frames} fr ({num_frames / fps:.1f}s)")
        elif inputs.frames == 0:
            print(f"[LTX23ICLoRADistilled] No strip selected — duration set by audio: "
                  f"{num_frames} fr ({dur_s:.1f}s)")
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
            image_conditions = [LTX2ImageCondition(image=image_input, frame=0, strength=_image_strength)]
            for _mp, _frac in _middle_paths:
                _frame_idx = max(1, min(num_frames - 2, round(_frac * (num_frames - 1))))
                try:
                    _mid_pil = _load_image(_mp).convert("RGB").resize((inputs.width, inputs.height))
                    image_conditions.append(LTX2ImageCondition(image=_mid_pil, frame=_frame_idx, strength=_image_strength))
                except Exception as _e:
                    print(f"[LTX23ICLoRADistilled] WARNING: skipping middle anchor {_mp!r}: {_e}")
            image_conditions.append(LTX2ImageCondition(image=last_input, frame=-1, strength=_image_strength))
        elif image_input is not None and last_input is not None:
            image_conditions = [
                LTX2ImageCondition(image=image_input, frame=0,  strength=_image_strength),
                LTX2ImageCondition(image=last_input,  frame=-1, strength=_image_strength),
            ]
        elif last_input is not None:
            image_conditions = [LTX2ImageCondition(image=last_input, frame=-1, strength=_image_strength)]
        elif image_input is not None:
            image_conditions = [LTX2ImageCondition(image=image_input, frame=0, strength=_image_strength)]

        _ctrl_audio_wave = None

        # ── Step 0: Text encoding ───────────────────────────────────────────
        self.set_phase(inputs, "Text encoding")
        from ...utils.helpers import suppress_text_encoder_warnings
        with suppress_text_encoder_warnings():
            embeds_pipe_kwargs = dict(
                transformer=None, vae=None, audio_vae=None, vocoder=None,
                scheduler=None,
                torch_dtype=torch_dtype, cache_dir=_cache_dir, local_files_only=_lfo,
            )
            if self.SDNQ_TEXT_ENCODER_BITS is not None:
                _sdnq_te_path = self.SDNQ_4BIT_MODEL_PATH if self.SDNQ_TEXT_ENCODER_BITS == 4 else self.SDNQ_8BIT_MODEL_PATH
                embeds_pipe_kwargs["text_encoder"] = Gemma3ForConditionalGeneration.from_pretrained(
                    _sdnq_te_path, subfolder="text_encoder", torch_dtype=torch_dtype,
                    cache_dir=_cache_dir, local_files_only=_lfo,
                )
            embeds_pipe = LTX2MultiModalPipeline.from_pretrained(MODEL_PATH, **embeds_pipe_kwargs)
            embeds_pipe.enable_group_offload(
                onload_device=onload_device,
                offload_type="block_level",
                num_blocks_per_group=4,
                use_stream=False,
                low_cpu_mem_usage=self.LOW_CPU_MEM_USAGE,
                # VAEs stay fully on-GPU (they're small): block_level parks
                # unmatched root layers (encoder.conv_in) behind a hook on the
                # module's own forward(), which never fires because pipelines
                # call vae.encode()/decode() directly — weights stranded on CPU.
                exclude_modules=["vae", "audio_vae"],
            )
            with torch.inference_mode():
                prompt_embeds, prompt_attention_mask, _, _ = embeds_pipe.encode_prompt(
                    prompt=inputs.prompt,
                    negative_prompt=inputs.neg_prompt,
                    do_classifier_free_guidance=False,
                    device=onload_device,
                )
        prompt_embeds         = prompt_embeds.detach().to(offload_device, copy=True)
        prompt_attention_mask = prompt_attention_mask.detach().to(offload_device, copy=True)
        # embeds_pipe_kwargs holds the Gemma3 text encoder — deleting only the
        # pipe would keep the encoder (and its pinned offload buffers) alive
        # for the rest of the job.
        del embeds_pipe, embeds_pipe_kwargs
        _flush()

        # ── Shared: LoRA config ─────────────────────────────────────────────
        from ...utils.helpers import bpy as _bpy
        _lora_folder   = _bpy.path.abspath(getattr(scene, "lora_folder", ""))
        _enabled_loras = [item for item in getattr(scene, "lora_files", []) if item.enabled]

        print_input_summary(
            "LTX23ICLoRADistilled",
            image_input=image_input, vid_path=vid_path, sound_path=sound_path,
            ref_image_path=_ref_image_path, ctrl_video_path=_ctrl_video_path,
            ctrl_audio_path=_ctrl_audio_path, video_is_control=_video_is_control,
            control_video_frames=control_video_frames,
            control_active=bool(control_video_frames is not None or _ctrl_audio_path or _ref_image_path),
            ctrl_strength=_ctrl_strength, ctrl_downscale=_ctrl_downscale,
            ctrl_audio_str=_ctrl_audio_str, identity_guid=_identity_guid,
            stage_mode=_stage_mode, lora_folder=_lora_folder, enabled_loras=_enabled_loras,
            modality_scale=_modality_scale,
        )

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
            print(f"[LTX23ICLoRADistilled] STEP2: encoding {_nf} frames (aligned to {pixels.size(2)})")
            pixels = pixels.to(dtype=vae.dtype, device=onload_device)
            with torch.inference_mode():
                video_latent = retrieve_latents(vae.encode(pixels), generator=generator, sample_mode="argmax")
            _target_latent_f = (num_frames - 1) // 8 + 1
            if video_latent.size(2) > _target_latent_f:
                video_latent = video_latent[:, :, :_target_latent_f, :, :]
            video_latent = video_latent.detach().to(offload_device, copy=True)

            del encode_pipe, vae, pixels, frames
            _flush()
            print(f"[LTX23ICLoRADistilled] STEP2: encoded input video → latent {tuple(video_latent.shape)}")

        # ====================================================================
        # FULL / STEP1: Stage 1 generation
        # ====================================================================
        if _stage_mode != "STEP2":
            self.set_phase(inputs, f"Stage 1: generating {stage1_w}×{stage1_h} ({self.STAGE_1_SDNQ_BITS}-bit)")
            transformer = LTX2VideoTransformer3DModel.from_pretrained(
                _stage1_sdnq_path, subfolder="transformer",
                torch_dtype=torch_dtype, device_map="cpu",
                cache_dir=_cache_dir, local_files_only=_lfo,
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

            if _enabled_loras and _lora_folder:
                print(f"LTX-2.3 IC-LoRA Distilled Stage 1: loading {len(_enabled_loras)} LoRA(s)")
            ensure_ic_lora(pipe, _lora_folder, _enabled_loras, _cache_dir, _lfo)

            if _ctrl_audio_path and os.path.isfile(_ctrl_audio_path):
                if hasattr(pipe, "audio_vae") and pipe.audio_vae:
                    target_sr = pipe.audio_vae.config.sample_rate
                    try:
                        _ctrl_audio_wave = load_audio(_ctrl_audio_path, target_sample_rate=target_sr, seconds=dur_s)
                        print(f"[LTX23ICLoRADistilled] Control audio loaded from {_ctrl_audio_path!r}")
                    except Exception as _e:
                        print(f"[LTX23ICLoRADistilled] WARNING: failed to load control audio ({_e})")

            if sound_path and hasattr(pipe, "audio_vae") and pipe.audio_vae:
                target_sr = pipe.audio_vae.config.sample_rate
                try:
                    waveform = load_audio(sound_path, target_sample_rate=target_sr, seconds=dur_s)
                    # start_time=0.0 always: sound_path is a rendered/trimmed file
                    # (render_meta_child_to_path for META SOUND children) that ALREADY
                    # spans the full output duration with the dialogue silence-padded
                    # to its correct relative offset. Passing that same offset again
                    # here double-applies the shift and truncates/misaligns the tail
                    # of the speech relative to what the video was conditioned on,
                    # even though the final mux (using this file untouched) plays back
                    # at the correct time — showing up as "audio's right, but no lip-sync".
                    audio_conditions = [LTX2AudioCondition(
                        audio=waveform, strength=1.0, start_time=0.0,
                    )]
                    print(f"[LTX23ICLoRADistilled] Stage 1: source audio conditions loaded from {os.path.basename(sound_path)}")
                except Exception:
                    import traceback; traceback.print_exc()

            if sound_path and audio_conditions is None:
                print("[LTX23ICLoRADistilled] WARNING: Stage 1 has NO audio conditions despite a source audio file — lip-sync will fail")

            pipe.enable_group_offload(
                onload_device=onload_device,
                offload_type="block_level",
                num_blocks_per_group=4,
                use_stream=False,
                low_cpu_mem_usage=self.LOW_CPU_MEM_USAGE,
                # VAEs stay fully on-GPU (they're small): block_level parks
                # unmatched root layers (encoder.conv_in) behind a hook on the
                # module's own forward(), which never fires because pipelines
                # call vae.encode()/decode() directly — weights stranded on CPU.
                exclude_modules=["vae", "audio_vae"],
            )

            stage1_kw = dict(
                prompt_embeds=prompt_embeds.to(onload_device, dtype=torch_dtype),
                prompt_attention_mask=prompt_attention_mask.to(onload_device),
                width=stage1_w, height=stage1_h,
                num_frames=num_frames, frame_rate=fps,
                num_inference_steps=len(DISTILLED_SIGMA_VALUES), sigmas=DISTILLED_SIGMA_VALUES,
                guidance_scale=guidance_scale,
                audio_guidance_scale=audio_guidance_scale,
                guidance_rescale=guidance_rescale,
                audio_guidance_rescale=audio_guidance_rescale,
                generator=generator,
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

            if audio_conditions is not None and _modality_scale != 1.0:
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

            del pipe, transformer, stage1_kw, outputs
            _flush()

        # Decode components: filled from the Stage-2 pipe (recipe style) in
        # FULL/STEP2 mode, or from a decode-only pipe load in STEP1 mode.
        vae = None
        audio_vae = None
        vocoder = None
        video_processor = None

        # ====================================================================
        # STEP1: skip upsample + Stage 2, jump straight to decode
        # ====================================================================
        if _stage_mode == "STEP1":
            final_v = video_latent
            final_a = audio_latent
            print(f"[LTX23ICLoRADistilled] STEP1: skipping upsample + Stage 2")

        # ====================================================================
        # FULL / STEP2: Latent upsampling + Stage 2 refinement
        # ====================================================================
        if _stage_mode != "STEP1":
            self.set_phase(inputs, f"Stage 1.5: latent upsampling ×{UPSCALE_FACTOR}")
            upsampler = LTX2LatentUpsamplerModel.from_pretrained(
                UPSAMPLER_PATH, torch_dtype=torch_dtype, cache_dir=_cache_dir, local_files_only=_lfo,
            ).to(onload_device)
            with torch.inference_mode():
                up_latent = upsampler(video_latent.to(onload_device, dtype=torch_dtype))
            up_latent = up_latent.detach().to(offload_device, copy=True)
            del upsampler, video_latent
            _flush()

            self.set_phase(inputs, f"Stage 2: refinement {w}×{h} ({self.STAGE_2_SDNQ_BITS}-bit)")
            transformer2 = LTX2VideoTransformer3DModel.from_pretrained(
                _stage2_sdnq_path, subfolder="transformer",
                torch_dtype=torch_dtype, device_map="cpu",
                cache_dir=_cache_dir, local_files_only=_lfo,
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
                print(f"LTX-2.3 IC-LoRA Distilled Stage 2: loading {len(_enabled_loras)} LoRA(s)")
            ensure_ic_lora(refine_pipe, _lora_folder, _enabled_loras, _cache_dir, _lfo)

            # Re-load the SOURCE input audio here so Stage 2 conditions on a
            # FRESH condition object rather than reusing the Stage-1 one (which
            # was built before `del pipe` / `_flush()` and may reference an
            # offloaded/freed device). This must run in FULL mode too, not just
            # STEP2 — previously it only fired for `_stage_mode == "STEP2"`, so
            # a FULL-mode job silently carried the stale Stage-1 audio_conditions
            # into Stage 2 instead of reloading the source file fresh.
            if _ctrl_audio_wave is None and _ctrl_audio_path and os.path.isfile(_ctrl_audio_path):
                if hasattr(refine_pipe, "audio_vae") and refine_pipe.audio_vae:
                    target_sr = refine_pipe.audio_vae.config.sample_rate
                    try:
                        _ctrl_audio_wave = load_audio(_ctrl_audio_path, target_sample_rate=target_sr, seconds=dur_s)
                        print(f"[LTX23ICLoRADistilled] Stage 2: control audio loaded from {_ctrl_audio_path!r}")
                    except Exception as _e:
                        print(f"[LTX23ICLoRADistilled] Stage 2: WARNING: failed to load control audio ({_e})")

            if sound_path and hasattr(refine_pipe, "audio_vae") and refine_pipe.audio_vae:
                target_sr = refine_pipe.audio_vae.config.sample_rate
                try:
                    waveform = load_audio(sound_path, target_sample_rate=target_sr, seconds=dur_s)
                    # start_time=0.0 always — see the Stage 1 comment above; sound_path
                    # already has its offset baked in via silence padding.
                    audio_conditions = [LTX2AudioCondition(
                        audio=waveform, strength=1.0, start_time=0.0,
                    )]
                    print(f"[LTX23ICLoRADistilled] Stage 2: source audio conditions (re)loaded from {os.path.basename(sound_path)}")
                except Exception:
                    import traceback; traceback.print_exc()

            refine_pipe.enable_group_offload(
                onload_device=onload_device,
                offload_type="block_level",
                num_blocks_per_group=4,
                use_stream=False,
                low_cpu_mem_usage=self.LOW_CPU_MEM_USAGE,
                # VAEs stay fully on-GPU (they're small): block_level parks
                # unmatched root layers (encoder.conv_in) behind a hook on the
                # module's own forward(), which never fires because pipelines
                # call vae.encode()/decode() directly — weights stranded on CPU.
                exclude_modules=["vae", "audio_vae"],
            )

            refine_kw = dict(
                prompt_embeds=prompt_embeds.to(onload_device, dtype=torch_dtype),
                prompt_attention_mask=prompt_attention_mask.to(onload_device),
                latents=up_latent.to(onload_device, dtype=torch_dtype),
                width=w, height=h, num_frames=num_frames, frame_rate=fps,
                num_inference_steps=len(STAGE_2_DISTILLED_SIGMA_VALUES),
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
                if _modality_scale != 1.0:
                    refine_kw["modality_scale"] = _modality_scale
            if audio_conditions is None and audio_latent is not None:
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

            # Keep the video VAE from this pipe (recipe style) instead of
            # reloading a decode-only pipeline afterwards. The audio VAE and
            # vocoder are NOT borrowed: enable_group_offload left hooks on them
            # that keep streaming the original bf16 weights, so the decode
            # block's `vocoder.float()` never takes effect and the float32 mel
            # then crashes with a dtype mismatch. They get a fresh hook-free
            # load in the audio-decode block instead. The video VAE is safe to
            # borrow because its inputs are cast to vae.dtype.
            vae = refine_pipe.vae
            video_processor = refine_pipe.video_processor

            del refine_pipe, transformer2, up_latent, prompt_embeds, prompt_attention_mask, refine_kw, outputs2
            _flush()

        # ── Decode ──────────────────────────────────────────────────────────
        self.set_phase(inputs, f"Decoding ({self.DECODE_MODE})")
        if vae is None:
            # STEP1 mode: no Stage-2 pipe to borrow the VAE from.
            decode_pipe = LTX2MultiModalPipeline.from_pretrained(
                MODEL_PATH,
                transformer=None, text_encoder=None, tokenizer=None, scheduler=None,
                torch_dtype=torch_dtype, cache_dir=_cache_dir, local_files_only=_lfo,
            )
            vae       = decode_pipe.vae
            audio_vae = getattr(decode_pipe, "audio_vae", None)
            vocoder   = getattr(decode_pipe, "vocoder", None)
            video_processor = decode_pipe.video_processor
            del decode_pipe
        if video_processor is None:
            video_processor = VideoProcessor(vae_scale_factor=vae.spatial_compression_ratio)

        vae = vae.to(onload_device)
        with torch.inference_mode():
            if self.DECODE_MODE == "streaming":
                # Streaming temporal decode: one tile at a time on GPU, blend on
                # CPU. No spatial tiling — avoids grid artifacts at high res.
                video = vae_temporal_decode_streaming(vae, final_v.to("cpu"), decode_device=onload_device)
            else:
                # Tiling decode: full video on GPU with spatial+temporal tiling.
                # Higher VRAM but faster. May show grid artifacts at high res.
                vae.enable_tiling()
                video = vae.decode(final_v.to(device=onload_device, dtype=vae.dtype), return_dict=False)[0]
            video = video_processor.postprocess_video(video, output_type="np")

        del vae, final_v
        _flush()

        audio_out = None
        audio_sr  = 24000

        if final_a is not None and (audio_vae is None or vocoder is None):
            # Fresh hook-free load (see the Stage-2 comment above) — only the
            # audio components, everything else excluded.
            audio_pipe = LTX2MultiModalPipeline.from_pretrained(
                MODEL_PATH,
                transformer=None, text_encoder=None, tokenizer=None,
                scheduler=None, vae=None,
                torch_dtype=torch_dtype, cache_dir=_cache_dir, local_files_only=_lfo,
            )
            audio_vae = getattr(audio_pipe, "audio_vae", None)
            vocoder   = getattr(audio_pipe, "vocoder", None)
            del audio_pipe

        if final_a is not None and audio_vae is not None and vocoder is not None:
            audio_vae = audio_vae.to(onload_device)
            vocoder   = vocoder.float().to(onload_device)
            audio_sr  = getattr(vocoder.config, "output_sampling_rate", 24000)
            with torch.inference_mode():
                mel = audio_vae.decode(final_a.to(onload_device, dtype=audio_vae.dtype), return_dict=False)[0]
                audio_out = vocoder(mel.float()).cpu()
            peak = audio_out.abs().max()
            if peak > 1.0:
                audio_out = audio_out / peak
            del mel

        del audio_vae, vocoder, final_a
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
                print(f"[LTX23ICLoRADistilled] Muxing input audio: {_sr} Hz → {dur_s:.2f}s")
            except Exception as _ae:
                print(f"[LTX23ICLoRADistilled] Input audio mux failed ({_ae}), falling back to model audio.")
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

        print(f"LTX-2.3 IC-LoRA Distilled ({_stage_mode}) saved: {dst_path}")
        return dst_path
