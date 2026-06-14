"""Single-stage image+audio-to-video via LTX-2.3 (OzzyGT/LTX-2.3-Distilled-1.1-sdnq-dynamic-int4)."""

import os
import gc
import ctypes

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, solve_path, clean_filename, load_first_frame


def vae_temporal_decode_streaming(vae, latents_cpu, *, decode_device, temb=None):
    import torch
    """Streaming temporal decode — faster than spatial tiling on ≥16 GB cards."""
    tile_latent_min = vae.tile_sample_min_num_frames // vae.temporal_compression_ratio
    n_latent_frames = latents_cpu.shape[2]
    n_sample_frames = (n_latent_frames - 1) * vae.temporal_compression_ratio + 1
    latent_stride   = vae.tile_sample_stride_num_frames // vae.temporal_compression_ratio
    sample_stride   = vae.tile_sample_stride_num_frames
    blend_n         = vae.tile_sample_min_num_frames - vae.tile_sample_stride_num_frames

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


class LTX2_3LipSyncPlugin(ModelPlugin):
    MODEL_ID     = "LTX-2.3 Lip Sync"
    DISPLAY_NAME = "Video: LTX-2.3 (Lip Sync)"
    MODEL_TYPE   = "video"
    DESCRIPTION  = "Single-stage LTX-2.3 (SDNQ 4-bit) — image+audio-to-video lip sync with audio output"

    INPUTS       = InputSpec.PROMPT | InputSpec.NEG_PROMPT | InputSpec.IMAGE | InputSpec.LORA | InputSpec.AUDIO_REF
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.NEG_PROMPT, UISection.VIDEO_STRIP,
        UISection.RESOLUTION, UISection.FRAMES, UISection.SEED, UISection.LORA,
    ]
    PARAMS            = ParamSpec(width=1280, height=704, frames=265, steps=8, guidance=1.0)
    REQUIRED_PACKAGES = ["torch", "torchaudio", "soundfile", "av", "diffusers", "transformers", "sdnq"]
    supports_inpaint  = False

    def draw_custom_ui(self, _col, _context) -> bool:
        return False

    def load(self, prefs, scene, **kw):
        return {"pipe": None, "refiner": None, "last_model_card": self.MODEL_ID}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs) -> str:
        import torch
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch._dynamo.config.disable = True

        from diffusers import LTX2VideoTransformer3DModel
        from diffusers.pipelines.ltx2.export_utils import encode_video
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
        from sdnq import SDNQConfig  # noqa: F401 — registers SDNQ weight loader

        try:
            from ._pipeline_ltx2_multimodal import LTX2MultiModalPipeline, LTX2AudioCondition, LTX2ImageCondition, load_audio
        except ImportError:
            from _pipeline_ltx2_multimodal import LTX2MultiModalPipeline, LTX2AudioCondition, LTX2ImageCondition, load_audio

        _cache_dir    = prefs.hf_cache_dir or None
        _lfo          = prefs.local_files_only
        MODEL_PATH    = "OzzyGT/LTX-2.3-Distilled-1.1-sdnq-dynamic-int4"

        torch_dtype   = torch.bfloat16
        onload_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        offload_device = torch.device("cpu")
        fps           = 24.0

        seed = inputs.seed or torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        _modality_scale = getattr(scene, "ltx23m_modality_scale", 1.5)

        def _flush():
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass

        w = max(32, round(inputs.width / 32) * 32)
        h = max(32, round(inputs.height / 32) * 32)

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
                    print(f"[LTX23LipSync] WARN audio={_audio_frames} fr >> strip={num_frames} fr "
                          f"(likely untrimmed MOVIE in META) — clamping to strip duration")
                    dur_s = num_frames / fps
                elif _audio_frames < num_frames - 8:
                    print(f"[LTX23LipSync] WARN audio={_audio_frames} fr << strip={num_frames} fr "
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
            print(f"[LTX23LipSync] Duration adjusted for 8n+1 alignment: "
                  f"requested {inputs.frames} fr → {num_frames} fr ({num_frames / fps:.1f}s)")
        elif inputs.frames == 0:
            print(f"[LTX23LipSync] No strip selected — duration set by audio: "
                  f"{num_frames} fr ({dur_s:.1f}s)")
        _flush()

        # ── Image Conditions ────────────────────────────────────────────────
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
                    _mid_pil = _load_image(_mp).convert("RGB").resize((w, h))
                    image_conditions.append(LTX2ImageCondition(image=_mid_pil, frame=_frame_idx, strength=1.0))
                except Exception as _e:
                    print(f"[LTX23LipSync] WARNING: skipping middle anchor {_mp!r}: {_e}")
            image_conditions.append(LTX2ImageCondition(image=last_input, frame=-1, strength=1.0))
            _anchor_frames = [0] + [c.frame for c in image_conditions[1:-1]] + [num_frames - 1]
            print(f"[LTX23LipSync] MODE: MULTI-ANCHOR — {len(image_conditions)} anchors at frames "
                  f"{_anchor_frames} (of {num_frames})"
                  + (f" [requested {inputs.frames}, adjusted for 8n+1]" if num_frames != inputs.frames else ""))
        elif image_input is not None and last_input is not None:
            image_conditions = [
                LTX2ImageCondition(image=image_input, frame=0,  strength=1.0),
                LTX2ImageCondition(image=last_input,  frame=-1, strength=1.0),
            ]
        elif last_input is not None:
            image_conditions = [LTX2ImageCondition(image=last_input, frame=-1, strength=1.0)]
        elif image_input is not None:
            image_conditions = [LTX2ImageCondition(image=image_input, frame=0, strength=1.0)]

        # ── Step 0: Text encoding (encode then unload — keeps vision tower off the inference graph) ──
        self.set_phase(inputs, "Text encoding")
        from transformers import Gemma3ForConditionalGeneration
        text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
            MODEL_PATH, subfolder="text_encoder",
            torch_dtype=torch_dtype, cache_dir=_cache_dir, local_files_only=_lfo,
        )
        embeds_pipe = LTX2MultiModalPipeline.from_pretrained(
            MODEL_PATH,
            text_encoder=text_encoder,
            transformer=None, vae=None, audio_vae=None, vocoder=None, scheduler=None,
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

        # ── Load inference pipeline (no text encoder → no vision tower for offload tracing) ──
        self.set_phase(inputs, "Loading model")
        transformer = LTX2VideoTransformer3DModel.from_pretrained(
            MODEL_PATH, subfolder="transformer",
            torch_dtype=torch_dtype, cache_dir=_cache_dir, local_files_only=_lfo,
        )
        pipe = LTX2MultiModalPipeline.from_pretrained(
            MODEL_PATH,
            transformer=transformer,
            text_encoder=None, tokenizer=None,
            torch_dtype=torch_dtype, cache_dir=_cache_dir, local_files_only=_lfo,
        )

        # ── Apply LoRAs ─────────────────────────────────────────────────────
        from ...utils.helpers import bpy as _bpy
        _lora_folder   = _bpy.path.abspath(getattr(scene, "lora_folder", ""))
        _enabled_loras = [item for item in getattr(scene, "lora_files", []) if item.enabled]
        _lora_names, _lora_weights = [], []
        if _enabled_loras and _lora_folder:
            print(f"LTX-2.3 LipSync: loading {len(_enabled_loras)} LoRA(s) from {_lora_folder}")
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

        # ── Audio Conditions ────────────────────────────────────────────────
        audio_conditions = None
        if sound_path and hasattr(pipe, "audio_vae") and pipe.audio_vae:
            target_sr = pipe.audio_vae.config.sample_rate
            try:
                waveform = load_audio(sound_path, target_sample_rate=target_sr, seconds=dur_s)
                audio_conditions = [LTX2AudioCondition(audio=waveform, strength=1.0)]
            except Exception:
                import traceback; traceback.print_exc()

        pipe.enable_group_offload(
            onload_device=onload_device,
            offload_type="leaf_level",
            use_stream=True,
            low_cpu_mem_usage=True,
        )

        # ── Inference ───────────────────────────────────────────────────────
        self.set_phase(inputs, f"Generating {w}×{h}, {num_frames} frames")

        pipe_kw = dict(
            prompt_embeds=prompt_embeds.to(onload_device, dtype=torch_dtype),
            prompt_attention_mask=prompt_attention_mask.to(onload_device),
            width=w, height=h,
            num_frames=num_frames,
            frame_rate=fps,
            num_inference_steps=len(DISTILLED_SIGMA_VALUES),
            sigmas=DISTILLED_SIGMA_VALUES,
            guidance_scale=1.0,
            control_downscale_factor=1,
            control_strength=1.0,
            use_cross_timestep=True,
            generator=generator,
            output_type="latent",
            return_dict=False,
            callback_on_step_end=self.step_callback(inputs),
        )

        if image_conditions is not None:
            pipe_kw["image_conditions"] = image_conditions
        if audio_conditions is not None:
            pipe_kw["audio_conditions"] = audio_conditions
            if _modality_scale != 1.0:
                pipe_kw["modality_scale"] = _modality_scale
        if image_conditions is not None and audio_conditions is not None:
            pipe_kw["stg_scale"] = 1.0
            pipe_kw["spatio_temporal_guidance_blocks"] = [28]
            pipe_kw["guidance_rescale"] = 0.7

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch_dtype):
            outputs = pipe(**pipe_kw)

        if isinstance(outputs, (tuple, list)):
            video_latent = outputs[0].detach().to(offload_device, copy=True)
            audio_latent = outputs[1].detach().to(offload_device, copy=True) if len(outputs) > 1 and outputs[1] is not None else None
        else:
            video_latent = outputs.detach().to(offload_device, copy=True)
            audio_latent = None

        del pipe, transformer, prompt_embeds, prompt_attention_mask
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
            video = vae_temporal_decode_streaming(vae, video_latent.to("cpu"), decode_device=onload_device)
            video = decode_pipe.video_processor.postprocess_video(video, output_type="np")

        audio_out = None
        audio_sr  = 24000

        if audio_latent is not None and hasattr(decode_pipe, "audio_vae") and decode_pipe.audio_vae:
            audio_vae = decode_pipe.audio_vae.to(onload_device)
            vocoder   = decode_pipe.vocoder.float().to(onload_device)
            audio_sr  = getattr(vocoder.config, "output_sampling_rate", 24000)
            with torch.inference_mode():
                mel       = audio_vae.decode(audio_latent.to(onload_device, dtype=audio_vae.dtype), return_dict=False)[0]
                audio_out = vocoder(mel.float()).cpu()
            peak = audio_out.abs().max()
            if peak > 1.0:
                audio_out = audio_out / peak
            del audio_vae, vocoder

        del decode_pipe, vae, video_latent, audio_latent
        _flush()

        # ── Save ─────────────────────────────────────────────────────────────
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
                print(f"[LTX23LipSync] Muxing input audio: {_sr} Hz, {_wav.shape[-1]} samples → {dur_s:.2f}s")
            except Exception as _ae:
                print(f"[LTX23LipSync] Input audio mux failed ({_ae}), falling back to model audio.")
                if audio_out is not None:
                    _use_audio    = audio_out
                    _use_audio_sr = audio_sr
        elif audio_out is not None:
            _use_audio    = audio_out
            _use_audio_sr = audio_sr

        if _use_audio is not None:
            encode_video(
                torch.from_numpy((video[0] * 255).round().astype("uint8")),
                fps=fps, audio=_use_audio, audio_sample_rate=_use_audio_sr,
                output_path=dst_path,
            )
        else:
            encode_video(
                torch.from_numpy((video[0] * 255).round().astype("uint8")),
                fps=fps, output_path=dst_path,
            )

        print(f"LTX-2.3 LipSync saved: {dst_path}")
        return dst_path
