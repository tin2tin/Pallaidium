"""Pika H3 Flex joint video+audio generation — the six-step post-trained MiniMax-H3, on SDNQ weights.

Pika H3 Flex is MiniMax-H3 plus two things: a pair of rank-128 LoRA adapters
(`fl2va/adapter.safetensors` for the `transformer/` partition,
`ref2va/adapter.safetensors` for `transformer_ref/`) and a fixed six-step
stochastic sampler (`denoising_step_list` [1000, 750, 500, 250, 100, 40, 0],
re-noising the clean prediction at every step). Pika's own repo ships the
MiniMax-H3 base in bf16 — 66GB per transformer partition plus a 66GB text
encoder — so this plugin mounts Pika's blocks, schedulers and adapters on the
same OzzyGT SDNQ-pruned components the MiniMax H3 plugin uses. The adapters
target only attention and feed-forward (`to_q/k/v/out.0`, `ff.net.0.proj`,
`ff.net.2`), never the AdaLN projections the pruned checkpoint folds into a
cached curve, so they apply to the pruned transformer unchanged.

Workflow routing per job, from what's actually supplied:

  - prompt only                              -> t2va  (text-to-video+audio)
  - Input image and/or Last Frame strip,
    nothing else                             -> fl2va (first / last / first+last keyframes)
  - a Ref Strip and/or an Audio Ref, on top
    of an optional Input image               -> ref2va (image/video reference(s) + optional audio)

A META input strip with two images at different frame starts is picked up by
the queue as first + last keyframe, as for LTX Multi.

Guidance-distilled like H3: no CFG, no negative prompt, and no step count —
six steps are the adapters' contract. Fixed 24fps, 5-15s (frame counts snapped
to the 17*n+5 grid, up to 362). Canvases snap to Pika's validated 480p/768p
aspect-ratio buckets.
"""

import math

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram, solve_path, clean_filename
from .minimax_h3 import _VIDEO_EXTS, _cached_snapshot_dir, _onload_vae_per_call, _queue_progress

_PIKA_REPO = "Pika-Labs/Pika-H3-Flex"
_PIKA_ADAPTERS = {
    "transformer": ("pika_fl2va", "fl2va/adapter.safetensors"),
    "transformer_ref": ("pika_ref2va", "ref2va/adapter.safetensors"),
}

# The released bucket table: (width, height), 11 aspect ratios per tier.
_BUCKETS = {
    480: [(960, 416), (832, 480), (704, 480), (640, 480), (608, 480), (480, 480),
          (480, 608), (480, 640), (480, 704), (480, 832), (416, 960)],
    768: [(1536, 672), (1344, 768), (1152, 768), (1024, 768), (960, 768), (768, 768),
          (768, 960), (768, 1024), (768, 1152), (768, 1344), (672, 1536)],
}

# Every video or standalone audio reference must run 2-15s (Pika's limits).
_REF_MIN_S = 2.0
_REF_MAX_S = 15.0


def _load_hub_file(repo_id, filename, cache_dir, local_files_only):
    """Cache-first hf_hub_download: no network round-trip once the file is cached."""
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(repo_id, filename, cache_dir=cache_dir, local_files_only=True)
    except Exception:
        if local_files_only:
            raise
        return hf_hub_download(repo_id, filename, cache_dir=cache_dir)


class PikaH3FlexPlugin(ModelPlugin):
    MODEL_ID     = "Pika-Labs/Pika-H3-Flex"
    DISPLAY_NAME = "Pika H3 Flex (video + audio, 6 steps)"
    MODEL_TYPE   = "video"
    DESCRIPTION  = ("Six-step joint video+audio generation via Pika H3 Flex (MiniMax-H3 post-train, SDNQ quantized); "
                    "text-to-video, first/last-frame-to-video, or add a Ref Strip / Audio Ref for reference-conditioned generation")

    INPUTS       = InputSpec.PROMPT | InputSpec.IMAGE | InputSpec.AUDIO_REF | InputSpec.LORA
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.IMAGE_STRIP,
        UISection.RESOLUTION, UISection.FRAMES, UISection.SEED,
        UISection.LORA,
    ]
    # 832x480 (480p 16:9 bucket) and 124 frames (~5.17s, the shortest valid
    # duration) match the upstream examples. Steps are fixed at six.
    PARAMS       = ParamSpec(width=832, height=480, frames=124, steps=6)
    REQUIRED_PACKAGES = ["torch", "diffusers", "transformers", "av", "sdnq"]

    def load(self, prefs, scene, **kw):
        import os
        import torch
        try:
            from diffusers import MiniMaxH3Scheduler, PikaH3FlexBlocks, PikaH3FlexModularPipeline
            MiniMaxH3Scheduler(shift=12.0, stochastic_sampling=True, shift_provided_sigmas=True)
        except (ImportError, TypeError) as e:
            raise RuntimeError(
                "Pika H3 Flex needs a newer diffusers (PikaH3FlexBlocks and the stochastic MiniMaxH3Scheduler). "
                "Run Install Dependencies to update diffusers, then restart Blender."
            ) from e
        from diffusers.hooks import apply_group_offloading
        from sdnq import SDNQConfig  # noqa: F401 — registers the SDNQ weight loader
        from sdnq.common import use_torch_compile as triton_is_available
        from sdnq.loader import apply_sdnq_options_to_model

        _cache_dir = prefs.hf_cache_dir or None
        _lfo = prefs.local_files_only
        # Same base weights and RAM rule as the MiniMax H3 plugin (shared HF
        # cache, no second download): 8-bit only with >= 64GB RAM.
        import psutil
        total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        bits = 4 if (low_vram() or total_ram_gb < 64) else 8
        repo_id = f"OzzyGT/MiniMax_H3_sdnq_{bits}bit_pruned"
        dtype = torch.bfloat16

        print(f"Loading Pika H3 Flex on {repo_id} (t2va + fl2va + ref2va, all workflows)…")
        # Pika's blocks over the OzzyGT index: the pipeline class follows the
        # blocks (PikaH3FlexModularPipeline, with its LoRA mixin and six-step
        # config defaults) while every component spec comes from the SDNQ repo.
        def _make_pipe(local_only):
            return PikaH3FlexModularPipeline(
                blocks=PikaH3FlexBlocks(), pretrained_model_name_or_path=repo_id,
                cache_dir=_cache_dir, local_files_only=local_only,
            )
        try:
            pipe = _make_pipe(True)
            from_cache = True
        except (OSError, ValueError):
            if _lfo:
                raise
            pipe = _make_pipe(False)
            from_cache = False

        # Pika's schedulers (its scheduler/ and audio_scheduler/ configs): the
        # base H3 shifts, re-noising at every step over the provided sigmas.
        # Registered before loading so the loops below skip the repo's own.
        pipe.update_components(
            scheduler=MiniMaxH3Scheduler(shift=12.0, stochastic_sampling=True, shift_provided_sigmas=True),
            audio_scheduler=MiniMaxH3Scheduler(shift=3.0, stochastic_sampling=True, shift_provided_sigmas=True),
        )

        # Vendored remote-code classes, as in the MiniMax H3 plugin.
        from ._minimax_h3_pruned import MiniMaxH3PrunedTransformer3DModel
        from ._minimax_h3_vae_fp16 import MiniMaxH3Fp16DecoderVae
        for _name, _cls in (("transformer", MiniMaxH3PrunedTransformer3DModel),
                            ("transformer_ref", MiniMaxH3PrunedTransformer3DModel),
                            ("vae", MiniMaxH3Fp16DecoderVae)):
            pipe._component_specs[_name].type_hint = _cls
        if from_cache:
            # Load from cached snapshot folders, not repo ids — see minimax_h3.py.
            for name, spec in list(pipe._component_specs.items()):
                if spec.default_creation_method != "from_pretrained" or getattr(pipe, name, None) is not None:
                    continue
                try:
                    local_dir = _cached_snapshot_dir(spec.pretrained_model_name_or_path, _cache_dir)
                except OSError:
                    continue  # not cached — fetched online below
                pipe.load_components(names=name, pretrained_model_name_or_path=local_dir, dtype=dtype,
                                     cache_dir=_cache_dir, local_files_only=True)
            uncached = [n for n, sp in pipe._component_specs.items()
                        if sp.default_creation_method == "from_pretrained" and getattr(pipe, n, None) is None]
            if uncached and not _lfo:
                print(f"Pika H3 Flex: not in local cache, fetching online: {uncached}")
                pipe.load_components(names=uncached, dtype=dtype, cache_dir=_cache_dir, local_files_only=False)
        else:
            uncached = [n for n, sp in pipe._component_specs.items()
                        if sp.default_creation_method == "from_pretrained" and getattr(pipe, n, None) is None]
            pipe.load_components(names=uncached, dtype=dtype, cache_dir=_cache_dir, local_files_only=_lfo)
        missing = [n for n in ("transformer", "transformer_ref", "text_encoder", "vae", "audio_vae") if getattr(pipe, n, None) is None]
        if missing:
            raise RuntimeError(
                f"Pika H3 Flex: component(s) failed to load: {missing}. Check the console above this "
                "for the underlying error (a Windows 'paging file is too small' error here means the "
                "text_encoder's ~21-35GB safetensors file couldn't be memory-mapped — increase the "
                "Windows pagefile size, or free up RAM/close other apps, and try again)."
            )

        # fp16-decoder VAE dtype alignment — see minimax_h3.py.
        vae_compute_dtype = pipe.vae.decoder.proj_in.weight.dtype
        pipe.vae.post_quant_conv.to(vae_compute_dtype)
        for name, module in pipe.vae.decoder.named_modules():
            if isinstance(module, (torch.nn.LayerNorm, torch.nn.RMSNorm)) and name.rsplit(".", 1)[-1].startswith("norm"):
                module.to(torch.float32)

        if triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
            pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
            pipe.transformer_ref = apply_sdnq_options_to_model(pipe.transformer_ref, use_quantized_matmul=True)
            pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)

        # Pika's adapters, one per partition, at their trained scale 1.0
        # (alpha == rank). Must load before group offload is enabled below.
        from diffusers.utils import logging as _dlog
        adapters = {part: ([], []) for part in _PIKA_ADAPTERS}
        for part, (adapter_name, filename) in _PIKA_ADAPTERS.items():
            path = _load_hub_file(_PIKA_REPO, filename, _cache_dir, _lfo)
            # Silence diffusers' "LoRA does not say which partition" notice: the
            # routing is explicit here, one adapter per partition.
            _prev = _dlog.get_verbosity()
            _dlog.set_verbosity_error()
            try:
                pipe.load_lora_weights(
                    os.path.dirname(path), weight_name=os.path.basename(path), adapter_name=adapter_name,
                    load_into_transformer_ref=(part == "transformer_ref"),
                )
            finally:
                _dlog.set_verbosity(_prev)
            adapters[part][0].append(adapter_name)
            adapters[part][1].append(1.0)
            print(f"Pika H3 Flex: adapter '{filename}' loaded into {part}")

        # User LoRAs stack on top, into both partitions — the workflow (and so
        # the partition that runs) is only known per job.
        enabled_items = kw.get("enabled_items", [])
        if enabled_items:
            from ...utils.helpers import bpy
            lora_folder = bpy.path.abspath(getattr(bpy.context.scene, "lora_folder", "") or "")
            for item in enabled_items:
                name = clean_filename(item.name).replace(".", "")
                for part in _PIKA_ADAPTERS:
                    _prev = _dlog.get_verbosity()
                    _dlog.set_verbosity_error()
                    try:
                        pipe.load_lora_weights(
                            lora_folder, weight_name=item.name + ".safetensors", adapter_name=name,
                            load_into_transformer_ref=(part == "transformer_ref"),
                        )
                    except Exception as e:
                        if part == "transformer":
                            raise
                        print(f"Pika H3 Flex: LoRA '{item.name}' did not load into transformer_ref (ref2va will run without it): {e}")
                        continue
                    finally:
                        _dlog.set_verbosity(_prev)
                    adapters[part][0].append(name)
                    adapters[part][1].append(item.weight_value)
                print(f"Pika H3 Flex: LoRA '{item.name}' loaded (adapter='{name}', weight={item.weight_value:.2f})")
        # Per partition: each holds its own Pika adapter, which the other lacks.
        for part, (names, weights) in adapters.items():
            getattr(pipe, part).set_adapters(names, weights)

        if gfx_device == "mps":
            pipe.to("mps")
            return {"pipe": pipe, "bits": bits}

        # Offload layout and its reasons — see minimax_h3.py.
        onload_device = torch.device(gfx_device)
        offload_device = torch.device("cpu")
        from ...utils.helpers import ensure_group_offload_keep_cpu_weights
        ensure_group_offload_keep_cpu_weights()
        offload = dict(onload_device=onload_device, offload_device=offload_device, use_stream=False, low_cpu_mem_usage=True)
        pipe.transformer.enable_group_offload(offload_type="block_level", num_blocks_per_group=2, **offload)
        pipe.transformer_ref.enable_group_offload(offload_type="block_level", num_blocks_per_group=1, **offload)
        apply_group_offloading(pipe.text_encoder.model, offload_type="leaf_level", **offload)
        _onload_vae_per_call(pipe.vae, onload_device)
        _onload_vae_per_call(pipe.audio_vae, onload_device)
        from ._h3_conditioner import install_lean_conditioner
        install_lean_conditioner("diffusers.modular_pipelines.pika_h3_flex.encoders")

        return {"pipe": pipe, "bits": bits}

    def draw_custom_ui(self, col, context) -> bool:
        scene = context.scene
        vse_scene = getattr(context, "sequencer_scene", None) or context.scene
        if vse_scene.sequence_editor is not None:
            row = col.row(align=True)
            row.prop_search(
                vse_scene, "h3_last_strip", vse_scene.sequence_editor, "strips",
                text="Last Frame", icon="IMAGE_DATA",
            )
            row.operator("sequencer.strip_picker", text="", icon="EYEDROPPER").action = "h3_last_select"
            row = col.row(align=True)
            row.prop_search(
                vse_scene, "h3_ref_strip", vse_scene.sequence_editor, "strips",
                text="Ref Strip", icon="SEQ_STRIP_DUPLICATE",
            )
            row.operator("sequencer.strip_picker", text="", icon="EYEDROPPER").action = "h3_ref_select"
        row = col.row(align=True)
        row.prop(scene, "ref_audio_path", text="Audio Ref.")
        row.operator("sequencer.open_audio_filebrowser", text="", icon="FILEBROWSER")
        return False  # additive only — doesn't replace the standard Input row

    @staticmethod
    def _snap_frames(target):
        """Snap up to the 17*n+5 decodable grid within Pika's 5-15s @ 24fps window (124-362 frames)."""
        target = max(120, min(int(target), 360))
        n = -(-(target - 5) // 17)  # ceil division
        return 17 * n + 5

    @staticmethod
    def _snap_bucket(width, height):
        """The validated bucket nearest in aspect ratio, in the 480p or 768p tier by short edge."""
        tier = 768 if min(width, height) >= 624 else 480
        ratio = math.log(width / height)
        return min(_BUCKETS[tier], key=lambda wh: abs(math.log(wh[0] / wh[1]) - ratio))

    @staticmethod
    def _load_image(path):
        from PIL import Image
        from ...utils.helpers import load_first_frame
        img = load_first_frame(path)
        return img.convert("RGB") if isinstance(img, Image.Image) else img

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import os
        import torch
        from diffusers.modular_pipelines.pika_h3_flex import (
            PikaH3FlexImageReference, PikaH3FlexVideoReference, PikaH3FlexAudioReference,
        )
        from diffusers.utils.export_utils import encode_video

        pipe = pipe_obj["pipe"]
        seed = inputs.seed
        generator = torch.Generator("cpu").manual_seed(seed) if seed != 0 else None

        w, h = self._snap_bucket(max(32, inputs.width), max(32, inputs.height))
        if (w, h) != (inputs.width, inputs.height):
            print(f"Pika H3 Flex: {inputs.width}x{inputs.height} snapped to the {w}x{h} bucket")
        num_frames = self._snap_frames(inputs.frames)

        first_image = inputs.image
        last_image = inputs.last_image
        last_path = getattr(scene, "h3_last_strip_path", "") or None
        if last_image is None and last_path:
            last_image = self._load_image(last_path)
        ref_path = getattr(scene, "h3_ref_strip_path", "") or None
        audio_path = inputs.audio_ref

        if audio_path and not first_image and not ref_path:
            print("Pika H3 Flex: Audio Ref needs the Input image or a Ref Strip set too; ignoring Audio Ref.")
            audio_path = None

        common = dict(
            prompt=inputs.prompt,
            height=h,
            width=w,
            num_frames=num_frames,
            generator=generator,
            output=["videos", "audio", "sampling_rate"],
        )

        if not ref_path and not audio_path:
            if first_image is None and last_image is None:
                self.set_phase(inputs, "Generating (t2va)")
                with _queue_progress(self, inputs):
                    result = pipe(**common)
            else:
                keyframes = {}
                if first_image is not None:
                    keyframes["image"] = first_image
                if last_image is not None:
                    keyframes["last_image"] = last_image
                label = {("image",): "first frame", ("last_image",): "last frame"}.get(tuple(keyframes), "first + last frame")
                self.set_phase(inputs, f"Generating (fl2va, {label})")
                with _queue_progress(self, inputs):
                    result = pipe(**keyframes, **common)
        else:
            if last_image is not None:
                print("Pika H3 Flex: a last frame can't be combined with a Ref Strip / Audio Ref (ref2va); ignoring it.")
            self.set_phase(inputs, "Generating (ref2va)")
            references = []
            if first_image is not None:
                references.append(PikaH3FlexImageReference(image=first_image))
            if ref_path:
                ext = os.path.splitext(ref_path)[1].lower()
                if ext in _VIDEO_EXTS:
                    ref = PikaH3FlexVideoReference.from_file(ref_path)
                    duration = len(ref.frames) / (ref.fps or 24.0)
                    if duration < _REF_MIN_S:
                        # Too short for a motion reference: keep its look as an image reference.
                        print(f"Pika H3 Flex: Ref Strip video is {duration:.1f}s (< {_REF_MIN_S:g}s); using its first frame as an image reference.")
                        ref = PikaH3FlexImageReference(image=self._load_image(ref_path))
                    references.append(ref)
                else:
                    references.append(PikaH3FlexImageReference.from_file(ref_path))
            if audio_path:
                aref = PikaH3FlexAudioReference.from_file(audio_path)
                sr = aref.sample_rate or pipe.audio_sampling_rate
                duration = aref.audio.shape[-1] / sr
                if duration < _REF_MIN_S:
                    print(f"Pika H3 Flex: Audio Ref is {duration:.1f}s (< {_REF_MIN_S:g}s); ignoring it.")
                else:
                    if duration > _REF_MAX_S:
                        print(f"Pika H3 Flex: Audio Ref is {duration:.1f}s; using its first {_REF_MAX_S:g}s.")
                        aref.audio = aref.audio[..., : int(_REF_MAX_S * sr)]
                    references.append(aref)
            if not any(not isinstance(r, PikaH3FlexAudioReference) for r in references):
                raise RuntimeError("Pika H3 Flex: ref2va needs at least one image or video reference.")
            with _queue_progress(self, inputs):
                result = pipe(references=references, **common)

        self.set_phase(inputs, "Saving")
        video = result["videos"][0]
        audio = result["audio"][0]
        sampling_rate = result["sampling_rate"]

        dst_path = solve_path(clean_filename(str(seed) + "_" + inputs.prompt) + ".mp4")
        encode_video(
            video, fps=24, output_path=dst_path,
            audio=audio, audio_sample_rate=sampling_rate,
        )
        return dst_path
