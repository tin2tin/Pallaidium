"""MiniMax-H3 joint video+audio generation (SDNQ quantized), auto-routing t2va/fl2va/ref2va from inputs.

Loading omits `workflow=`, so both transformer partitions (t2va/fl2va's
`transformer/` and ref2va's `transformer_ref/`) are resident and the plugin
picks the workflow per job from what's actually supplied:

  - prompt only                         -> t2va   (text-to-video+audio)
  - the standard Input strip is an
    image, nothing else set             -> fl2va  (first-frame conditioning, binds the canvas)
  - anything richer (a second Ref
    Strip and/or an Audio Ref, on top
    of an optional Input image)         -> ref2va (image/video reference(s) + optional audio)

The standard Input strip (image or blank) supplies the primary/fl2va image.
A second optional "Ref Strip" (image or video, via draw_custom_ui) and an
optional "Audio Ref." row (reusing the shared ref_audio_path property) add
ref2va conditioning — this intentionally matches ltx23_multi's scope (one
extra reference strip) rather than a full 9-slot multi-image picker: H3
splits that richer conditioning into a second transformer partition
(transformer_ref/), so loading it costs ~11.4GB more download than
workflow="t2va" alone. If ref2va isn't needed, dropping transformer_ref
saves that download — see this file's git history for the leaner version.

H3 is guidance-distilled: no CFG, no negative_prompt. Fixed 24fps, 5-15s
duration (frame counts snapped to the model's 17*n+5 grid).
"""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram, solve_path, clean_filename

_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


class MiniMaxH3Plugin(ModelPlugin):
    MODEL_ID     = "MiniMaxAI/MiniMax-H3"
    DISPLAY_NAME = "MiniMax H3 (video + audio)"
    MODEL_TYPE   = "video"
    DESCRIPTION  = "Joint video+audio generation via MiniMax H3 (SDNQ quantized); text-to-video, image-to-video, or add a Ref Strip / Audio Ref for reference-conditioned generation"

    INPUTS       = InputSpec.PROMPT | InputSpec.IMAGE | InputSpec.AUDIO_REF | InputSpec.LORA
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.IMAGE_STRIP,
        UISection.RESOLUTION, UISection.FRAMES, UISection.STEPS, UISection.SEED,
        UISection.LORA,
    ]
    # height=480/width=864 (short edge 480, well under the trained 768) and
    # frames=124 (~5.17s, the smallest valid duration) match the upstream
    # 24-32GB recipe's own defaults — smaller canvas is the single biggest
    # speed lever (960x544 runs ~2.3x faster/step than 1344x768 per the docs).
    PARAMS       = ParamSpec(width=864, height=480, frames=124, steps=20)
    REQUIRED_PACKAGES = ["torch", "diffusers", "transformers", "av", "sdnq"]

    def load(self, prefs, scene, **kw):
        import torch
        from diffusers import AutoencoderKLMiniMaxH3, ModularPipeline
        from diffusers.hooks import apply_group_offloading
        from sdnq import SDNQConfig  # noqa: F401 — registers the SDNQ weight loader
        from sdnq.common import use_torch_compile as triton_is_available
        from sdnq.loader import apply_sdnq_options_to_model

        # Upstream bug workaround (diffusers-recipes/minimax_h3): the fp32 pin
        # covers the whole VAE and ignores dtype otherwise.
        AutoencoderKLMiniMaxH3._keep_in_fp32_modules = []

        _cache_dir = prefs.hf_cache_dir or None
        _lfo = prefs.local_files_only
        # The "pruned" repos only prune the transformer; text_encoder loads from
        # a separate, unpruned repo (OzzyGT/MiniMax_H3_sdnq_dynamic_{8,4}bit) —
        # 35GB at 8-bit, 21GB at 4-bit. Loading without workflow= pulls BOTH
        # transformer partitions: real total footprint is ~55GB (8-bit) or
        # ~32GB (4-bit). 8-bit's 35GB text_encoder safetensors mmap needs
        # Windows pagefile headroom well beyond what fits comfortably under
        # 64GB RAM, so require that much before defaulting to it.
        import psutil
        total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        bits = 4 if (low_vram() or total_ram_gb < 64) else 8
        repo_id = f"OzzyGT/MiniMax_H3_sdnq_{bits}bit_pruned"
        dtype = torch.bfloat16

        print(f"Loading {repo_id} (t2va + fl2va + ref2va, all workflows)…")
        # No workflow= at either call: keeps both transformer partitions
        # resident so the plugin can route per-job at generate() time.
        pipe = ModularPipeline.from_pretrained(repo_id, trust_remote_code=True, cache_dir=_cache_dir, local_files_only=_lfo)
        pipe.load_components(dtype=dtype, trust_remote_code=True, cache_dir=_cache_dir, local_files_only=_lfo)
        # load_components() logs a per-component failure as a warning and leaves
        # the attribute None instead of raising — fail loudly here instead of
        # letting the first .model/.enable_group_offload() access below crash
        # with an opaque AttributeError.
        missing = [n for n in ("transformer", "transformer_ref", "text_encoder", "vae", "audio_vae") if getattr(pipe, n, None) is None]
        if missing:
            raise RuntimeError(
                f"MiniMax H3: component(s) failed to load: {missing}. Check the console above this "
                "for the underlying error (a Windows 'paging file is too small' error here means the "
                "text_encoder's ~21-35GB safetensors file couldn't be memory-mapped — increase the "
                "Windows pagefile size, or free up RAM/close other apps, and try again)."
            )

        if triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
            pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
            pipe.transformer_ref = apply_sdnq_options_to_model(pipe.transformer_ref, use_quantized_matmul=True)
            pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)

        # LoRA — loaded into both partitions up front since the workflow (and
        # therefore which partition actually runs) is only known per-job, at
        # generate() time, long after this cached load() has returned.
        enabled_items = kw.get("enabled_items", [])
        if enabled_items:
            from ...utils.helpers import bpy
            lora_folder = bpy.path.abspath(getattr(bpy.context.scene, "lora_folder", "") or "")
            names, weights = [], []
            for item in enabled_items:
                name = clean_filename(item.name).replace(".", "")
                names.append(name)
                weights.append(item.weight_value)
                pipe.load_lora_weights(
                    lora_folder, weight_name=item.name + ".safetensors", adapter_name=name,
                )
                try:
                    pipe.load_lora_weights(
                        lora_folder, weight_name=item.name + ".safetensors", adapter_name=name,
                        load_into_transformer_ref=True,
                    )
                except Exception as e:
                    print(f"MiniMax H3: LoRA '{item.name}' did not load into transformer_ref (ref2va will run without it): {e}")
                print(f"MiniMax H3: LoRA '{item.name}' loaded (adapter='{name}', weight={item.weight_value:.2f})")
            pipe.set_adapters(names, adapter_weights=weights)

        if gfx_device == "mps":
            pipe.to("mps")
            return {"pipe": pipe, "bits": bits}

        onload_device = torch.device(gfx_device)
        offload_device = torch.device("cpu")
        # use_stream=False: streamed group-offload pins page-locked host RAM on
        # every onload, which dies mid-forward on this Windows/Blender stack
        # ("CUDA error: out of memory" / "resource already mapped") even when
        # VRAM is free — see the same fix in ltx23_multi.py.
        #
        # Both transformer partitions stream here even though their static
        # weights (~11GB each at 4-bit) would fit resident on a 24GB card: for
        # ref2va with a video+image reference, the conditioner's activation
        # memory (attention over many extra vision tokens) is large enough
        # that resident transformers pushed total usage past 24GB — observed
        # as Windows silently spilling into shared GPU memory (reported as
        # ~32GB "VRAM" on a 24GB card) rather than an OOM error, which is far
        # slower than streaming, not faster.
        #
        # `transformer` (the only partition t2va/fl2va ever actually run)
        # uses num_blocks_per_group=2 as a speed test: a 1280x736x120f fl2va
        # job measured 19GB/24GB used at group=1 (34.04s/it) — 5GB of
        # headroom going unused specifically because group=1 maximizes
        # transfer round-trips over VRAM footprint. Doubling the group size
        # halves those round-trips per step in exchange for a modest VRAM
        # bump. `transformer_ref` stays at 1: it's never invoked by t2va/fl2va
        # (idle weights add ~0 VRAM cost while offloaded), so there's no
        # speed to gain there, only risk, on the workflow this was tested on.
        # If this regresses VRAM (spills into shared memory again — check
        # Task Manager GPU usage, a real number should stay <= 24GB) or
        # doesn't measurably help, drop it back to 1.
        offload = dict(onload_device=onload_device, offload_device=offload_device, use_stream=False, low_cpu_mem_usage=True)
        pipe.transformer.enable_group_offload(offload_type="block_level", num_blocks_per_group=2, **offload)
        pipe.transformer_ref.enable_group_offload(offload_type="block_level", num_blocks_per_group=1, **offload)
        apply_group_offloading(pipe.text_encoder.model, offload_type="leaf_level", **offload)
        # VAEs stay fully on-GPU: they're small, and block-level offload hooks
        # a module's forward(), which pipelines calling .encode()/.decode()
        # directly never trigger — weights would be stranded on CPU otherwise.
        pipe.vae.to(gfx_device)
        pipe.audio_vae.to(gfx_device)

        return {"pipe": pipe, "bits": bits}

    def draw_custom_ui(self, col, context) -> bool:
        scene = context.scene
        # Strip refs live in the scene shown in the VSE (context.sequencer_scene
        # in Blender 5.x), which can differ from the active scene.
        vse_scene = getattr(context, "sequencer_scene", None) or context.scene
        if vse_scene.sequence_editor is not None:
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
        """Snap to H3's 17*n+5 decodable grid, clamped to its 5-15s @ 24fps window."""
        target = max(120, min(int(target), 345))
        n = -(-(target - 5) // 17)  # ceil division
        return min(17 * n + 5, 345)

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import os
        import torch
        from diffusers.modular_pipelines.minimax_h3 import (
            MiniMaxH3ImageReference, MiniMaxH3VideoReference, MiniMaxH3AudioReference,
        )
        from diffusers.utils.export_utils import encode_video

        pipe = pipe_obj["pipe"]
        seed = inputs.seed
        generator = torch.Generator("cpu").manual_seed(seed) if seed != 0 else None

        w = max(32, (inputs.width // 32) * 32)
        h = max(32, (inputs.height // 32) * 32)
        num_frames = self._snap_frames(inputs.frames)

        primary_image = inputs.image
        ref_path = getattr(scene, "h3_ref_strip_path", "") or None
        audio_path = inputs.audio_ref

        # An audio reference has to be paired with at least one image/video
        # reference (H3 does not support conditioning on audio alone).
        if audio_path and not primary_image and not ref_path:
            print("MiniMax H3: Audio Ref needs the Input image or a Ref Strip set too; ignoring Audio Ref.")
            audio_path = None

        # No callback_on_step_end: MiniMaxH3Blocks doesn't declare it as an
        # input (diffusers warns "Unexpected input... will be ignored" if
        # passed), so Blender's own progress bar can't track per-step
        # progress here — watch the console's native tqdm step counter instead.
        common = dict(
            prompt=inputs.prompt,
            height=h,
            width=w,
            num_frames=num_frames,
            num_inference_steps=inputs.steps,
            generator=generator,
            output=["videos", "audio", "sampling_rate"],
        )

        if not primary_image and not ref_path and not audio_path:
            self.set_phase(inputs, "Generating (t2va)")
            result = pipe(**common)
        elif primary_image is not None and not ref_path and not audio_path:
            self.set_phase(inputs, "Generating (fl2va)")
            result = pipe(image=primary_image, **common)
        else:
            self.set_phase(inputs, "Generating (ref2va)")
            references = []
            if primary_image is not None:
                references.append(MiniMaxH3ImageReference(image=primary_image))
            if ref_path:
                ext = os.path.splitext(ref_path)[1].lower()
                if ext in _VIDEO_EXTS:
                    references.append(MiniMaxH3VideoReference.from_file(ref_path))
                else:
                    references.append(MiniMaxH3ImageReference.from_file(ref_path))
            if audio_path:
                references.append(MiniMaxH3AudioReference.from_file(audio_path))
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
