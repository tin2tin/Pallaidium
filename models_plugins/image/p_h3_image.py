"""P-H3-Image (PrunaAI's MiniMax-H3 image fine-tune): text-to-image and multi-image editing, 8 steps, no CFG.

Diffusers integrates it as Modular Diffusers blocks only (PH3ImageBlocks); the workflow follows the inputs:
  - prompt only                              -> text2image
  - the Input strip and/or Ref. strips set   -> image_conditioned (edit); the first image is edited, the
                                                others are references, named <Picture 1>, <Picture 2>, ...

Weights: OzzyGT/P_H3_Image_sdnq_dynamic_4bit (SDNQ uint4 transformer 18.8 GB + Qwen3-VL-4B text encoder
3.1 GB; VAE 5.6 GB and the text connector in bf16). Unlike MiniMax H3 Image's 51-layer encoder, the 4B
encoder plus connector stay on the GPU; the transformer streams block by block (the VAE and the
activations of a 4-8 MP canvas don't fit beside a resident 18.8 GB transformer on a 24 GB card).

The pipeline's own prompt upsampler (Qwen3.5-9B, bf16) is removed: the prompt renders as written. The
model card: it works best at 8 MP (4 MP for edits) and with full, specific prompts.
"""

import os

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram
from .minimax_h3_image import _Timer, _check_materialized, _crop_to_subject, _in_cache, _release_pipeline


_REPO = "OzzyGT/P_H3_Image_sdnq_dynamic_4bit"
_COMPONENTS = ("text_encoder", "text_connector", "tokenizer", "processor", "vae", "transformer",
               "scheduler", "audio_scheduler")
_DOWNLOADS = (
    ("transformer (~18.8 GB)", ("transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
                                "transformer/diffusion_pytorch_model-00002-of-00002.safetensors")),
    ("text encoder (~3.1 GB)", ("text_encoder/model.safetensors",)),
    ("VAE (~5.6 GB)", ("vae/diffusion_pytorch_model.safetensors",)),
)
_MAX_REFS = 8        # the pipeline's limit: the image to edit + 7 references
_MIN_REF_SIDE = 32
_LABEL = "P-H3 Image"


class PH3ImagePlugin(ModelPlugin):
    MODEL_ID     = "PrunaAI/P-H3-Image"
    DISPLAY_NAME = "P-H3 Image"
    MODEL_TYPE   = "image"
    DESCRIPTION  = (
        "P-H3-Image (PrunaAI's MiniMax-H3 fine-tune, 8 steps): text-to-image and editing with up to 8 images "
        "(name them <Picture 1>, <Picture 2>, ...). Small Qwen3-VL-4B text encoder; best at 4-8 MP. "
        "OzzyGT's SDNQ 4-bit build"
    )
    INPUTS       = InputSpec.PROMPT | InputSpec.IMAGE
    UI_SECTIONS  = [UISection.PROMPT, UISection.IMAGE_STRIP, UISection.RESOLUTION, UISection.FRAMES, UISection.SEED]
    # Steps and guidance are fixed by the checkpoint (8 steps on its own sigma grid, no CFG).
    PARAMS       = ParamSpec(width=2720, height=1536, steps=8, guidance=1.0)
    REQUIRED_PACKAGES = ["torch", "diffusers", "transformers", "sdnq"]
    supports_inpaint  = False
    uses_strip_power  = False

    # ------------------------------------------------------------------ load

    def load(self, prefs, scene, **kw):
        import torch
        try:
            from diffusers import ModularPipeline  # noqa: F401
            import diffusers.modular_pipelines.p_h3_image  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "P-H3 Image needs a newer diffusers with P-H3-Image support. Run Install Dependencies in the "
                "add-on Preferences, then restart Blender."
            ) from exc
        import sdnq  # noqa: F401 — registers the SDNQ quantizer with diffusers/transformers
        if kw.get("enabled_items"):
            print("P-H3 Image: LoRAs are not supported by this model in diffusers — ignoring the LoRA list.")

        self._check_offline(prefs)
        print(f"P-H3 Image: loading ({_REPO})…")
        timer = _Timer("load", _LABEL)
        pipe = self._cache_first(prefs, lambda lfo: self._load_pipe(prefs, lfo))
        timer.mark("components")
        missing = [n for n in _COMPONENTS if getattr(pipe, n, None) is None]
        if missing:
            raise RuntimeError(f"P-H3 Image: failed to load {', '.join(missing)} (see the console above)")
        _check_materialized(pipe.transformer, "P-H3 Image transformer")
        tokenizer = pipe.tokenizer
        if not tokenizer("a photo", add_special_tokens=False)["input_ids"]:
            raise RuntimeError("P-H3 Image: the tokenizer loaded without a vocabulary (incomplete download?). "
                               f"Delete the {_REPO} folder from the HF cache and run again.")

        from sdnq.common import use_torch_compile as triton_is_available
        if triton_is_available and torch.cuda.is_available():
            from sdnq.loader import apply_sdnq_options_to_model
            pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
            pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)

        self._place(pipe)
        timer.mark("placement")
        timer.report()
        return {"pipe": pipe, "release_hook": lambda: _release_pipeline(
            pipe, ("transformer", "text_encoder", "text_connector", "vae"))}

    @staticmethod
    def _load_pipe(prefs, local_only):
        import torch
        from diffusers import ModularPipeline
        hub = dict(cache_dir=prefs.hf_cache_dir or None, local_files_only=local_only)
        blocks = ModularPipeline.from_pretrained(_REPO, **hub).blocks
        # The upsampler is a bf16 Qwen3.5-9B from PrunaAI's repo; the prompt renders as written.
        blocks.sub_blocks.pop("prompt_upsampler", None)
        pipe = blocks.init_pipeline(_REPO)  # takes no hub kwargs; the index is cached by the line above
        # Cached components load from the snapshot folder: by repo id, local-only, the processor's inner
        # tokenizer probes a processor/config.json the repo doesn't have and fails (a logged traceback).
        # load_components logs a failed component and goes on; what's still missing is fetched online.
        from ..video.minimax_h3 import _cached_snapshot_dir
        try:
            local_dir = _cached_snapshot_dir(_REPO, hub["cache_dir"])
        except OSError:
            local_dir = None
        if local_dir:
            pipe.load_components(names=list(_COMPONENTS), pretrained_model_name_or_path=local_dir,
                                 dtype=torch.bfloat16, cache_dir=hub["cache_dir"], local_files_only=True)
        else:
            pipe.load_components(names=list(_COMPONENTS), dtype=torch.bfloat16, **hub)
        missing = [n for n in _COMPONENTS if getattr(pipe, n, None) is None]
        if missing:
            if not prefs.local_files_only:
                print(f"P-H3 Image: not in local cache, fetching online: {missing}")
                pipe.load_components(names=missing, dtype=torch.bfloat16, cache_dir=hub["cache_dir"],
                                     local_files_only=False)
        for name in ("text_encoder", "text_connector", "vae", "transformer"):
            model = getattr(pipe, name, None)
            if model is not None:
                model.eval()
        return pipe

    @staticmethod
    def _missing_downloads(cache_dir) -> list:
        return [label for label, files in _DOWNLOADS if not _in_cache(_REPO, files, cache_dir)]

    def _check_offline(self, prefs):
        if not prefs.local_files_only:
            return
        todo = self._missing_downloads(prefs.hf_cache_dir)
        if todo:
            raise RuntimeError(
                "P-H3 Image is not downloaded yet and 'Use Local Files Only' is on. Uncheck it in the add-on "
                f"Preferences to download: {', '.join(todo)}."
            )

    @staticmethod
    def _cache_first(prefs, fn):
        """Call fn(local_files_only) from the HF cache first; online only for what isn't cached."""
        try:
            return fn(True)
        except (OSError, ValueError):
            if prefs.local_files_only:
                raise
            return fn(False)

    def _place(self, pipe) -> None:
        import torch
        if gfx_device == "mps":
            pipe.to("mps")
            print("P-H3 Image: on Apple GPU (MPS)")
            return
        if gfx_device != "cuda":
            print("P-H3 Image: on CPU (no CUDA GPU found) — expect very slow generation")
            return

        from ...utils.helpers import ensure_group_offload_keep_cpu_weights
        from ..video._h3_conditioner import install_fast_patch_embed

        onload, offload = torch.device("cuda"), torch.device("cpu")
        # Unstreamed group offload reusing the original CPU tensors: pinned host memory and the
        # GPU->CPU copy-back both crash or leak on this Windows/Blender stack (see minimax_h3.py).
        ensure_group_offload_keep_cpu_weights()
        opts = dict(onload_device=onload, offload_device=offload, use_stream=False, low_cpu_mem_usage=True)
        pipe.transformer.enable_group_offload(offload_type="block_level", num_blocks_per_group=2, **opts)
        if low_vram():
            from diffusers.hooks import apply_group_offloading
            apply_group_offloading(pipe.text_encoder.model, offload_type="leaf_level", **opts)
            pipe.vae.enable_group_offload(offload_type="leaf_level", **opts)
            pipe.text_connector.to(onload)
            placement = "everything streamed (low VRAM)"
        else:
            for name in ("text_encoder", "text_connector", "vae"):
                getattr(pipe, name).to(onload)
            placement = "text encoder + VAE on the GPU"
        # Reference images: bf16 Conv3d patch embed is pathologically slow here; a matmul is ~2 ms.
        install_fast_patch_embed(pipe.text_encoder)
        print(f"P-H3 Image: {placement}; transformer streamed block by block")

    # ------------------------------------------------------------------ UI

    def draw_custom_ui(self, col, context) -> bool:
        scene = context.scene
        vse_scene = getattr(context, "sequencer_scene", None) or scene
        try:
            col.prop(scene, "input_strips", text="Input")
        except Exception:
            pass
        if vse_scene.sequence_editor is None:
            return True
        visible = min(scene.klein_visible_strips, _MAX_REFS - 1)
        for i in range(1, visible + 1):
            row = col.row(align=True)
            row.prop_search(
                vse_scene, f"klein_strip_{i}", vse_scene.sequence_editor, "strips",
                text="Ref.", icon="FILE_IMAGE",
            )
            row.operator("sequencer.strip_picker", text="", icon="EYEDROPPER").action = f"klein_select{i}"
            if i == visible and visible < _MAX_REFS - 1:
                if visible > 3:
                    row.operator("object.klein_hide_strip", text="", icon="REMOVE").strip_index = i
                row.operator("object.klein_add_strip", text="", icon="ADD")
        if any(getattr(vse_scene, f"klein_strip_{i}", "") for i in range(1, _MAX_REFS)) or \
                getattr(scene, "input_strips", "") == "input_strips":
            col.label(text="The first image is edited; name images <Picture 1>, <Picture 2>…", icon="INFO")
        return True

    def draw_model_options_ui(self, col, context):
        scene = context.scene
        sub = col.column(align=True)
        if hasattr(scene, "h3img_crop_refs"):
            sub.prop(scene, "h3img_crop_refs", text="Crop Refs to Subject")
        if scene.render.resolution_x * scene.render.resolution_y < 3 * 2**20:
            sub.label(text="Works best at 4-8 MP (e.g. 2720x1536, 3872x2176)", icon="INFO")
        try:
            from ...utils.helpers import ADDON_ID
            prefs = context.preferences.addons[ADDON_ID].preferences
        except Exception:
            return
        todo = self._missing_downloads(getattr(prefs, "hf_cache_dir", "") or None)
        if todo:
            box = col.box().column(align=True)
            for item in todo:
                box.label(text=f"Not downloaded yet: {item}", icon="IMPORT")
            if getattr(prefs, "local_files_only", False):
                box.label(text="'Use Local Files Only' is on: downloads are blocked", icon="ERROR")

    # ------------------------------------------------------------------ generate

    def _reference_images(self, inputs: ModelInputs, scene) -> list:
        from PIL import Image as _PILImage
        refs = [inputs.image.convert("RGB")] if inputs.image is not None else []
        crop = bool(getattr(scene, "h3img_crop_refs", False))
        for n in range(1, 10):
            path = getattr(scene, f"klein_strip_{n}_path", "")
            if not path:
                continue
            try:
                img = _PILImage.open(path).convert("RGB")
            except Exception as exc:
                print(f"P-H3 Image: reference '{path}' could not be opened: {exc}")
                continue
            if crop:  # Ref. pickers only — the image to edit keeps its framing
                try:
                    cropped = _crop_to_subject(img)
                except Exception as exc:
                    print(f"P-H3 Image: crop of Ref. {n} failed, using it whole: {exc}")
                    cropped = img
                if cropped is not img:
                    print(f"P-H3 Image: Ref. {n} cropped to subject {img.size} -> {cropped.size}")
                img = cropped
            refs.append(img)
        small = [i + 1 for i, img in enumerate(refs) if min(img.size) < _MIN_REF_SIDE]
        if small:
            print(f"P-H3 Image: skipping image(s) {small} — under {_MIN_REF_SIDE} px on a side")
            refs = [img for img in refs if min(img.size) >= _MIN_REF_SIDE]
        if len(refs) > _MAX_REFS:
            print(f"P-H3 Image: {len(refs)} images, keeping the first {_MAX_REFS}")
            refs = refs[:_MAX_REFS]
        return refs

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import time
        import torch
        import diffusers.modular_pipelines.p_h3_image.encoders as _enc
        from ..video.minimax_h3 import _queue_progress

        pipe = pipe_obj["pipe"]
        prompt = (inputs.prompt or "").strip()
        if not prompt:
            raise RuntimeError("P-H3 Image needs a prompt (it has no unconditional mode).")
        refs = self._reference_images(inputs, scene)
        width, height = max(64, int(inputs.width)), max(64, int(inputs.height))
        if refs and "<picture" not in prompt.lower():
            print("P-H3 Image: images are set but the prompt names none — the first image is edited; refer to "
                  "them as <Picture 1>, <Picture 2>, … for precise edits")

        kwargs = dict(prompt=prompt, height=height, width=width, output="images")
        if inputs.seed != 0:
            kwargs["generator"] = torch.Generator().manual_seed(inputs.seed)
        if refs:
            kwargs["image"] = refs
        workflow = f"edit, {len(refs)} image(s)" if refs else "text-to-image"
        print(f"P-H3 Image: {workflow}, {width}x{height} ({width * height / 2**20:.1f} MP)")
        self.set_phase(inputs, "Generating")

        timer = _Timer("job", _LABEL)
        parts = {}
        undo = [_Timer.wrap(_enc, "get_p_h3_image_prompt_embeds", parts, "text encode"),
                _Timer.wrap(pipe.vae, "encode", parts, "vae encode"),
                _Timer.wrap(pipe.vae, "decode", parts, "vae decode")]
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        timer.mark("prep")
        t = time.perf_counter()
        try:
            with _queue_progress(self, inputs):
                image = pipe(**kwargs)[0]
        finally:
            for u in reversed(undo):
                u()
        total = time.perf_counter() - t
        timer.add("text encode", parts.get("text encode", 0.0))
        if "vae encode" in parts:
            timer.add(f"vae encode ({len(refs)} img)", parts["vae encode"])
        timer.add("denoise 8 steps", total - sum(parts.values()))
        timer.add("vae decode", parts.get("vae decode", 0.0))
        if torch.cuda.is_available():
            timer.title += f", torch peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB"
        timer.report()
        return image.convert("RGB")
