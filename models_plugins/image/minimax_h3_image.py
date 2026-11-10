"""MiniMax-H3-Image: text-to-image and multi-reference image editing (6 steps, guidance-distilled).

Diffusers integrates it as Modular Diffusers blocks only (MiniMaxH3ImageBlocks); the workflow follows
the inputs:
  - prompt only                                -> text2image
  - the Input strip and/or Ref. strips set     -> image_conditioned (edit); the prompt names the
                                                  references in order as <Picture 1>, <Picture 2>, ...

The released transformer and text encoder are ~62 GB each in bf16, so this plugin assembles a
24 GB-card version from parts:
  - transformer: WaveCut/MiniMax-H3-Image-Pruned (the AdaLN projections stored as exact tables;
    images bit-identical to the original), quantized with SDNQ on the first load and saved to
    <HF cache>/pallaidium_sdnq so later loads skip the 40 GB bf16 read. The model class is
    vendored beside this file (_minimax_h3_image_pruned.py).
  - text encoder: the MiniMax-H3 Qwen3-VL conditioner, which the image model shares (diffusers
    docs), from OzzyGT's pre-quantized SDNQ MiniMax-H3 repos -- the same files the MiniMax H3
    video plugin uses, so a machine that runs H3 video downloads nothing new for it.
  - VAE, tokenizer, processor, scheduler: from the WaveCut repo (original files).

Sampling is fixed by the checkpoint: six steps, one output head per step, no CFG, no negative
prompt, no LoRA support in diffusers. Output size is any height x width; below "Min Generation
Area" the model generates on a larger canvas of the same aspect ratio and resizes back.

Prompt expansion runs MiniMax's official PE task prompts (MiniMaxAI/MiniMax-H3-Image-PE) on
Qwen3-VL-8B-Instruct, SDNQ-quantized on the fly. The official PE weights are a 27B NVFP4
checkpoint served by SGLang, which can't run in-process here; the task prompts are the same.
"""

import os

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram


_REPO = "WaveCut/MiniMax-H3-Image-Pruned"
_REPO_REVISION_NOTE = "3a804c0fc3e73a04606fd9618d947b08b5cfc970"  # vendored modeling file's revision
_TE_REPOS = {"sdnq4": "OzzyGT/MiniMax_H3_sdnq_dynamic_4bit", "sdnq8": "OzzyGT/MiniMax_H3_sdnq_dynamic_8bit"}
_SDNQ_DTYPE = {"sdnq4": "uint4", "sdnq8": "int8"}
# Total VRAM (GiB) to keep the quantized transformer resident; below it, it streams block by block.
_RESIDENT_GIB = {"sdnq4": 20, "sdnq8": 32}
# First-run downloads in GB: the bf16 transformer source, and each text encoder.
_TRANSFORMER_SOURCE_GB = 40
_TE_GB = {"sdnq4": 16, "sdnq8": 27}
_SMALL_COMPONENTS = ("tokenizer", "processor", "vae", "scheduler")
# A file each component needs in a cached snapshot: a snapshot folder can exist with only the
# index in it, and a tokenizer "loaded" from an empty folder comes back with no vocabulary.
_SMALL_FILES = {
    "tokenizer": "tokenizer/tokenizer.json",
    "processor": "processor/preprocessor_config.json",
    "vae": "vae/diffusion_pytorch_model.safetensors",
    "scheduler": "scheduler/scheduler_config.json",
}

# Area (pixels) below which the model generates on an upscaled canvas (diffusers' min_generation_area).
_MIN_AREA = {"4mp": 2048 * 2048, "2mp": 2048 * 1024, "1mp": 1024 * 1024, "off": 1024}
_MAX_REFS = 10   # the Input strip + 9 Ref. pickers
_MIN_REF_SIDE = 32

_PE_REPO = "MiniMaxAI/MiniMax-H3-Image-PE"
_PE_FILES = ("prompt/t2i/system_prompt.txt", "prompt/t2i/task_prompt.txt", "prompt/edit/task_prompt.txt")
_PE_VLM = "Qwen/Qwen3-VL-8B-Instruct"
_PE_VLM_FILES = ("config.json", "model-00004-of-00004.safetensors")
_PE_VLM_GB = 18
_PE_GIB = {"int8": 13, "uint4": 9}   # VRAM incl. activations for the long task prompt
_PE_IMAGE_PIXELS = 1024 * 1024       # the VLM only needs to see the references, not every pixel
_PE_MAX_NEW_TOKENS = 2048
_T2I_RATIOS = ("1:1", "5:4", "4:3", "3:2", "16:9", "21:9", "4:5", "3:4", "2:3", "9:16")


def _weights_of(scene) -> str:
    w = getattr(scene, "h3img_weights", "sdnq4")
    return w if w in _SDNQ_DTYPE else "sdnq4"


def _in_cache(repo: str, files, cache_dir) -> bool:
    """True when every file of *repo* is in the local HF cache (no network)."""
    try:
        from huggingface_hub import try_to_load_from_cache
        return all(isinstance(try_to_load_from_cache(repo, f, cache_dir=cache_dir or None), str) for f in files)
    except Exception:
        return True  # can't tell: don't block or warn


def _total_vram_gib() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 2**30
    except Exception:
        pass
    return 0.0


def _quant_dir(cache_dir, weights) -> str:
    from huggingface_hub import constants
    return os.path.join(cache_dir or constants.HF_HUB_CACHE, "pallaidium_sdnq",
                        f"MiniMax-H3-Image-Pruned_transformer_{_SDNQ_DTYPE[weights]}")


def _pe_quant_dir(cache_dir, quant) -> str:
    from huggingface_hub import constants
    return os.path.join(cache_dir or constants.HF_HUB_CACHE, "pallaidium_sdnq",
                        f"{_PE_VLM.replace('/', '--')}_{quant}")


_QUANT_MARKER = "pallaidium_complete.json"


def _quant_ready(path: str) -> bool:
    return os.path.isfile(os.path.join(path, _QUANT_MARKER))


def _check_materialized(model, label: str) -> None:
    """Fail on weights that never left the meta device (an incomplete load would crash mid-denoise)."""
    import itertools
    for name, tensor in itertools.chain(model.named_parameters(), model.named_buffers()):
        if tensor.is_meta:
            raise RuntimeError(f"{label}: '{name}' was not loaded (still on the meta device)")


def _reduced_ratio(width: int, height: int) -> str:
    import math
    g = math.gcd(int(width), int(height)) or 1
    return f"{int(width) // g}:{int(height) // g}"


def _size_for_ratio(ratio: str, area: int, multiple: int = 32):
    """(width, height) of *area* pixels at *ratio* ("W:H"), both sides rounded to *multiple*."""
    try:
        rw, rh = (float(p) for p in ratio.split(":"))
        if rw <= 0 or rh <= 0:
            raise ValueError
    except (ValueError, AttributeError):
        return None
    width = (area * rw / rh) ** 0.5
    height = width * rh / rw
    return (max(multiple, round(width / multiple) * multiple), max(multiple, round(height / multiple) * multiple))


def _parse_pe_answer(text: str):
    """(caption, ratio) from the PE answer: one JSON object, possibly after a think block or in a fence."""
    import json
    import re
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None, None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        # Usually an unescaped quote inside the caption: take everything up to the last delimiter.
        m = (re.search(r'"expanded_caption"\s*:\s*"(.*)"\s*,\s*"(?:target_)?aspect_ratio"', text, flags=re.S)
             or re.search(r'"expanded_caption"\s*:\s*"(.*)"\s*\}', text, flags=re.S))
        if not m:
            return None, None
        try:
            caption = json.loads(f'"{m.group(1)}"')
        except json.JSONDecodeError:
            caption = m.group(1).replace('\\"', '"').replace("\\n", "\n")
        data = {"expanded_caption": caption}
        r = re.search(r'"(?:target_)?aspect_ratio"\s*:\s*"([^"]+)"', text)
        if r:
            data["aspect_ratio"] = r.group(1)
    caption = data.get("expanded_caption")
    ratio = data.get("aspect_ratio") or data.get("target_aspect_ratio")
    caption = caption.strip() if isinstance(caption, str) else None
    ratio = ratio.strip() if isinstance(ratio, str) and re.fullmatch(r"\s*\d+\s*:\s*\d+\s*", ratio) else None
    return caption or None, (ratio.replace(" ", "") if ratio else None)


def _crop_to_subject(img, thresh=28, pad=0.06, max_frac=0.85):
    """Crop a reference shot on a plain backdrop (studio / character sheet) to its subject; return the
    image unchanged when the subject fills most of it (a scene).

    The pipeline scales each reference so its short edge is 1024 px, so a landscape character shot
    costs ~2.2 MP of reference tokens of mostly backdrop; cropped to a portrait it is ~1.4 MP with the
    subject larger. Measured 2026-11-10 (2 refs, 1920x896 studio shots, 4090): text encode 20 -> 15 s,
    denoise 53 -> 37 s, no loss of likeness. The backdrop is modelled as an 8x8 bilinear blur of the
    image, so soft gradients and vignettes count as backdrop."""
    import numpy as np
    from PIL import Image as _PILImage, ImageFilter
    small = img.resize((256, max(1, round(256 * img.height / img.width))), _PILImage.BILINEAR)
    a = np.asarray(small).astype(np.float32)
    h, w = a.shape[:2]
    bg = np.asarray(small.resize((8, 8), _PILImage.BILINEAR).resize((w, h), _PILImage.BILINEAR)).astype(np.float32)
    mask = np.abs(a - bg).max(axis=2) > thresh
    mask = np.asarray(_PILImage.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MedianFilter(5))) > 0
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return img
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    if (x1 - x0) * (y1 - y0) > max_frac * w * h:
        return img
    sx, sy = img.width / w, img.height / h
    px, py = pad * (x1 - x0) * sx, pad * (y1 - y0) * sy
    box = (max(0, int(x0 * sx - px)), max(0, int(y0 * sy - py)),
           min(img.width, int(x1 * sx + px)), min(img.height, int(y1 * sy + py)))
    if min(box[2] - box[0], box[3] - box[1]) < 256:
        return img
    return img.crop(box)


class _Timer:
    """Wall-clock timings of the load / job steps, printed as one summary line (GPU work is synced)."""

    def __init__(self, title):
        import time
        self.title, self.marks, self._t = title, [], time.perf_counter()

    def mark(self, label):
        import time
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        now = time.perf_counter()
        self.marks.append((label, now - self._t))
        self._t = now

    def add(self, label, seconds):
        self.marks.append((label, seconds))

    @staticmethod
    def wrap(obj, attr, store, key):
        """Patch obj.attr to add its (synced) wall time to store[key]; returns an undo callable."""
        import functools
        import time
        original = getattr(obj, attr)

        @functools.wraps(original)
        def timed(*args, **kwargs):
            import torch
            sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
            sync()
            t = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                sync()
                store[key] = store.get(key, 0.0) + time.perf_counter() - t

        setattr(obj, attr, timed)
        return lambda: setattr(obj, attr, original)

    def report(self):
        total = sum(s for _, s in self.marks)
        parts = ", ".join(f"{label} {s:.1f}s" for label, s in self.marks)
        print(f"MiniMax H3 Image timings ({self.title}): {parts} | total {total:.1f}s")


class MiniMaxH3ImagePlugin(ModelPlugin):
    MODEL_ID     = "MiniMaxAI/MiniMax-H3-Image"
    DISPLAY_NAME = "MiniMax H3 Image"
    MODEL_TYPE   = "image"
    DESCRIPTION  = (
        "MiniMax-H3-Image (6 steps, guidance-distilled): text-to-image and multi-reference editing "
        "(name references as <Picture 1>, <Picture 2>, ...). SDNQ 4/8-bit on the pruned checkpoint, "
        "optional prompt expansion with MiniMax's PE prompts"
    )
    INPUTS       = InputSpec.PROMPT | InputSpec.IMAGE
    # FRAMES = how long the image strip lasts on the timeline (the queue sets the strip duration).
    UI_SECTIONS  = [UISection.PROMPT, UISection.IMAGE_STRIP, UISection.RESOLUTION, UISection.FRAMES, UISection.SEED]
    # Steps and guidance are fixed by the checkpoint (6 distilled steps, no CFG); listed for the record.
    PARAMS       = ParamSpec(width=2048, height=2048, steps=6, guidance=1.0)
    REQUIRED_PACKAGES = ["torch", "diffusers", "transformers", "sdnq"]
    supports_inpaint  = False   # edits are instruction-driven through the references
    uses_strip_power  = False   # references condition the model; no denoise blend
    LOAD_VARIANT_KEYS = ("h3img_weights",)

    # ------------------------------------------------------------------ load

    def load(self, prefs, scene, **kw):
        import torch
        try:
            from diffusers import ModularPipeline, MiniMaxH3ImageTransformer2DModel  # noqa: F401
            import diffusers.modular_pipelines.minimax_h3_image  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "MiniMax H3 Image needs a newer diffusers with MiniMax-H3-Image support. Run Install "
                "Dependencies in the add-on Preferences, then restart Blender."
            ) from exc
        import sdnq  # noqa: F401 — registers the SDNQ quantizer with diffusers/transformers
        if kw.get("enabled_items"):
            print("MiniMax H3 Image: LoRAs are not supported by this model in diffusers — ignoring the LoRA list.")

        weights = _weights_of(scene)
        self._check_offline(prefs, weights)
        print(f"MiniMax H3 Image: loading (weights={weights})…")
        timer = _Timer("load")
        pipe = self._load_shell(prefs)
        timer.mark("index")
        self._load_small_components(pipe, prefs)
        timer.mark("vae+tokenizer+processor")
        text_encoder = self._load_text_encoder(weights, prefs)
        timer.mark("text encoder")
        transformer = self._load_transformer(weights, prefs)
        timer.mark("transformer")
        pipe.update_components(text_encoder=text_encoder, transformer=transformer)

        missing = [n for n in ("transformer", "text_encoder", "vae", "tokenizer", "processor", "scheduler")
                   if getattr(pipe, n, None) is None]
        if missing:
            raise RuntimeError(f"MiniMax H3 Image: failed to load {', '.join(missing)} (see the console above)")

        from sdnq.common import use_torch_compile as triton_is_available
        if triton_is_available and torch.cuda.is_available():
            from sdnq.loader import apply_sdnq_options_to_model
            pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
            pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)

        on_gpu = self._place(pipe, weights)
        timer.mark("placement")
        timer.report()
        return {"pipe": pipe, "weights": weights, "h3img_on_gpu": on_gpu, "h3img_pe_cache": {}}

    def _check_offline(self, prefs, weights):
        if not prefs.local_files_only:
            return
        todo = []
        if not _quant_ready(_quant_dir(prefs.hf_cache_dir, weights)) and not _in_cache(
                _REPO, ("transformer/diffusion_pytorch_model.safetensors.index.json",), prefs.hf_cache_dir):
            todo.append(f"transformer (~{_TRANSFORMER_SOURCE_GB} GB, quantized once)")
        if not _in_cache(_TE_REPOS[weights], ("text_encoder/model.safetensors",), prefs.hf_cache_dir):
            todo.append(f"text encoder (~{_TE_GB[weights]} GB)")
        if not _in_cache(_REPO, ("vae/diffusion_pytorch_model.safetensors",), prefs.hf_cache_dir):
            todo.append("VAE (~2.3 GB)")
        if todo:
            raise RuntimeError(
                "MiniMax H3 Image is not downloaded yet and 'Use Local Files Only' is on. Uncheck it in the "
                f"add-on Preferences to download: {', '.join(todo)}."
            )

    @staticmethod
    def _hub(prefs, local_only):
        return dict(cache_dir=prefs.hf_cache_dir or None, local_files_only=local_only)

    def _cache_first(self, prefs, fn):
        """Call fn(local_files_only) from the HF cache first; online only for what isn't cached."""
        try:
            return fn(True)
        except (OSError, ValueError):
            if prefs.local_files_only:
                raise
            return fn(False)

    def _load_shell(self, prefs):
        from diffusers import ModularPipeline
        return self._cache_first(prefs, lambda lfo: ModularPipeline.from_pretrained(_REPO, **self._hub(prefs, lfo)))

    def _load_small_components(self, pipe, prefs):
        """VAE, tokenizer, processor and scheduler of the WaveCut repo (original files).

        Cached components load from their snapshot folder: the processor's inner tokenizer probes a
        processor/config.json the repo doesn't have — a quick 404 online, a hard error local-only.
        """
        import torch
        from ..video.minimax_h3 import _cached_snapshot_dir
        cache_dir = prefs.hf_cache_dir or None
        for name in _SMALL_COMPONENTS:
            spec = pipe._component_specs.get(name)
            if spec is None:
                continue
            try:
                local_dir = _cached_snapshot_dir(spec.pretrained_model_name_or_path or _REPO, cache_dir)
            except OSError:
                continue
            if not os.path.isfile(os.path.join(local_dir, _SMALL_FILES[name])):
                continue  # snapshot folder exists, but this component was never downloaded
            try:
                pipe.load_components(names=name, pretrained_model_name_or_path=local_dir, dtype=torch.bfloat16,
                                     cache_dir=cache_dir, local_files_only=True)
            except Exception as exc:
                print(f"MiniMax H3 Image: cached {name} unusable ({exc}); fetching online")
        uncached = [n for n in _SMALL_COMPONENTS if getattr(pipe, n, None) is None]
        if uncached and not prefs.local_files_only:
            print(f"MiniMax H3 Image: not in local cache, fetching online: {uncached}")
            pipe.load_components(names=uncached, dtype=torch.bfloat16, cache_dir=cache_dir, local_files_only=False)
        tokenizer = getattr(pipe, "tokenizer", None)
        if tokenizer is not None and not tokenizer("a photo", add_special_tokens=False)["input_ids"]:
            raise RuntimeError("MiniMax H3 Image: the tokenizer loaded without a vocabulary (incomplete download?). "
                               f"Delete the {_REPO} folder from the HF cache and run again.")

    def _load_text_encoder(self, weights, prefs):
        import torch
        from transformers import Qwen3VLForConditionalGeneration
        repo = _TE_REPOS[weights]
        print(f"MiniMax H3 Image: text encoder from {repo} (shared with MiniMax H3 video)")
        te = self._cache_first(prefs, lambda lfo: Qwen3VLForConditionalGeneration.from_pretrained(
            repo, subfolder="text_encoder", dtype=torch.bfloat16, **self._hub(prefs, lfo)))
        layers = te.config.text_config.num_hidden_layers
        if layers <= 50:
            raise RuntimeError(f"MiniMax H3 Image: the text encoder has {layers} layers; hidden_states[50] needs 51+")
        return te.eval()

    def _load_transformer(self, weights, prefs):
        """The SDNQ transformer: the saved quantized copy when there is one, else quantize the bf16 source."""
        import shutil
        import torch
        from ._minimax_h3_image_pruned import MiniMaxH3ImagePrunedTransformer2DModel as Transformer

        qdir = _quant_dir(prefs.hf_cache_dir, weights)
        if _quant_ready(qdir):
            try:
                model = Transformer.from_pretrained(qdir, dtype=torch.bfloat16, local_files_only=True)
                _check_materialized(model, "MiniMax H3 Image transformer")
                print(f"MiniMax H3 Image: SDNQ {_SDNQ_DTYPE[weights]} transformer from {qdir}")
                return model.eval()
            except Exception as exc:
                print(f"MiniMax H3 Image: saved SDNQ transformer unusable ({exc}); quantizing it again")
                shutil.rmtree(qdir, ignore_errors=True)

        from sdnq import SDNQConfig
        on_cuda = gfx_device == "cuda" and torch.cuda.is_available()
        print(f"MiniMax H3 Image: quantizing the transformer to {_SDNQ_DTYPE[weights]} — once; the bf16 source "
              f"is ~{_TRANSFORMER_SOURCE_GB} GB on first download")
        # Attention and feed-forward of the blocks and the text refiner are quantized; the patch
        # projection, text embedder, output heads, norms, RoPE and the AdaLN tables stay as released
        # (the recipe of OzzyGT's MiniMax-H3 SDNQ repos).
        cfg = SDNQConfig(
            weights_dtype=_SDNQ_DTYPE[weights],
            quantization_device="cuda" if on_cuda else None,
            return_device="cpu",
            modules_to_not_convert=["proj_in", "context_embedder", "pdd_proj_out", "norm_out", "rope"],
        )
        model = self._cache_first(prefs, lambda lfo: Transformer.from_pretrained(
            _REPO, subfolder="transformer", dtype=torch.bfloat16, quantization_config=cfg, **self._hub(prefs, lfo)))
        _check_materialized(model, "MiniMax H3 Image transformer")
        self._save_quantized(model, qdir, {"source": _REPO, "revision": _REPO_REVISION_NOTE, "weights": weights},
                             "transformer")
        return model.eval()

    @staticmethod
    def _save_quantized(model, qdir, marker, label):
        """Save a quantized model next to the HF cache so later loads skip the bf16 read + quantization."""
        import json
        import shutil
        tmp = qdir + ".partial"
        try:
            shutil.rmtree(tmp, ignore_errors=True)
            os.makedirs(os.path.dirname(qdir), exist_ok=True)
            from sdnq.loader import save_sdnq_model
            save_sdnq_model(model, tmp)
            with open(os.path.join(tmp, _QUANT_MARKER), "w", encoding="utf-8") as fh:
                json.dump(marker, fh)
            shutil.rmtree(qdir, ignore_errors=True)
            os.replace(tmp, qdir)
            print(f"MiniMax H3 Image: saved the quantized {label} to {qdir}")
        except Exception as exc:
            shutil.rmtree(tmp, ignore_errors=True)
            print(f"MiniMax H3 Image: could not save the quantized {label} ({exc}); "
                  "the next load quantizes it again")

    def _place(self, pipe, weights) -> bool:
        """Device placement; True when the transformer is resident on the GPU."""
        import torch
        if gfx_device == "mps":
            pipe.to("mps")
            print("MiniMax H3 Image: on Apple GPU (MPS)")
            return True
        if gfx_device != "cuda":
            print("MiniMax H3 Image: on CPU (no CUDA GPU found) — expect very slow generation")
            return False

        from diffusers.hooks import apply_group_offloading
        from ...utils.helpers import ensure_group_offload_keep_cpu_weights
        from ..video.minimax_h3 import _onload_vae_per_call
        from ..video._h3_conditioner import install_fast_patch_embed, install_lean_conditioner

        onload, offload = torch.device("cuda"), torch.device("cpu")
        # Unstreamed group offload reusing the original CPU tensors: pinned host memory and the
        # GPU->CPU copy-back both crash or leak on this Windows/Blender stack (see minimax_h3.py).
        ensure_group_offload_keep_cpu_weights()
        opts = dict(onload_device=onload, offload_device=offload, use_stream=False, low_cpu_mem_usage=True)
        # The conditioner runs once per job; diffusers calls text_encoder.model directly. Leaf level on
        # purpose: block-level groups of 5 decoder layers (measured 2026-11-10, 2-reference edit at
        # 1920x896) pushed the 24 GB card into shared memory — text encode 285 s, denoise 49 s.
        apply_group_offloading(pipe.text_encoder.model, offload_type="leaf_level", **opts)
        resident = not low_vram() and _total_vram_gib() >= _RESIDENT_GIB[weights]
        if resident:
            pipe.transformer.to(onload)
            placement = "transformer resident on GPU"
        else:
            # The pruned model's last block calls its submodules directly (skipping block.forward and
            # so the offload hook); the full last block is bit-identical, only slightly slower.
            pipe.transformer.skip_dead_rows = False
            pipe.transformer.enable_group_offload(offload_type="block_level", num_blocks_per_group=2, **opts)
            placement = f"transformer streamed block by block ({_total_vram_gib():.0f} GB VRAM)"
        # VAE weights onload only inside encode()/decode() (group offload hooks forward(), which the
        # blocks never call).
        _onload_vae_per_call(pipe.vae, onload)
        # Qwen3-VL conditioner: query-chunked attention + stop at hidden_states[50] (diffusers'
        # pass spikes past 20 GB with reference images in the presentation).
        install_lean_conditioner("diffusers.modular_pipelines.minimax_h3_image.encoders")
        # Reference images: bf16 Conv3d patch embed ran 240 s for two references; a matmul is ~2 ms.
        install_fast_patch_embed(pipe.text_encoder)
        print(f"MiniMax H3 Image: {placement}; text encoder offloaded")
        return resident

    # ------------------------------------------------------------------ UI

    def draw_custom_ui(self, col, context) -> bool:
        scene = context.scene
        # Strip refs live in the scene shown in the VSE (context.sequencer_scene in Blender 5.x).
        vse_scene = getattr(context, "sequencer_scene", None) or scene
        try:
            col.prop(scene, "input_strips", text="Input")
        except Exception:
            pass
        if vse_scene.sequence_editor is None:
            return True
        for i in range(1, scene.klein_visible_strips + 1):
            row = col.row(align=True)
            row.prop_search(
                vse_scene, f"klein_strip_{i}", vse_scene.sequence_editor, "strips",
                text="Ref.", icon="FILE_IMAGE",
            )
            row.operator("sequencer.strip_picker", text="", icon="EYEDROPPER").action = f"klein_select{i}"
            if i == scene.klein_visible_strips and scene.klein_visible_strips < 9:
                if scene.klein_visible_strips > 3:
                    row.operator("object.klein_hide_strip", text="", icon="REMOVE").strip_index = i
                row.operator("object.klein_add_strip", text="", icon="ADD")
        if any(getattr(vse_scene, f"klein_strip_{i}", "") for i in range(1, 10)) or \
                getattr(scene, "input_strips", "") == "input_strips":
            col.label(text="Name references <Picture 1>, <Picture 2>… in the prompt", icon="INFO")
        return True

    def draw_model_options_ui(self, col, context):
        """Weights, generation area and prompt expansion, between the HF token and Batch Count."""
        scene = context.scene
        sub = col.column(align=True)
        sub.prop(scene, "h3img_weights", text="Weights")
        sub.prop(scene, "h3img_min_area", text="Min Area")
        sub.prop(scene, "h3img_crop_refs", text="Crop Refs to Subject")
        sub.prop(scene, "h3img_prompt_expansion", text="Prompt Expansion")
        pe = sub.column(align=True)
        pe.active = scene.h3img_prompt_expansion
        pe.prop(scene, "h3img_pe_aspect", text="Use Suggested Aspect Ratio")
        pe.prop(scene, "h3img_pe_quant", text="Expander Quant")
        self._draw_download_info(col, context)

    def _draw_download_info(self, col, context):
        try:
            from ...utils.helpers import ADDON_ID
            prefs = context.preferences.addons[ADDON_ID].preferences
        except Exception:
            return
        scene = context.scene
        cache_dir = getattr(prefs, "hf_cache_dir", "") or None
        weights = _weights_of(scene)
        notes = []
        try:
            if not _quant_ready(_quant_dir(cache_dir, weights)):
                notes.append(f"First load: ~{_TRANSFORMER_SOURCE_GB} GB, quantized once to {_SDNQ_DTYPE[weights]}")
        except Exception:
            pass
        if not _in_cache(_TE_REPOS[weights], ("text_encoder/model.safetensors",), cache_dir):
            notes.append(f"Text encoder not downloaded: ~{_TE_GB[weights]} GB")
        if scene.h3img_prompt_expansion and not _quant_ready(_pe_quant_dir(cache_dir, scene.h3img_pe_quant)) \
                and not _in_cache(_PE_VLM, _PE_VLM_FILES, cache_dir):
            notes.append(f"Prompt expander not downloaded: ~{_PE_VLM_GB} GB")
        if not notes:
            return
        box = col.box().column(align=True)
        for note in notes:
            box.label(text=note, icon="IMPORT")
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
                print(f"MiniMax H3 Image: reference '{path}' could not be opened: {exc}")
                continue
            if crop:  # Ref. pickers only — the Input image keeps its framing (edits of Picture 1 need it)
                try:
                    cropped = _crop_to_subject(img)
                except Exception as exc:
                    print(f"MiniMax H3 Image: crop of Ref. {n} failed, using it whole: {exc}")
                    cropped = img
                if cropped is not img:
                    print(f"MiniMax H3 Image: Ref. {n} cropped to subject {img.size} -> {cropped.size}")
                img = cropped
            refs.append(img)
        small = [i + 1 for i, img in enumerate(refs) if min(img.size) < _MIN_REF_SIDE]
        if small:
            print(f"MiniMax H3 Image: skipping reference(s) {small} — under {_MIN_REF_SIDE} px on a side")
            refs = [img for img in refs if min(img.size) >= _MIN_REF_SIDE]
        if len(refs) > _MAX_REFS:
            print(f"MiniMax H3 Image: {len(refs)} references, keeping the first {_MAX_REFS}")
            refs = refs[:_MAX_REFS]
        return refs

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch
        from ..video.minimax_h3 import _queue_progress

        pipe = pipe_obj["pipe"]
        prompt = (inputs.prompt or "").strip()
        if not prompt:
            raise RuntimeError("MiniMax H3 Image needs a prompt (it has no unconditional mode).")
        refs = self._reference_images(inputs, scene)
        width, height = max(64, int(inputs.width)), max(64, int(inputs.height))
        timer = _Timer("job")

        if getattr(scene, "h3img_prompt_expansion", False):
            expanded, ratio = self._expand(pipe_obj, prompt, refs, width, height, scene, prefs, inputs, timer)
            if expanded:
                prompt = expanded
            if ratio and getattr(scene, "h3img_pe_aspect", False):
                size = _size_for_ratio(ratio, width * height)
                if size and size != (width, height):
                    print(f"MiniMax H3 Image: suggested aspect {ratio} -> {size[0]}x{size[1]}")
                    width, height = size
        if refs and "<picture" not in prompt.lower():
            print("MiniMax H3 Image: references are set but the prompt names none — refer to them as "
                  "<Picture 1>, <Picture 2>, … for precise edits")
        if prompt != inputs.prompt.strip():
            print(f"MiniMax H3 Image final prompt: {prompt}")
        try:
            scene.h3img_final_prompt = prompt  # read back into the strip metadata
        except Exception:
            pass

        min_area = _MIN_AREA.get(getattr(scene, "h3img_min_area", "4mp"), _MIN_AREA["4mp"])
        kwargs = dict(prompt=prompt, height=height, width=width, min_generation_area=min_area, output="images")
        if inputs.seed != 0:
            kwargs["generator"] = torch.Generator().manual_seed(inputs.seed)
        if refs:
            kwargs["image"] = refs
        workflow = f"edit, {len(refs)} reference(s)" if refs else "text-to-image"
        print(f"MiniMax H3 Image: {workflow}, {width}x{height}, min area {min_area / 2**20:.1f} MP")
        self.set_phase(inputs, "Generating")
        # Time the pipeline's parts: the conditioner (looked up as a module global at call time),
        # the VAE calls, and the rest (canvas/layout prep + the 6 denoise steps).
        import time
        import diffusers.modular_pipelines.minimax_h3_image.encoders as _enc
        parts = {}
        undo = [_Timer.wrap(_enc, "get_qwen3vl_prompt_embeds", parts, "text encode"),
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
            for u in reversed(undo):  # outermost patch first
                u()
        total = time.perf_counter() - t
        timer.add("text encode", parts.get("text encode", 0.0))
        if "vae encode" in parts:
            timer.add(f"vae encode ({len(refs)} ref)", parts["vae encode"])
        timer.add("denoise 6 steps", total - sum(parts.values()))
        timer.add("vae decode", parts.get("vae decode", 0.0))
        if torch.cuda.is_available():
            timer.title += f", torch peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB"
        timer.report()
        return image.convert("RGB")

    # ------------------------------------------------------------------ prompt expansion

    def _expand(self, pipe_obj, prompt, refs, width, height, scene, prefs, inputs, timer=None):
        """(expanded prompt or None, suggested ratio or None). Cached per request within the loaded model,
        so a batch expands once."""
        import gc
        import hashlib
        import torch

        quant = getattr(scene, "h3img_pe_quant", "int8")
        aspect = "adaptive" if getattr(scene, "h3img_pe_aspect", False) else _reduced_ratio(width, height)
        digest = hashlib.md5()
        for img in refs:
            digest.update(f"{img.size}".encode())
            digest.update(img.tobytes())
        key = (prompt, digest.hexdigest(), quant, aspect)
        cache = pipe_obj.setdefault("h3img_pe_cache", {})
        if key in cache:
            print("MiniMax H3 Image: reusing the expanded prompt of the previous job")
            return cache[key]

        timer = timer or _Timer("expansion")
        self.set_phase(inputs, "Expanding prompt")
        timer.mark("prep")
        pipe = pipe_obj["pipe"]
        parked = False
        if pipe_obj.get("h3img_on_gpu") and torch.cuda.is_available():
            free = torch.cuda.mem_get_info()[0] / 2**30
            if free < _PE_GIB.get(quant, 13):
                print(f"MiniMax H3 Image: {free:.1f} GiB free — moving the transformer to CPU for the expander")
                pipe.transformer.to("cpu")
                gc.collect()
                torch.cuda.empty_cache()
                parked = True
                timer.mark("park transformer")
        pe = None
        try:
            pe = self._load_pe(quant, prefs)
            timer.mark("load expander")
            answer = self._run_pe(pe, prompt, refs, aspect, inputs)
            timer.mark("expand")
        except Exception as exc:
            print(f"MiniMax H3 Image: prompt expansion failed ({exc}) — using the prompt as typed")
            answer = ""
        finally:
            del pe
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if parked:
                pipe.transformer.to("cuda")
                timer.mark("unpark transformer")

        caption, ratio = _parse_pe_answer(answer)
        if not caption:
            if answer:
                print(f"MiniMax H3 Image: the expander's answer had no caption — using the prompt as typed\n{answer[:500]}")
            return None, None
        if not refs and ratio not in _T2I_RATIOS:
            ratio = None
        cache.clear()  # one entry is enough for a batch; don't grow across prompts
        cache[key] = (caption, ratio)
        return caption, ratio

    def _load_pe(self, quant: str, prefs) -> dict:
        """The expander VLM: the saved quantized copy when there is one (~9/5 GB read), else quantize the
        17.5 GB bf16 weights and save them for the next job."""
        import shutil
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from sdnq import SDNQConfig

        hub = dict(cache_dir=prefs.hf_cache_dir or None, local_files_only=prefs.local_files_only)
        prompts = {}
        for name in _PE_FILES:
            with open(hf_hub_download(_PE_REPO, name, **hub), encoding="utf-8") as fh:
                prompts[name] = fh.read()
        on_cuda = torch.cuda.is_available()
        qdir = _pe_quant_dir(prefs.hf_cache_dir, quant)
        model = None
        if _quant_ready(qdir):
            try:
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    qdir, dtype=torch.bfloat16, local_files_only=True).eval()
                _check_materialized(model, "prompt expander")
                print(f"MiniMax H3 Image: prompt expander ({quant}) from {qdir}")
            except Exception as exc:
                print(f"MiniMax H3 Image: saved prompt expander unusable ({exc}); quantizing it again")
                model = None
                shutil.rmtree(qdir, ignore_errors=True)
        if model is None:
            if prefs.local_files_only and not _in_cache(_PE_VLM, _PE_VLM_FILES, prefs.hf_cache_dir):
                raise RuntimeError(f"{_PE_VLM} is not downloaded and 'Use Local Files Only' is on "
                                   f"(~{_PE_VLM_GB} GB, once)")
            print(f"MiniMax H3 Image: quantizing {_PE_VLM} to {quant} for prompt expansion — once")
            cfg = SDNQConfig(
                weights_dtype=quant,
                quantization_device="cuda" if on_cuda else None,
                return_device="cuda" if on_cuda else "cpu",
                modules_to_not_convert=["visual"],  # small, and reads the references
            )
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                _PE_VLM, dtype=torch.bfloat16, quantization_config=cfg, **hub).eval()
            _check_materialized(model, "prompt expander")
            self._save_quantized(model, qdir, {"source": _PE_VLM, "weights": quant}, "prompt expander")
        if on_cuda:
            # SDNQ returns only the quantized layers to the GPU; embeddings, norms and the
            # (unquantized) vision tower are still on the CPU.
            model.to("cuda")
        # Edit expansion reads the references through the same kind of vision tower as the conditioner.
        from ...utils.fast_patch_embed import install_fast_patch_embed
        install_fast_patch_embed(model, "MiniMax H3 Image prompt expander")
        processor = AutoProcessor.from_pretrained(_PE_VLM, **hub)
        return {"model": model, "processor": processor, "prompts": prompts}

    @staticmethod
    def _run_pe(pe, prompt, refs, aspect, inputs) -> str:
        """MiniMax's PE request: the t2i system + task prompt, or the edit task prompt with labelled references."""
        import json
        import torch
        from PIL import Image as _PILImage
        from transformers import StoppingCriteria, StoppingCriteriaList

        model, processor, prompts = pe["model"], pe["processor"], pe["prompts"]
        if refs:
            n = len(refs)
            content = [
                {"type": "text", "text": prompts["prompt/edit/task_prompt.txt"]},
                {"type": "text", "text": f"Reference Image Count:\n{n} input image(s), labeled <Picture 1> "
                                         f"through <Picture {n}>."},
            ]
            for i, img in enumerate(refs, start=1):
                ratio = _reduced_ratio(*img.size)
                pixels = img.width * img.height
                if pixels > _PE_IMAGE_PIXELS:
                    scale = (_PE_IMAGE_PIXELS / pixels) ** 0.5
                    img = img.resize((max(32, round(img.width * scale)), max(32, round(img.height * scale))),
                                     _PILImage.Resampling.LANCZOS)
                content += [
                    {"type": "text", "text": f"Reference Media:\n- <Picture {i}> [aspect_ratio {ratio}]"},
                    {"type": "image", "image": img},
                ]
            content.append({"type": "text", "text": (
                "\nUser Editing Instruction (JSON string):\n"
                f"{json.dumps(prompt, ensure_ascii=False)}\n\n"
                f"User Aspect Ratio Selection:\n{aspect}\n\n"
                "Apply the instructions above to the supplied inputs and reference images. "
                "Output only the required two-field JSON object."
            )})
            messages = [{"role": "user", "content": content}]
        else:
            system = prompts["prompt/t2i/system_prompt.txt"].rstrip("\n")
            user = prompts["prompt/t2i/task_prompt.txt"].format(prompt)
            messages = [{"role": "system", "content": [{"type": "text", "text": system}]},
                        {"role": "user", "content": [{"type": "text", "text": user}]}]

        enc = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
        ).to(model.device)
        progress_fn = inputs.progress_fn
        max_new = _PE_MAX_NEW_TOKENS

        class _Progress(StoppingCriteria):
            # Never stops generation; reports tokens written so far.
            def __init__(self, start):
                self.start = start

            def __call__(self, input_ids, scores, **kw):
                if progress_fn is not None:
                    progress_fn(min(input_ids.shape[1] - self.start, max_new), max_new)
                return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        start = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 stopping_criteria=StoppingCriteriaList([_Progress(start)]))
        return processor.decode(out[0, start:], skip_special_tokens=True).strip()
