"""MOSS-Image (OpenMOSS): 5B single-stream DiT text-to-image with strong text
rendering (English and Chinese) and bounding-box layout prompts, plus img2img
and inpaint built on the same sampler.

Two checkpoints, one plugin class each:
  - OpenMOSS-Team/MOSS-Image-Base  the base model (what LoRAs are trained on)
  - OpenMOSS-Team/MOSS-Image-SFT   fine-tuned on curated data (cleaner layout and text)
Both ship the Qwen3.5-9B text encoder, tokenizer, FLUX.2 VAE and model code bit
for bit identical, so those always load from the Base repo: one ~17 GB text
encoder download, not two. Only the transformer differs.

The hub pipeline (trust_remote_code) is text-to-image only and takes no step
callback. This plugin uses its transformer, scheduler and encode_prompt() but
runs the Euler/CFG loop itself, matching the pipeline exactly for txt2img,
so img2img, inpaint, queue progress and cancel work too.

Weights are bf16 on the Hub. scene.mossimg_weights quantizes them on the fly
with SDNQ; LoRAs are fused into the bf16 transformer first, so they work with
every weight choice. Layout: a compiled MOSS layout prompt passes through, a
MOSS layout draft (abs/xyxy, as written by the official prompt-enhancer skill)
is compiled, and Box Editor JSON or a FLUX.3 layout prompt is converted.
"""

import json
import math
import os
import re

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram


_SHARED_REPO = "OpenMOSS-Team/MOSS-Image-Base"  # text encoder, tokenizer, VAE, model code
_CODE_FILE   = "pipeline_moss_image.py"

# Files that must be in the HF cache before an offline load, and download sizes.
_SHARED_FILES = (
    "text_encoder/model-00001-of-00004.safetensors",
    "text_encoder/model-00002-of-00004.safetensors",
    "text_encoder/model-00003-of-00004.safetensors",
    "text_encoder/model-00004-of-00004.safetensors",
    "tokenizer/tokenizer.json",
    "vae/diffusion_pytorch_model.safetensors",
    _CODE_FILE,
)
_TRANSFORMER_FILES = (
    "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
)
_SHARED_GB      = 17
_TRANSFORMER_GB = 10

# (transformer, text encoder) precision per weight choice. The text encoder is
# never kept in bf16: 19 GB on top of the transformer exceeds most system RAM.
_WEIGHTS = {
    "sdnq8": ("int8", "int8"),
    "sdnq4": ("uint4", "uint4"),
    "bf16":  (None, "int8"),
}
# Total VRAM (GiB) to keep the whole pipeline resident; below it the components
# are moved to the GPU one at a time (whole-component CPU offload).
_RESIDENT_GIB = {"sdnq8": 22, "sdnq4": 14, "bf16": 30}
# Transformer layers that stay in bf16 when quantizing: the small,
# precision-sensitive embedders/heads, and each block's q projection: the hub
# code casts the block's activations to proj['q'].weight.dtype, which would be
# int8/uint8 once quantized (~0.4 GB extra, no patching of the hub code).
# SDNQ matches these against parameter names ("text_embedder.proj.weight"):
# plain names match a path component, a leading "." makes it a prefix match.
_KEEP_BF16 = [
    "video_embedder", ".text_embedder.proj", "time_embedding",
    "shared_block_modulation", "video_head_modulation", "final_linear_video",
    "q",
]
# GPUs with at least this much VRAM (GiB) load the bf16 transformer straight
# into VRAM and quantize it there instead of staging 10 GB in system RAM.
_GPU_STAGE_GIB = 16
# Above this many pixels the fp32 VAE decodes in tiles.
_VAE_TILE_PIXELS = 2_200_000


def _in_cache(repo: str, files, cache_dir) -> bool:
    """True when every file of *repo* is in the local HF cache (no network)."""
    try:
        from huggingface_hub import try_to_load_from_cache
        return all(isinstance(try_to_load_from_cache(repo, f, cache_dir=cache_dir or None), str)
                   for f in files)
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


def _parse_layers(text: str, depth: int) -> tuple:
    """"22, 24" -> (22, 24); entries outside the model's block range are dropped."""
    layers = []
    for part in re.split(r"[,\s;]+", text or ""):
        if not part:
            continue
        try:
            index = int(part)
        except ValueError:
            print(f"MOSS-Image: skip layer '{part}' is not a number — ignored")
            continue
        if 0 <= index < depth:
            layers.append(index)
        else:
            print(f"MOSS-Image: skip layer {index} is outside 0-{depth - 1} — ignored")
    return tuple(dict.fromkeys(layers))


# ---------------------------------------------------------------------------
# Layout prompts
# ---------------------------------------------------------------------------
# The model reads a compact JSON object:
#   {"summary": {"meta", "canvas", "caption", "palette"},
#    "items": [{"label": object|character|text, "caption", "text", "bbox", ...}],
#    "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}
# with boxes as [y_min, x_min, y_max, x_max] in pixels, origin at the canvas
# centre — so they depend on the output size and are computed per job.

_ITEM_KEYS    = ("label", "caption", "text", "bbox", "tags", "palette")
_SUMMARY_KEYS = ("meta", "canvas", "caption", "palette")
_FLUX3_TAG_RE = re.compile(r"\s*<[A-Za-z]+_\d+>")


def _parse_json(text: str):
    """The JSON object in *text* (``//`` comment lines skipped), or None."""
    body = "\n".join(l for l in (text or "").splitlines() if not l.strip().startswith("//")).strip()
    if not body.startswith("{"):
        return None
    try:
        data = json.loads(body)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _centered(x1, y1, x2, y2, width, height) -> list:
    """Top-left-origin pixel box -> centred yxyx, as the official compiler does
    (floor left/top, ceil right/bottom, then subtract the centre)."""
    x1, x2 = sorted((min(max(float(x1), 0.0), width), min(max(float(x2), 0.0), width)))
    y1, y2 = sorted((min(max(float(y1), 0.0), height), min(max(float(y2), 0.0), height)))
    box = [math.floor(round(y1, 9)) - height // 2, math.floor(round(x1, 9)) - width // 2,
           math.ceil(round(y2, 9)) - height // 2, math.ceil(round(x2, 9)) - width // 2]
    box[2] = max(box[2], box[0] + 1)
    box[3] = max(box[3], box[1] + 1)
    return box


def _compile(summary: dict, items: list) -> str:
    return json.dumps(
        {"summary": summary, "items": items,
         "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"},
        ensure_ascii=False, separators=(",", ":"),
    )


def _from_draft(data: dict, width: int, height: int) -> str:
    """MOSS layout draft (abs pixels, xyxy, its own width/height) -> model prompt.
    A draft made for another canvas size is rescaled to the output size."""
    sx = width / float(data.get("width") or width)
    sy = height / float(data.get("height") or height)
    items = []
    for item in data.get("items") or []:
        box = item.get("bbox") if isinstance(item, dict) else None
        if not (isinstance(box, list) and len(box) == 4):
            continue
        x1, y1, x2, y2 = box
        items.append({
            k: (_centered(x1 * sx, y1 * sy, x2 * sx, y2 * sy, width, height) if k == "bbox" else v)
            for k, v in item.items() if k in _ITEM_KEYS
        })
    summary = {k: v for k, v in (data.get("summary") or {}).items() if k in _SUMMARY_KEYS}
    return _compile(summary, items)


def _from_box_editor(data: dict, width: int, height: int) -> str:
    """Box Editor / Ideogram JSON ([top, left, bottom, right] on 0-1000) -> model prompt."""
    comp = data.get("compositional_deconstruction") or {}
    style = data.get("style_description") or {}
    caption = _FLUX3_TAG_RE.sub("", (data.get("high_level_description") or "")).strip()
    background = (comp.get("background") or "").strip()
    if background:
        caption = f"{caption.rstrip('. ')}. Background: {background}" if caption else f"Background: {background}"
    look = [style.get(k) for k in ("photo", "aesthetics", "lighting")]
    look = [s.strip() for s in look if isinstance(s, str) and s.strip()]
    if look:
        caption = f"{caption.rstrip('. ')}. Style: {'; '.join(look)}." if caption else f"Style: {'; '.join(look)}."

    summary = {}
    meta = [style.get(k) for k in ("medium", "art_style")]
    meta = [s.strip() for s in meta if isinstance(s, str) and s.strip()]
    if meta:
        summary["meta"] = meta
    summary["canvas"] = "single"
    summary["caption"] = caption or "An image."
    if style.get("color_palette"):
        summary["palette"] = [str(c) for c in style["color_palette"]]

    items = []
    for elem in comp.get("elements") or []:
        if not isinstance(elem, dict) or elem.get("remove"):
            continue
        top, left, bottom, right = (list(elem.get("bbox") or [0, 0, 1000, 1000]) + [1000] * 4)[:4]
        desc = (elem.get("desc") or "").strip()
        text = (elem.get("text") or "").strip()
        is_text = elem.get("type") == "text" and bool(text)
        item = {"label": "text" if is_text else "object",
                "caption": desc or (f'The text "{text}".' if is_text else "An element.")}
        if is_text:
            item["text"] = text
        item["bbox"] = _centered(left * width / 1000, top * height / 1000,
                                 right * width / 1000, bottom * height / 1000, width, height)
        if elem.get("color_palette"):
            item["palette"] = [str(c) for c in elem["color_palette"]]
        items.append(item)
    return _compile(summary, items) if items else summary["caption"]


def to_moss_prompt(prompt: str, width: int, height: int) -> str:
    """Turn any supported layout format into the model's layout prompt; plain
    text (and anything unrecognized) passes through unchanged."""
    data = _parse_json(prompt)
    if data is not None:
        if data.get("bbox_coord_space") == "abs_centralized":
            return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        if isinstance(data.get("summary"), dict) and isinstance(data.get("items"), list):
            return _from_draft(data, width, height)
        if isinstance(data.get("compositional_deconstruction"), dict):
            return _from_box_editor(data, width, height)
        if set(data) == {"caption"} and isinstance(data["caption"], str):
            return data["caption"]
        return prompt
    try:
        from ...utils.flux3_layout import split_flux3_layout, flux3_to_ideogram
        _caption, regions = split_flux3_layout(prompt)
        if regions:
            data = flux3_to_ideogram(prompt)
            if data:
                return _from_box_editor(data, width, height)
    except Exception as exc:
        print(f"MOSS-Image: FLUX.3 layout conversion failed ({exc}) — using the prompt as typed")
    return prompt


# ---------------------------------------------------------------------------
# LoRA fusing
# ---------------------------------------------------------------------------

_LORA_PART_RE = re.compile(
    r"^(?P<base>.+?)\.(?:lora_(?P<ab>[AB])(?:\.[^.]+)?\.weight|lora[._](?P<du>down|up)\.weight|(?P<alpha>alpha))$"
)
_LORA_PREFIXES = ("base_model.model.", "diffusion_model.", "transformer.", "model.",
                  "lora_unet_", "lora_transformer_")


def _fuse_loras(transformer, enabled_items, folder: str) -> None:
    """Merge LoRA deltas (W += weight * alpha/rank * B @ A) into the bf16 transformer.

    Format-agnostic: diffusers/PEFT (lora_A/lora_B, metadata alpha), kohya
    (lora_down/lora_up, per-layer .alpha, underscore module names) and the
    common diffusion_model./transformer. prefixes. Fusing before quantization
    makes LoRAs work with the SDNQ weights too.
    """
    import torch
    from safetensors import safe_open

    linears = {n: m for n, m in transformer.named_modules() if isinstance(m, torch.nn.Linear)}
    by_underscore = {n.replace(".", "_"): n for n in linears}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for item in enabled_items:
        path = os.path.join(folder, item.name + ".safetensors")
        with safe_open(path, framework="pt") as fh:
            meta = fh.metadata() or {}
            state = {k: fh.get_tensor(k) for k in fh.keys()}
        default_alpha = None
        try:
            cfg = json.loads(meta.get("lora_adapter_metadata") or "{}")
            cfg = cfg.get("transformer", cfg) if isinstance(cfg, dict) else {}
            if cfg.get("lora_alpha") and cfg.get("r"):
                default_alpha = (float(cfg["lora_alpha"]), int(cfg["r"]))
        except Exception:
            pass

        groups, ignored = {}, 0
        for key, tensor in state.items():
            m = _LORA_PART_RE.match(key)
            if not m:
                ignored += 1
                continue
            base = m.group("base")
            for prefix in _LORA_PREFIXES:
                if base.startswith(prefix):
                    base = base[len(prefix):]
            name = base if base in linears else by_underscore.get(base)
            if name is None:
                ignored += 1
                continue
            part = ("down" if m.group("ab") == "A" else "up") if m.group("ab") else (m.group("du") or "alpha")
            groups.setdefault(name, {})[part] = tensor

        fused = 0
        with torch.no_grad():
            for name, g in groups.items():
                if "down" not in g or "up" not in g:
                    continue
                down = g["down"].to(device, torch.float32).flatten(1)
                up = g["up"].to(device, torch.float32).flatten(1)
                rank = down.shape[0]
                if "alpha" in g:
                    scale = float(g["alpha"]) / rank
                elif default_alpha is not None:
                    scale = default_alpha[0] / default_alpha[1]
                else:
                    scale = 1.0
                weight = linears[name].weight
                if tuple(weight.shape) != (up.shape[0], down.shape[1]):
                    print(f"MOSS-Image: LoRA '{item.name}' layer {name} has the wrong shape — skipped")
                    continue
                merged = weight.data.to(device, torch.float32) + (up @ down) * (scale * item.weight_value)
                weight.data.copy_(merged.to(weight.dtype).to(weight.device))
                fused += 1
        if not fused:
            raise RuntimeError(
                f"MOSS-Image: no layer of LoRA '{item.name}' matches the MOSS-Image transformer "
                "(is it a LoRA for another model?)"
            )
        note = f", {ignored} key(s) not used" if ignored else ""
        print(f"MOSS-Image: LoRA '{item.name}' fused into {fused} layers (weight={item.weight_value:.2f}{note})")


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

class _MossImageBase(ModelPlugin):
    MODEL_ID          = ""   # abstract: plugin discovery skips classes without one
    MODEL_TYPE        = "image"
    INPUTS            = InputSpec.PROMPT | InputSpec.NEG_PROMPT | InputSpec.IMAGE | InputSpec.LORA
    UI_SECTIONS       = [
        UISection.PROMPT, UISection.NEG_PROMPT, UISection.IMAGE_STRIP,
        UISection.RESOLUTION, UISection.FRAMES, UISection.STEPS, UISection.GUIDANCE,
        UISection.IMAGE_STRENGTH, UISection.SEED,
        UISection.LORA,
    ]
    # Official defaults: 50 Euler steps, CFG 3 (3-6 for short prompts), shift 5.
    PARAMS            = ParamSpec(steps=50, guidance=3.0)
    REQUIRED_PACKAGES = ["torch", "diffusers", "transformers", "sdnq"]
    supports_inpaint      = True
    inpaint_uses_strength = True
    uses_box_editor       = True   # prompt may be Box Editor JSON
    # Scene props load() reads: a change makes the queue reload the model.
    LOAD_VARIANT_KEYS     = ("mossimg_weights",)

    # ------------------------------------------------------------------ load

    def _missing_download_gb(self, cache_dir) -> int:
        gb = 0
        if not _in_cache(_SHARED_REPO, _SHARED_FILES, cache_dir):
            gb += _SHARED_GB
        if not _in_cache(self.MODEL_ID, _TRANSFORMER_FILES, cache_dir):
            gb += _TRANSFORMER_GB
        return gb


    @staticmethod
    def _resident(weights: str) -> bool:
        return not low_vram() and _total_vram_gib() >= _RESIDENT_GIB.get(weights, 30)

    def load(self, prefs, scene, **kw):
        import torch
        from diffusers import AutoencoderKLFlux2
        from diffusers.utils.dynamic_modules_utils import get_class_from_dynamic_module
        from transformers import AutoModel, AutoTokenizer
        from sdnq import SDNQConfig, sdnq_post_load_quant

        weights = getattr(scene, "mossimg_weights", "sdnq8")
        if weights not in _WEIGHTS:
            weights = "sdnq8"
        tr_quant, te_quant = _WEIGHTS[weights]
        cache_dir = prefs.hf_cache_dir or None
        hub = dict(cache_dir=cache_dir, local_files_only=prefs.local_files_only)

        missing = self._missing_download_gb(cache_dir)
        if prefs.local_files_only and missing:
            raise RuntimeError(
                f"{self.DISPLAY_NAME} is not downloaded yet and 'Use Local Files Only' is on. "
                f"Uncheck it in the add-on Preferences to download (~{missing} GB, once)."
            )
        print(f"Loading {self.MODEL_ID} (weights={weights})…")

        def _cls(name):
            return get_class_from_dynamic_module(
                _SHARED_REPO, _CODE_FILE, class_name=name, trust_remote_code=True, **hub,
            )
        transformer_cls = _cls("MOSSImageTransformer")
        scheduler_cls = _cls("MOSSImageScheduler")
        pipeline_cls = _cls("MOSSImagePipeline")

        on_gpu = gfx_device == "cuda" and self._resident(weights)
        q_device = "cuda" if gfx_device == "cuda" else None
        return_device = "cuda" if on_gpu else "cpu"

        # Transformer first, while the GPU is still empty: loaded in bf16 straight
        # into VRAM where it fits (10 GB), LoRAs fused in, then quantized there,
        # so only the quantized weights ever reach system RAM. Smaller GPUs
        # stage it in system RAM instead.
        stage = "cuda" if gfx_device == "cuda" and _total_vram_gib() >= _GPU_STAGE_GIB else "cpu"
        transformer = transformer_cls.from_pretrained(
            self.MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16,
            device_map=stage, **hub,
        ).eval()
        enabled_items = kw.get("enabled_items") or []
        if enabled_items:
            from ...utils.helpers import bpy
            folder = bpy.path.abspath(getattr(scene, "lora_folder", "") or "")
            _fuse_loras(transformer, enabled_items, folder)
        if tr_quant:
            transformer = sdnq_post_load_quant(
                transformer, weights_dtype=tr_quant, torch_dtype=torch.bfloat16,
                quantization_device=q_device, return_device=return_device,
                modules_to_not_convert=list(_KEEP_BF16),
            )
        transformer.to(return_device if gfx_device == "cuda" else "cpu")
        if stage == "cuda":
            torch.cuda.empty_cache()

        # Text encoder: Qwen3.5-9B, quantized shard by shard on the GPU while
        # loading. Its vision tower never runs for text prompts: kept out of
        # quantization and dropped afterwards (~1 GB).
        text_encoder = AutoModel.from_pretrained(
            _SHARED_REPO, subfolder="text_encoder", dtype=torch.bfloat16,
            quantization_config=SDNQConfig(
                weights_dtype=te_quant, quantization_device=q_device,
                return_device=return_device, modules_to_not_convert=["visual"],
            ),
            **hub,
        ).eval()
        if getattr(text_encoder, "visual", None) is not None:
            text_encoder.visual = None
        tokenizer = AutoTokenizer.from_pretrained(_SHARED_REPO, subfolder="tokenizer", **hub)
        vae = AutoencoderKLFlux2.from_pretrained(
            _SHARED_REPO, subfolder="vae", torch_dtype=torch.float32, **hub,
        ).eval()

        pipe = pipeline_cls(
            transformer=transformer, scheduler=scheduler_cls(),
            text_encoder=text_encoder, tokenizer=tokenizer, vae=vae,
        )

        if gfx_device == "mps":
            pipe.to("mps")
            placement = "on Apple GPU (MPS)"
        elif on_gpu:
            pipe.to("cuda")
            placement = "resident on GPU"
        elif gfx_device == "cuda":
            # Whole-component moves only: group offload can strand SDNQ scale
            # and zero-point buffers on the CPU (see the Qwen-Image 2.1 plugin).
            pipe.enable_model_cpu_offload()
            placement = f"component CPU offload ({_total_vram_gib():.0f} GB VRAM)"
        else:
            placement = "on CPU (no CUDA GPU found) — expect very slow generation"
        print(f"MOSS-Image: {placement}")
        return {"pipe": pipe, "converter": None, "refiner": None, "preprocessor": None}

    # ------------------------------------------------------------------ UI

    def draw_model_options_ui(self, col, context):
        """Weights and sampler options, between the model selector and Batch Count."""
        scene = context.scene
        sub = col.column(align=True)
        sub.prop(scene, "mossimg_weights", text="Weights")
        sub.prop(scene, "mossimg_guidance_mode", text="Layer Guidance")
        mode = scene.mossimg_guidance_mode
        if mode != "off":
            sub.prop(scene, "mossimg_skip_layers", text="Skip Layers")
        if mode == "slg":
            sub.prop(scene, "mossimg_slg_scale", text="SLG Scale")
        sub.prop(scene, "mossimg_shift", text="Shift")
        self._draw_download_info(col, context)

    def _draw_download_info(self, col, context):
        try:
            from ...utils.helpers import ADDON_ID
            prefs = context.preferences.addons[ADDON_ID].preferences
        except Exception:
            return
        missing = self._missing_download_gb(getattr(prefs, "hf_cache_dir", "") or None)
        if not missing:
            return
        box = col.box().column(align=True)
        box.label(text=f"Not downloaded yet: ~{missing} GB on first run", icon="IMPORT")
        if getattr(prefs, "local_files_only", False):
            box.label(text="'Use Local Files Only' is on: downloads are blocked", icon="ERROR")

    # ------------------------------------------------------------------ generate

    @staticmethod
    def _embeds(pipe_obj, prompt: str, negative: str, do_cfg: bool):
        """Prompt embeddings, reused while prompt and negative are unchanged
        (batch runs and seed sweeps skip the text encoder)."""
        pipe = pipe_obj["pipe"]
        key = (prompt, negative if do_cfg else None)
        cached = pipe_obj.get("mossimg_embeds")
        if cached is not None and cached[0] == key:
            return cached[1]
        pos, pos_mask = pipe.encode_prompt(prompt)
        neg = neg_mask = None
        if do_cfg:
            neg, neg_mask = pipe.encode_prompt(negative)
        pipe_obj["mossimg_embeds"] = (key, (pos, pos_mask, neg, neg_mask))
        return pos, pos_mask, neg, neg_mask

    @staticmethod
    def _encode_image(pipe, image, width, height, device):
        """PIL image -> normalized 128-channel latents (FLUX.2 patchify + BN)."""
        import torch
        vae = pipe.vae
        x = pipe.image_processor.preprocess(image, height=height, width=width)
        x = x.to(device=device, dtype=vae.dtype)
        z = vae.encode(x).latent_dist.mode()
        b, c, h, w = z.shape
        z = z.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)
        bn = vae.bn
        mean = bn.running_mean.view(1, -1, 1, 1).to(z)
        std = torch.sqrt(bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(z)
        return ((z - mean) / std).float()

    @staticmethod
    def _decode(pipe, latents, width, height):
        """Normalized latents -> PIL image (the hub pipeline's decode)."""
        vae = pipe.vae
        if width * height > _VAE_TILE_PIXELS:
            vae.enable_tiling()
        else:
            vae.disable_tiling()
        z = latents.to(dtype=vae.dtype)
        bn = vae.bn
        eps = vae.config.batch_norm_eps
        z = z * (bn.running_var + eps).sqrt().to(z).view(1, -1, 1, 1) + bn.running_mean.to(z).view(1, -1, 1, 1)
        b, c, h, w = z.shape
        z = z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)
        images = vae.decode(z, return_dict=False)[0].clamp(-1, 1)
        return pipe.image_processor.postprocess(images, output_type="pil")[0]

    @staticmethod
    def _source(inputs: ModelInputs, width: int, height: int):
        """(source image, inpaint mask, denoise strength) for img2img/inpaint;
        (None, None, 1.0) for txt2img. img2img keeps Strip Power of the source,
        inpaint repaints the mask with Strip Power as the denoise strength."""
        from PIL import Image as _PILImage
        if inputs.mode not in ("img2img", "inpaint") or inputs.image is None:
            return None, None, 1.0
        src = inputs.image.convert("RGB").resize((width, height), _PILImage.LANCZOS)
        mask = None
        if inputs.mode == "inpaint":
            if inputs.inpaint_mask is None:
                raise RuntimeError("Inpaint mask not available. Check inpaint_selected_strip.")
            mask = inputs.inpaint_mask.convert("L").resize((width, height), _PILImage.LANCZOS)
            denoise = float(inputs.strength)
        else:
            denoise = 1.0 - float(inputs.strength)
        return src, mask, min(max(denoise, 0.0), 1.0)

    @staticmethod
    def _masks(mask, device):
        """(feathered pixel mask for the final composite, latent-grid mask).
        Repaints a little past the mask edge; the latent grid is 16 px."""
        import numpy as np
        import torch
        import torch.nn.functional as F
        from PIL import ImageFilter
        soft_mask = mask.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(4))
        m = torch.from_numpy(np.asarray(soft_mask, dtype=np.float32) / 255.0)
        return soft_mask, F.max_pool2d(m[None, None], kernel_size=16, stride=16).to(device)

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch
        from PIL import Image as _PILImage
        from diffusers.utils.torch_utils import randn_tensor

        pipe = pipe_obj["pipe"]
        transformer = pipe.transformer
        width = max(16, int(inputs.width) // 16 * 16)
        height = max(16, int(inputs.height) // 16 * 16)

        prompt = to_moss_prompt(inputs.prompt, width, height)
        if prompt != inputs.prompt:
            print(f"MOSS-Image final prompt: {prompt}")
        if not prompt.strip():
            raise RuntimeError("MOSS-Image needs a prompt.")

        # Guidance: CFG, optionally with the negative branch run with blocks
        # skipped ("uncond skip", the official default: block 22) or with an
        # extra skipped positive pass (standard SLG, 0.01-1.0 of the schedule).
        depth = len(transformer.layers)
        mode = getattr(scene, "mossimg_guidance_mode", "uncond")
        layers = _parse_layers(getattr(scene, "mossimg_skip_layers", "22"), depth)
        uncond_skip = layers if mode == "uncond" else ()
        slg_layers = layers if mode == "slg" else ()
        slg_scale = float(getattr(scene, "mossimg_slg_scale", 3.0)) if mode == "slg" else 0.0
        slg_start = 0.01 if mode == "slg" else 0.0
        guidance = max(0.0, float(inputs.guidance))
        do_cfg = guidance != 1.0

        src, mask, denoise = self._source(inputs, width, height)
        if src is not None and denoise <= 0.0:
            print("MOSS-Image: Strip Power keeps the whole source — returning it unchanged")
            return src

        scheduler = type(pipe.scheduler)(
            num_train_timesteps=1000, shift=float(getattr(scene, "mossimg_shift", 5.0)), training_shift=1.0,
        )
        pipe.scheduler = scheduler

        self.set_phase(inputs, "Encoding prompt")
        with torch.no_grad():
            pos, pos_mask, neg, neg_mask = self._embeds(pipe_obj, prompt, inputs.neg_prompt or "", do_cfg)
        device = pipe._execution_device

        generator = torch.Generator("cpu").manual_seed(inputs.seed) if inputs.seed != 0 else None
        shape = (1, transformer.config.latent_channels, height // 16, width // 16)
        noise = randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)

        scheduler.set_timesteps(inputs.steps, device=device)
        timesteps = scheduler.timesteps
        sigmas = scheduler.sigmas
        n = len(timesteps)
        start = 0
        x0 = latent_mask = soft_mask = None
        if src is not None:
            start = min(n - 1, int(round(n * (1.0 - denoise))))
            with torch.no_grad():
                x0 = self._encode_image(pipe, src, width, height, device)
            sigma = float(sigmas[start])
            latents = sigma * noise + (1.0 - sigma) * x0
            if start:
                scheduler.set_begin_index(start)
            if mask is not None:
                soft_mask, latent_mask = self._masks(mask, device)
        else:
            latents = noise
        print(f"MOSS-Image: {inputs.mode} {width}x{height}, {n - start}/{n} steps, CFG {guidance:g}, "
              f"layer guidance={mode} {list(layers) if mode != 'off' else ''}")

        self.set_phase(inputs, "Generating")
        progress_fn = inputs.progress_fn
        should_cancel = inputs.should_cancel
        total = max(1, n - start)
        try:
            with torch.no_grad():
                for i in range(start, n):
                    if should_cancel is not None and should_cancel():
                        raise KeyboardInterrupt("MOSS-Image: cancelled")
                    t = timesteps[i]
                    skip_active = slg_start <= i / n < 1.0
                    prediction = transformer(latents, t, pos, pos_mask).sample.float()
                    conditional = prediction
                    if do_cfg:
                        skip_kw = {"skip_layers": uncond_skip} if uncond_skip and skip_active else {}
                        uncond = transformer(latents, t, neg, neg_mask, **skip_kw).sample.float()
                        prediction = uncond + guidance * (prediction - uncond)
                    if slg_layers and slg_scale > 0 and skip_active:
                        skipped = transformer(latents, t, pos, pos_mask, skip_layers=slg_layers).sample.float()
                        if not do_cfg:
                            prediction = skipped
                        prediction = prediction + slg_scale * (conditional - skipped)
                    latents = scheduler.step(prediction, t, latents, return_dict=False)[0]
                    if latent_mask is not None:
                        # Outside the mask, follow the source at the next noise level.
                        s_next = float(sigmas[i + 1])
                        known = s_next * noise + (1.0 - s_next) * x0
                        latents = latent_mask * latents + (1.0 - latent_mask) * known
                    if progress_fn is not None:
                        progress_fn(i - start + 1, total)

                self.set_phase(inputs, "Decoding")
                image = self._decode(pipe, latents, width, height)
        finally:
            pipe.maybe_free_model_hooks()

        if soft_mask is not None:
            # Keep the source pixels exact outside the (feathered) mask.
            image = _PILImage.composite(image, src, soft_mask)
        return image.convert("RGB")


class MossImageBasePlugin(_MossImageBase):
    MODEL_ID     = "OpenMOSS-Team/MOSS-Image-Base"
    DISPLAY_NAME = "MOSS-Image Base"
    DESCRIPTION  = (
        "MOSS-Image Base (OpenMOSS, 5B DiT, 50 steps, CFG 3): text-to-image with accurate English/Chinese "
        "text rendering, prompts up to 4K tokens and bounding-box layouts via the Box Editor; img2img, "
        "inpaint and LoRAs. The checkpoint LoRAs are trained on. SDNQ 8/4-bit on the fly"
    )


class MossImageSFTPlugin(_MossImageBase):
    MODEL_ID     = "OpenMOSS-Team/MOSS-Image-SFT"
    DISPLAY_NAME = "MOSS-Image SFT"
    DESCRIPTION  = (
        "MOSS-Image SFT (OpenMOSS, 5B DiT fine-tuned on curated data): cleaner text grouping and layout "
        "than Base; text-to-image, Box Editor layouts, img2img, inpaint and LoRAs. Shares the text "
        "encoder download with MOSS-Image Base"
    )
