"""Kandinsky ImageSR 2.0: one-step diffusion super-resolution (x2 / x4), tiled at 1024 px.

Weights (default): visualbruno/Kandinsky-ImageSR-2.0 kandinsky_image_sr_2.0_fp8.safetensors (4.0 GB):
the official baked transformer with block linears stored as fp8 e4m3 + one fp32 scale per tensor, and
the time-modulation Linears replaced by their precomputed outputs for the 19 baked timesteps
(`*.modulation.table` [19, N]). Reported 52 dB PSNR (x2) / 44 dB (x4) vs bf16. Weights are dequantized
to bf16 per matmul, so it runs on any CUDA GPU (~5 GB resident). The FLUX.1 VAE comes from the
official repo's vae/ subfolder only.
Set PALLAIDIUM_KANDINSKY_SR_BF16=1 to load the official kandinskylab/Kandinsky-ImageSR-2.0 bf16
transformer instead (10.3 GB, reference quality). The pipeline code is vendored from
github.com/kandinskylab/kandinsky-image-sr (see _kandinsky_sr_vendor/SOURCE_COMMIT.txt) instead of
pip-installing the package, whose pins (transformers/huggingface-hub ranges, loguru, omegaconf, ...)
would fight the add-on's own. The library itself only needs torch, diffusers >= 0.41 and Pillow.

UI mapping (no extra scene properties):
  - Resolution  -> target size. The model runs x2 when the target is at most twice the input,
                   otherwise x4; the result is then resized to the exact target with Lanczos.
  - Creativity  -> the shared Strip Power value (scene.image_power, drawn here as "Creativity"),
                   "creativity" level 0..8 (round(strength * 8)); 0.5 = level 4 = the model default.
"""

import os
import sys

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import low_vram


_REPO = "kandinskylab/Kandinsky-ImageSR-2.0"
_FP8_REPO = "visualbruno/Kandinsky-ImageSR-2.0"
_FP8_FILE = "kandinsky_image_sr_2.0_fp8.safetensors"
_VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_kandinsky_sr_vendor")


def _import_library():
    # model_index.json names "kandinsky_image_sr" as the transformer's library, so it must be
    # importable under that exact top-level name.
    if _VENDOR_DIR not in sys.path:
        sys.path.insert(0, _VENDOR_DIR)
    try:
        from diffusers.models.transformers import transformer_kandinsky  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Kandinsky ImageSR needs diffusers 0.41 or newer (Kandinsky 5 transformer). Run Install "
            "Dependencies in the add-on Preferences, then restart Blender."
        ) from exc
    import kandinsky_image_sr
    return kandinsky_image_sr.KandinskyImageSRPipeline


def _make_fp8_modules():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class FP8Linear(nn.Module):
        """fp8 e4m3 weight + per-tensor fp32 scale, dequantized to the input dtype per call."""

        def __init__(self, src: nn.Linear):
            super().__init__()
            self.in_features, self.out_features = src.in_features, src.out_features
            self.weight = nn.Parameter(torch.empty(src.out_features, src.in_features,
                                                   dtype=torch.float8_e4m3fn, device="meta"),
                                       requires_grad=False)
            self.register_buffer("weight_scale", torch.empty((), dtype=torch.float32, device="meta"))
            self.bias = None if src.bias is None else nn.Parameter(
                torch.empty(src.out_features, dtype=torch.bfloat16, device="meta"), requires_grad=False)

        def forward(self, x):
            dtype = x.dtype if x.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
            w = self.weight.to(dtype) * self.weight_scale.to(dtype)
            b = None if self.bias is None else self.bias.to(dtype)
            return F.linear(x.to(dtype), w, b)

    class TableModulation(nn.Module):
        """Precomputed modulation output per baked timestep; the row index is set per forward."""

        def __init__(self, rows: int, width: int):
            super().__init__()
            self.register_buffer("table", torch.empty(rows, width, dtype=torch.float32, device="meta"))
            self.indices = None

        def forward(self, x):
            return self.table.index_select(0, self.indices)

    return FP8Linear, TableModulation


def _load_fp8_transformer(prefs):
    import json
    import torch
    from accelerate import init_empty_weights
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from safetensors.torch import load_file
    from kandinsky_image_sr import KandinskyImageSRBakedTransformer3DModel as Model

    cache_dir = prefs.hf_cache_dir or None
    try:
        path = hf_hub_download(_FP8_REPO, _FP8_FILE, cache_dir=cache_dir, local_files_only=True)
    except Exception:
        if prefs.local_files_only:
            raise
        print(f"Kandinsky ImageSR: downloading {_FP8_REPO}/{_FP8_FILE} (~4.0 GB)…")
        path = hf_hub_download(_FP8_REPO, _FP8_FILE, cache_dir=cache_dir)

    with safe_open(path, "pt") as f:
        cfg = json.loads(f.metadata()["kandinsky_image_sr_config"])
    for key in ("baked_modulation", "weight_quantization"):
        cfg.pop(key, None)

    FP8Linear, TableModulation = _make_fp8_modules()
    with init_empty_weights():
        model = Model(**cfg)
    rows = len(model.baked_timesteps)
    tables = []
    for block in model.visual_transformer_blocks:
        for parent in (block.self_attention, block.feed_forward):
            for name, child in list(parent.named_children()):
                if isinstance(child, torch.nn.Linear):
                    setattr(parent, name, FP8Linear(child))
        block.visual_modulation = TableModulation(rows, block.visual_modulation.out_layer.out_features)
        tables.append(block.visual_modulation)
    model.out_layer.modulation = TableModulation(rows, model.out_layer.modulation.out_layer.out_features)
    tables.append(model.out_layer.modulation)

    state = {k: v for k, v in load_file(path).items() if not k.endswith(".comfy_quant")}
    result = model.load_state_dict(state, strict=False, assign=True)
    if result.missing_keys:
        raise RuntimeError(f"Kandinsky ImageSR fp8: missing weights {result.missing_keys[:8]}")
    if result.unexpected_keys:
        print(f"Kandinsky ImageSR fp8: ignored {len(result.unexpected_keys)} unexpected keys")
    left = [n for n, p in list(model.named_parameters()) + list(model.named_buffers()) if p.is_meta]
    if left:
        raise RuntimeError(f"Kandinsky ImageSR fp8: tensors left unloaded {left[:8]}")

    def _set_indices(module, args, kwargs):
        timestep = kwargs.get("timestep", args[2] if len(args) > 2 else None)
        idx = module._timestep_indices(timestep)
        for t in tables:
            t.indices = idx

    model.register_forward_pre_hook(_set_indices, with_kwargs=True)
    return model.eval()


class KandinskyImageSRPlugin(ModelPlugin):
    MODEL_ID     = _REPO
    DISPLAY_NAME = "Kandinsky Super Resolution"
    MODEL_TYPE   = "image"
    DESCRIPTION  = (
        "Kandinsky ImageSR 2.0: one-step diffusion upscaler (x2 / x4, tiled). Set the target size with "
        "Resolution; Creativity (0.5 = default, higher invents more detail)"
    )

    INPUTS      = InputSpec.IMAGE
    UI_SECTIONS = [UISection.RESOLUTION, UISection.FRAMES]
    PARAMS      = ParamSpec(width=3840, height=2160, steps=1, guidance=1.0, strength=0.5)

    REQUIRED_PACKAGES         = ["torch", "diffusers", "PIL"]
    supports_inpaint          = False
    supports_img2img          = True
    requires_input_strip      = True
    uses_standard_input_strip = False
    show_enhance              = False
    supports_batch            = False

    def draw_post_seed_ui(self, col, context):
        # The standard Strip Power slider lives in the input-strip block this plugin replaces.
        level = int(round(context.scene.image_power * 8))
        col.prop(context.scene, "image_power", text=f"Creativity ({level}/8)", slider=True)

    def on_model_selected(self, scene, context):
        # Strength (image_power) is shared across plugins and not reset by the generic
        # model-change code, so apply this model's default here.
        scene.image_power = self.PARAMS.strength

    def load(self, prefs, scene, **kw):
        import torch
        Pipeline = _import_library()
        cache_dir = prefs.hf_cache_dir or None

        def _cached(fn):
            try:
                return fn(True)
            except (OSError, ValueError):
                if prefs.local_files_only:
                    raise
                return fn(False)

        if os.environ.get("PALLAIDIUM_KANDINSKY_SR_BF16") == "1":
            print(f"Kandinsky ImageSR: loading bf16 ({_REPO})…")
            pipe = _cached(lambda lfo: Pipeline.from_pretrained(
                _REPO, dtype=torch.bfloat16, cache_dir=cache_dir, local_files_only=lfo))
        else:
            from diffusers import AutoencoderKL
            print(f"Kandinsky ImageSR: loading fp8 ({_FP8_REPO})…")
            transformer = _load_fp8_transformer(prefs)
            vae = _cached(lambda lfo: AutoencoderKL.from_pretrained(
                _REPO, subfolder="vae", torch_dtype=torch.bfloat16, cache_dir=cache_dir,
                local_files_only=lfo))
            pipe = Pipeline(transformer=transformer, vae=vae)
        if os.environ.get("PALLAIDIUM_KANDINSKY_SR_BF16") == "1" and low_vram():
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        return pipe

    def generate(self, pipe, inputs: ModelInputs, scene, prefs):
        import torch
        from PIL import Image

        image = inputs.image
        if image is None:
            raise ValueError("Kandinsky Super Resolution needs an input image strip.")
        image = image.convert("RGB")
        target_w = int(inputs.width) or image.width * 4
        target_h = int(inputs.height) or image.height * 4
        ratio = max(target_w / image.width, target_h / image.height)
        factor = 2 if ratio <= 2.0 else 4
        creativity = max(0, min(8, int(round(float(inputs.strength) * 8))))

        self.set_phase(inputs, f"Upscaling x{factor} (creativity {creativity})")
        print(f"Kandinsky ImageSR: {image.width}x{image.height} -> x{factor}, creativity {creativity}, "
              f"target {target_w}x{target_h}")
        with torch.inference_mode():
            out = pipe(image, upscale_factor=factor, creativity=creativity).images[0]

        if out.size != (target_w, target_h):
            out = out.resize((target_w, target_h), Image.LANCZOS)
        return out
