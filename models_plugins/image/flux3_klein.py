"""FLUX.3 Image Klein: text-to-image, multi-reference editing, bounding-box layout
and layout edits, with the optional official prompt upsampler.

Two checkpoints, one plugin class each:
  - black-forest-labs/flux-3-image-klein       step-distilled, 16 steps, no CFG
  - black-forest-labs/flux-3-image-klein-base  50 steps, CFG, negative prompt

Weights (scene.flux3_weights): "sdnq8"/"sdnq4" load OzzyGT's pre-quantized
SDNQ repos where they exist (Klein; not gated) and otherwise quantize the
official weights on the fly (Base); "fp8r"/"bf16" load the official BFL
weights, which are gated (accept the license on the model page, set an HF
token). Reference images ride the shared klein_strip_N pickers (the active input strip,
when there is one, comes first as <ref_image_0>). Layout boxes ride the prompt
text: Box Editor JSON is converted to a FLUX.3 layout prompt on the fly, and a
FLUX.3 layout prompt passes through unchanged.
"""

import json

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram


# Pre-quantized SDNQ repos (transformer + text encoder quantized; not gated).
# Only the distilled Klein has them so far; for a model without one, the SDNQ
# options quantize the official weights on the fly. New ones show up under
# https://huggingface.co/models?sort=created&search=flux.3
_SDNQ_REPOS = {
    ("black-forest-labs/flux-3-image-klein", "sdnq8"): "OzzyGT/Flux_3_Image_Klein_sdnq_dynamic_8bit",
    ("black-forest-labs/flux-3-image-klein", "sdnq4"): "OzzyGT/Flux_3_Image_Klein_sdnq_dynamic_4bit",
}
_SDNQ_DTYPE   = {"sdnq8": "int8", "sdnq4": "uint4"}  # on-the-fly quantization
_SDNQ_MIN     = (0, 2, 8)  # the OzzyGT repos need sdnq >= 0.2.8
_WEIGHT_ALIAS = {"sdnq": "sdnq8"}  # older scenes/metadata

_PUP_ID   = "black-forest-labs/flux-3-image-klein-pup"
_MAX_REFS = 10  # model and upsampler limit; the input strip + 9 Ref. pickers

# Total VRAM (GiB) to keep the whole pipeline resident, per weight source.
# SDNQ 8-bit peaks at ~15.4 GiB at 2048x1152 (OzzyGT); below the threshold the
# components are offloaded to the CPU between uses.
_RESIDENT_GIB = {"sdnq8": 15, "sdnq4": 11, "fp8r": 20, "bf16": 28}
# Upsampler VRAM (GiB) incl. activations for reference images.
_PUP_GIB = {"int8": 12, "uint4": 7}

# Files that must be in the HF cache before an offline load (last shard of each
# sharded component), and the first-run download in GB.
_OFFICIAL_FILES = (
    "modular_model_index.json",
    "text_encoder/model-00002-of-00002.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
)
_OFFICIAL_TRANSFORMER = {
    "fp8r": "variants/fp8r/model.safetensors",
    "bf16": "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
}
_SDNQ_REPO_FILES = (
    "modular_model_index.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "text_encoder/model.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
)
_DOWNLOAD_GB = {"sdnq8": 10, "sdnq4": 6, "fp8r": 15, "bf16": 20}
_PUP_FILES = ("model.safetensors", "system_prompts.json")
_PUP_DOWNLOAD_GB = 19


def _weights_of(scene) -> str:
    w = getattr(scene, "flux3_weights", "sdnq8")
    return _WEIGHT_ALIAS.get(w, w)


def _check_sdnq_version() -> None:
    import re
    try:
        import sdnq
        version = getattr(sdnq, "__version__", "")
        if not version:
            from importlib.metadata import version as _dist_version
            version = _dist_version("sdnq")
    except Exception:
        return
    parts = tuple(int(p) for p in re.findall(r"\d+", version)[:3])
    if parts and parts < _SDNQ_MIN:
        raise RuntimeError(
            f"FLUX.3 SDNQ weights need sdnq {'.'.join(map(str, _SDNQ_MIN))} or newer (found {version}). "
            "Run Install Dependencies in the add-on Preferences, then restart Blender."
        )


def _in_cache(repo: str, files, cache_dir) -> bool:
    """True when every file of *repo* is in the local HF cache (no network)."""
    try:
        from huggingface_hub import try_to_load_from_cache
        return all(isinstance(try_to_load_from_cache(repo, f, cache_dir=cache_dir or None), str)
                   for f in files)
    except Exception:
        return True  # can't tell: don't block or warn


def _access_error(exc) -> bool:
    """True for a gated-repo / missing-token refusal from the Hub, anywhere in
    the exception chain (diffusers re-raises it as a generic OSError)."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if type(exc).__name__ == "GatedRepoError" or status in (401, 403):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


# Panel license status per repo: "checking", "ok", "denied" or "unknown"
# (offline / Hub error), probed by a background thread so the panel's draw()
# never makes a network call. "ok" is kept; anything else is re-checked after
# _RECHECK_S, so accepting the license on the website shows up in the panel.
_ACCESS_STATUS = {}
_ACCESS_CHECKED = {}
_RECHECK_S = 20.0


def _access_status(repo: str) -> str:
    import time
    status = _ACCESS_STATUS.get(repo)
    stale = status not in ("ok", "checking") and time.monotonic() - _ACCESS_CHECKED.get(repo, 0) > _RECHECK_S
    if status is None or stale:
        _ACCESS_STATUS[repo] = "checking" if status is None else status
        _ACCESS_CHECKED[repo] = time.monotonic()
        import threading

        def _probe():
            try:
                from huggingface_hub import auth_check
                auth_check(repo)
                _ACCESS_STATUS[repo] = "ok"
            except Exception as exc:
                _ACCESS_STATUS[repo] = "denied" if _access_error(exc) else "unknown"
            _ACCESS_CHECKED[repo] = time.monotonic()

        threading.Thread(target=_probe, daemon=True).start()
        status = _ACCESS_STATUS[repo]
    return status


def _check_access(repo: str) -> None:
    """Fail early with a readable message when the account may not download *repo*.

    One small Hub request. diffusers hides the 401/403 behind "could not find
    modular_model_index.json", so the refusal has to be caught before it loads.
    Network trouble is ignored here: the load reports it.
    """
    try:
        from huggingface_hub import auth_check
        auth_check(repo)
        _ACCESS_STATUS[repo] = "ok"
    except ImportError:
        return
    except Exception as exc:
        if _access_error(exc):
            _ACCESS_STATUS[repo] = "denied"
            raise RuntimeError(_gated_message(repo)) from exc


def _gated_message(repo: str) -> str:
    return (
        f"No access to {repo}: the model is gated and this Hugging Face account has not accepted "
        f"its license. Open huggingface.co/{repo}, log in with the account of the token set in the "
        "add-on Preferences, and accept the FLUX license. Then run again."
    )


def _patch_bfl_fp8r() -> None:
    """Make diffusers' FP8-rowwise linear work on RTX 40xx (sm89).

    diffusers sends every sm89+ GPU to torch._scaled_mm with rowwise scales, but
    the Windows torch builds only ship that kernel for sm90+ ("Rowwise scaling is
    not currently supported on your device"). sm89 does support tensorwise FP8,
    so on the first refusal switch to the same math with unit scales: FP8 GEMM,
    then multiply by the per-row activation and weight scales. If that fails
    too, dequantize the weight (diffusers' path for non-FP8 GPUs).
    """
    import torch
    import torch.nn.functional as F
    from diffusers.quantizers.bfl import utils as bfl

    cls = bfl.BFLFP8RLinear
    if getattr(cls, "_pallaidium_patched", False):
        return
    original = cls.forward
    mode = {"gemm": None}   # None = untried, "rowwise", "unit", "dequant"
    ones = {}

    def _unit_scale(self, inputs):
        x = inputs.reshape(-1, self.in_features).float()
        x_scale = (x.abs().amax(dim=1, keepdim=True) / bfl.FP8_MAX).clamp(min=1e-12)
        x_q = (x / x_scale).clamp(-bfl.FP8_MAX, bfl.FP8_MAX).to(torch.float8_e4m3fn)
        one = ones.get(inputs.device)
        if one is None:
            one = ones[inputs.device] = torch.ones((), device=inputs.device, dtype=torch.float32)
        out = torch._scaled_mm(x_q, self.weight.T, one, one, out_dtype=torch.float32, use_fast_accum=True)
        out = (out * x_scale * self.weight_scale[None, :]).to(self.compute_dtype)
        out = out.reshape(*inputs.shape[:-1], self.out_features)
        return out + self.bias if self.bias is not None else out

    def forward(self, inputs):
        fp8_capable = (
            inputs.is_cuda and self.in_features % 16 == 0 and self.out_features % 16 == 0
            and torch.cuda.get_device_capability(inputs.device) >= (8, 9)
        )
        if not fp8_capable or mode["gemm"] == "rowwise":
            return original(self, inputs)
        if mode["gemm"] is None:
            try:
                out = original(self, inputs)
                mode["gemm"] = "rowwise"
                return out
            except RuntimeError as exc:
                if "Rowwise scaling" not in str(exc):
                    raise
                mode["gemm"] = "unit"
                print("FLUX.3: rowwise FP8 GEMM unsupported on this GPU — using tensorwise FP8 with row rescaling")
        if mode["gemm"] == "unit":
            try:
                return _unit_scale(self, inputs)
            except RuntimeError as exc:
                mode["gemm"] = "dequant"
                print(f"FLUX.3: FP8 GEMM unavailable ({exc}) — dequantizing FP8 weights per step")
        weight = bfl.dequantize_fp8r(self.weight, self.weight_scale).to(self.compute_dtype)
        return F.linear(inputs, weight, self.bias)

    cls.forward = forward
    cls._pallaidium_patched = True


def _total_vram_gib() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 2**30
    except Exception:
        pass
    return 0.0


class _Flux3KleinBase(ModelPlugin):
    MODEL_ID          = ""   # abstract: plugin discovery skips classes without one
    MODEL_TYPE        = "image"
    REQUIRED_PACKAGES = ["torch", "diffusers", "transformers", "sdnq"]
    supports_inpaint  = False  # region edits go through layout boxes instead
    uses_strip_power  = False  # references condition the model; no denoise blend
    uses_box_editor   = True   # "Open Box Editor" in the main panel
    # Scene props load() reads: a change makes the queue reload the model.
    LOAD_VARIANT_KEYS = ("flux3_weights",)
    DISTILLED         = True

    # ------------------------------------------------------------------ load

    def _source(self, weights: str):
        """(repo, files to find in the cache, gated?) for a weight choice."""
        sdnq_repo = _SDNQ_REPOS.get((self.MODEL_ID, weights))
        if sdnq_repo:
            return sdnq_repo, _SDNQ_REPO_FILES, False
        # Official BFL weights (fp8r / bf16, or the bf16 source of on-the-fly SDNQ).
        transformer = _OFFICIAL_TRANSFORMER.get(weights, _OFFICIAL_TRANSFORMER["bf16"])
        return self.MODEL_ID, _OFFICIAL_FILES + (transformer,), True

    def _download_gb(self, weights: str) -> int:
        if _SDNQ_REPOS.get((self.MODEL_ID, weights)):
            return _DOWNLOAD_GB[weights]
        return _DOWNLOAD_GB["fp8r" if weights == "fp8r" else "bf16"]

    def load(self, prefs, scene, **kw):
        weights = _weights_of(scene)
        repo, files, gated = self._source(weights)
        cached = _in_cache(repo, files, prefs.hf_cache_dir)
        # Explain the first-run failures instead of surfacing a raw Hub
        # traceback: offline with nothing cached, and the gated-repo refusal.
        if prefs.local_files_only and not cached:
            gate = (f" The model is gated: accept the license on huggingface.co/{repo} and set your "
                    "Hugging Face token first." if gated else "")
            raise RuntimeError(
                f"{self.DISPLAY_NAME} ({weights}) is not downloaded yet and 'Use Local Files Only' is on. "
                f"Uncheck it in the add-on Preferences to download (~{self._download_gb(weights)} GB, "
                f"once).{gate}"
            )
        if gated and not prefs.local_files_only and not cached:
            _check_access(repo)
        try:
            return self._load_pipeline(prefs, scene, weights, repo, **kw)
        except Exception as exc:
            if gated and _access_error(exc):
                raise RuntimeError(_gated_message(repo)) from exc
            raise

    def _load_pipeline(self, prefs, scene, weights, repo, **kw):
        import torch
        try:
            from diffusers import ModularPipeline, Flux3Transformer2DModel  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "FLUX.3 needs a newer diffusers with Flux3 support. Run Install Dependencies in the "
                "add-on Preferences, then restart Blender."
            ) from exc

        hub = dict(cache_dir=prefs.hf_cache_dir or None, local_files_only=prefs.local_files_only)
        print(f"Loading {repo} (weights={weights})…")
        if weights in _SDNQ_DTYPE:
            _check_sdnq_version()
            # Importing sdnq registers its quantizer with diffusers/transformers.
            import sdnq  # noqa: F401

        pipe = ModularPipeline.from_pretrained(repo, **hub)
        # The VAE runs in fp32, as BFL and the diffusers docs recommend.
        dtype = {"vae": torch.float32, "default": torch.bfloat16}
        others = [n for n in pipe.pretrained_component_names if n != "transformer"]

        if weights == "fp8r":
            from diffusers import BFLQuantizationConfig, Flux3Transformer2DModel
            from huggingface_hub import hf_hub_download
            _patch_bfl_fp8r()
            pipe.load_components(names=others, dtype=dtype, **hub)
            ckpt = hf_hub_download(self.MODEL_ID, "variants/fp8r/model.safetensors", **hub)
            transformer = Flux3Transformer2DModel.from_single_file(
                ckpt,
                config=self.MODEL_ID, subfolder="transformer",
                quantization_config=BFLQuantizationConfig(compute_dtype=torch.bfloat16),
                dtype=torch.bfloat16, **hub,
            )
            pipe.update_components(transformer=transformer)
        elif weights in _SDNQ_DTYPE and repo == self.MODEL_ID:
            # No pre-quantized repo for this model: quantize transformer and text
            # encoder on the fly, shard by shard on the GPU (as in the Ming plugin).
            from sdnq import SDNQConfig
            q_device = "cuda" if gfx_device == "cuda" else None
            ret = "cuda" if gfx_device == "cuda" and self._resident(weights) else "cpu"
            cfg = dict(weights_dtype=_SDNQ_DTYPE[weights], quantization_device=q_device, return_device=ret)
            quant = {"transformer": SDNQConfig(**cfg), "text_encoder": SDNQConfig(**cfg)}
            pipe.load_components(names=["transformer", "text_encoder"], dtype=torch.bfloat16,
                                 quantization_config=quant, **hub)
            pipe.load_components(names=[n for n in others if n != "text_encoder"], dtype=dtype, **hub)
        else:
            # bf16, or a pre-quantized SDNQ repo (its config carries the quantization)
            pipe.load_components(dtype=dtype, **hub)

        # load_components() only warns on a per-component failure — fail loudly.
        missing = [n for n in ("transformer", "text_encoder", "vae") if getattr(pipe, n, None) is None]
        if missing:
            raise RuntimeError(f"FLUX.3: failed to load {', '.join(missing)} (see the console above)")

        self._load_loras(pipe, scene, kw.get("enabled_items") or [], weights)

        on_gpu = gfx_device == "cuda" and self._resident(weights)
        if gfx_device == "mps":
            pipe.to("mps")
            placement = "on Apple GPU (MPS)"
        elif on_gpu:
            pipe.to("cuda")
            placement = "resident on GPU"
        elif gfx_device == "cuda":
            self._enable_offload(pipe)
            placement = f"component CPU offload ({_total_vram_gib():.0f} GB VRAM)"
        else:
            placement = "on CPU (no CUDA GPU found) — expect very slow generation"
        print(f"FLUX.3: {placement}")
        return {"pipe": pipe, "converter": None, "refiner": None, "preprocessor": None,
                "flux3_on_gpu": on_gpu}

    @staticmethod
    def _resident(weights: str) -> bool:
        return not low_vram() and _total_vram_gib() >= _RESIDENT_GIB.get(weights, 28)

    @staticmethod
    def _enable_offload(pipe):
        # Whole-component moves only: group offload can strand SDNQ scale and
        # zero-point buffers on the CPU (see the Qwen-Image 2.1 plugin).
        if hasattr(pipe, "enable_auto_cpu_offload"):
            pipe.enable_auto_cpu_offload(device="cuda")
            return
        from diffusers import ComponentsManager
        manager = ComponentsManager()
        for name in ("text_encoder", "transformer", "vae"):
            manager.add(name, getattr(pipe, name))
        manager.enable_auto_cpu_offload(device="cuda")

    @staticmethod
    def _load_loras(pipe, scene, enabled_items, weights):
        if not enabled_items:
            return
        from ...utils.helpers import clean_filename, bpy
        folder = bpy.path.abspath(getattr(scene, "lora_folder", "") or "")
        names, strengths = [], []
        try:
            for item in enabled_items:
                name = clean_filename(item.name).replace(".", "")
                pipe.load_lora_weights(folder, weight_name=item.name + ".safetensors", adapter_name=name)
                names.append(name)
                strengths.append(item.weight_value)
                print(f"FLUX.3: LoRA '{item.name}' loaded (adapter='{name}', weight={item.weight_value:.2f})")
            pipe.set_adapters(names, adapter_weights=strengths)
        except Exception as exc:
            hint = "" if weights == "bf16" else " LoRAs may not attach to quantized weights: set Weights to BF16 (Official)."
            raise RuntimeError(f"FLUX.3: LoRA loading failed ({exc}).{hint}") from exc

    # ------------------------------------------------------------------ UI

    def draw_custom_ui(self, col, context) -> bool:
        scene = context.scene
        # Strip refs live in the scene shown in the VSE (context.sequencer_scene
        # in Blender 5.x), which can differ from the active scene.
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
        return True

    def needs_hf_token(self, context) -> bool:
        """The token field matters only for the gated official BFL weights."""
        return self._source(_weights_of(context.scene))[2]

    def draw_model_options_ui(self, col, context):
        """Weights and prompt upsampling, between the HF token and Batch Count."""
        scene = context.scene
        sub = col.column(align=True)
        sub.prop(scene, "flux3_weights", text="Weights")
        weights = _weights_of(scene)
        if weights in _SDNQ_DTYPE and not _SDNQ_REPOS.get((self.MODEL_ID, weights)):
            sub.label(text="No pre-quantized SDNQ yet: quantized while loading", icon="INFO")
        sub.prop(scene, "flux3_prompt_upsampling", text="Prompt Upsampling")
        quant = sub.column(align=True)
        quant.active = scene.flux3_prompt_upsampling
        quant.prop(scene, "flux3_pup_quant", text="Upsampler Quant")
        self._draw_download_info(col, context)

    def _draw_download_info(self, col, context):
        """First-run notes: what still has to download, gating, and local-only blocking."""
        try:
            from ...utils.helpers import ADDON_ID
            prefs = context.preferences.addons[ADDON_ID].preferences
        except Exception:
            return
        scene = context.scene
        cache_dir = getattr(prefs, "hf_cache_dir", "") or None
        local_only = getattr(prefs, "local_files_only", False)
        weights = _weights_of(scene)
        repo, files, gated = self._source(weights)
        model_ok = _in_cache(repo, files, cache_dir)
        pup_ok = not scene.flux3_prompt_upsampling or _in_cache(_PUP_ID, _PUP_FILES, cache_dir)
        if model_ok and pup_ok:
            return

        box = col.box().column(align=True)
        if not model_ok:
            box.label(text=f"Not downloaded yet: ~{self._download_gb(weights)} GB on first run",
                      icon="IMPORT")
        if not model_ok and gated:
            # Gated repo: say whether this account may download it.
            access = "unknown" if local_only else _access_status(repo)
            if access == "denied":
                box.label(text="License not accepted: downloads are refused", icon="ERROR")
                box.label(text="Log in on the model page with your HF token's")
                box.label(text="account and accept the FLUX license")
                box.operator("wm.url_open", text="Accept License", icon="URL").url = \
                    f"https://huggingface.co/{repo}"
            elif access == "ok":
                box.label(text="License accepted", icon="CHECKMARK")
            else:
                box.label(text="Gated: accept the FLUX license on the model page", icon="LOCKED")
                box.operator("wm.url_open", text="Open Model Page", icon="URL").url = \
                    f"https://huggingface.co/{repo}"
        if not pup_ok:
            box.label(text=f"Prompt upsampler not downloaded: ~{_PUP_DOWNLOAD_GB} GB", icon="IMPORT")
        if local_only:
            box.label(text="'Use Local Files Only' is on: downloads are blocked", icon="ERROR")

    # ------------------------------------------------------------------ generate

    def _reference_images(self, inputs: ModelInputs, scene) -> list:
        from PIL import Image as _PILImage
        refs = [inputs.image.convert("RGB")] if inputs.image is not None else []
        for n in range(1, 10):
            path = getattr(scene, f"klein_strip_{n}_path", "")
            if not path:
                continue
            try:
                refs.append(_PILImage.open(path).convert("RGB"))
            except Exception as exc:
                print(f"FLUX.3: reference '{path}' could not be opened: {exc}")
        if len(refs) > _MAX_REFS:
            print(f"FLUX.3: {len(refs)} references, keeping the first {_MAX_REFS}")
            refs = refs[:_MAX_REFS]
        return refs

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch
        from ...utils.flux3_layout import to_flux3_prompt

        pipe = pipe_obj["pipe"]
        width = max(16, int(inputs.width) // 16 * 16)
        height = max(16, int(inputs.height) // 16 * 16)
        refs = self._reference_images(inputs, scene)

        prompt = to_flux3_prompt(inputs.prompt, len(refs))
        if getattr(scene, "flux3_prompt_upsampling", False):
            prompt = self._upsample(pipe_obj, prompt, refs, width, height, scene, prefs, inputs)
        if prompt != inputs.prompt:
            print(f"FLUX.3 final prompt: {prompt}")
        try:
            scene.flux3_final_prompt = prompt  # read back into the strip metadata
        except Exception:
            pass

        kwargs = dict(
            prompt=prompt, height=height, width=width,
            num_inference_steps=inputs.steps, output="images",
        )
        if inputs.seed != 0:
            kwargs["generator"] = torch.Generator().manual_seed(inputs.seed)
        if refs:
            kwargs["image"] = refs
        if not self.DISTILLED:
            pipe.update_components(guider=pipe.guider.new(guidance_scale=float(inputs.guidance)))
            if inputs.neg_prompt:
                kwargs["negative_prompt"] = inputs.neg_prompt
        print(f"FLUX.3: {len(refs)} reference(s), {width}x{height}, {inputs.steps} steps")

        # Modular pipelines take no callback_on_step_end; both denoise loops call
        # scheduler.step once per step, so report progress from there.
        scheduler = pipe.scheduler
        original_step = scheduler.step
        progress_fn = inputs.progress_fn
        total = max(1, inputs.steps)
        counter = [0]

        def _step(*args, **kw):
            out = original_step(*args, **kw)
            counter[0] += 1
            if progress_fn is not None:
                progress_fn(counter[0], total)
            return out

        scheduler.step = _step
        self.set_phase(inputs, "Generating")
        try:
            image = pipe(**kwargs)[0]
        finally:
            scheduler.step = original_step
        return image.convert("RGB")

    # ------------------------------------------------------------------ prompt upsampler

    def _upsample(self, pipe_obj, prompt, refs, width, height, scene, prefs, inputs) -> str:
        import gc
        import torch
        from ...utils.flux3_layout import (
            is_finished_layout, pup_reconcile, pup_reference_size, pup_request,
        )

        if not prompt.strip():
            return prompt
        if is_finished_layout(prompt):
            print("FLUX.3: prompt is already an upsampled layout caption — upsampler skipped")
            return prompt

        key, text, locked = pup_request(prompt, len(refs), width, height)
        quant = getattr(scene, "flux3_pup_quant", "int8")
        self.set_phase(inputs, "Upsampling prompt")

        # Make room: with the image pipeline resident and too little free VRAM,
        # park it on the CPU while the upsampler runs.
        pipe = pipe_obj["pipe"]
        parked = False
        cached = pipe_obj.get("flux3_pup")
        if cached is not None and cached["quant"] != quant:
            pipe_obj["flux3_pup"] = cached = None
        if cached is None and pipe_obj.get("flux3_on_gpu") and torch.cuda.is_available():
            free = torch.cuda.mem_get_info()[0] / 2**30
            if free < _PUP_GIB.get(quant, 12):
                print(f"FLUX.3: {free:.1f} GiB free — moving the image pipeline to CPU for the upsampler")
                pipe.to("cpu")
                gc.collect()
                torch.cuda.empty_cache()
                parked = True

        pup = cached or self._load_pup(quant, prefs)
        try:
            answer = self._run_pup(pup, key, text, refs, inputs)
        finally:
            if parked or cached is None and not pipe_obj.get("flux3_on_gpu"):
                # Not enough VRAM to keep both: free the upsampler again.
                pipe_obj["flux3_pup"] = None
                del pup
                gc.collect()
                torch.cuda.empty_cache()
            else:
                pipe_obj["flux3_pup"] = pup
            if parked:
                pipe.to("cuda")

        if not answer:
            print("FLUX.3: the upsampler returned nothing — using the prompt as typed")
            return prompt
        return pup_reconcile(answer, locked) if locked else answer

    @staticmethod
    def _load_pup(quant: str, prefs) -> dict:
        import torch
        from huggingface_hub import hf_hub_download
        try:
            from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "FLUX.3 prompt upsampling needs a transformers build with Qwen3.5 "
                "(Qwen3_5ForConditionalGeneration). Turn Prompt Upsampling off."
            ) from exc
        from sdnq import SDNQConfig

        hub = dict(cache_dir=prefs.hf_cache_dir or None, local_files_only=prefs.local_files_only)
        if prefs.local_files_only and not _in_cache(_PUP_ID, _PUP_FILES, prefs.hf_cache_dir):
            raise RuntimeError(
                "The FLUX.3 prompt upsampler is not downloaded yet and 'Use Local Files Only' is on. "
                f"Uncheck it in the add-on Preferences to download it (~{_PUP_DOWNLOAD_GB} GB, once), "
                "or turn Prompt Upsampling off."
            )
        print(f"Loading {_PUP_ID} ({quant} on the fly)…")
        # 9B in bf16 is 18.8 GB: quantize on the fly on the GPU. The vision
        # tower is small and reads the references, so it stays in bf16.
        on_cuda = torch.cuda.is_available()
        cfg = SDNQConfig(
            weights_dtype=quant,
            quantization_device="cuda" if on_cuda else None,
            return_device="cuda" if on_cuda else "cpu",
            modules_to_not_convert=["visual"],
        )
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            _PUP_ID, dtype=torch.bfloat16, quantization_config=cfg, **hub,
        ).eval()
        if on_cuda:
            model.to("cuda")
        processor = AutoProcessor.from_pretrained(_PUP_ID, **hub)
        with open(hf_hub_download(_PUP_ID, "system_prompts.json", **hub), encoding="utf-8") as fh:
            prompts = json.load(fh)
        return {"model": model, "processor": processor, "prompts": prompts, "quant": quant}

    @staticmethod
    def _run_pup(pup, key, text, refs, inputs) -> str:
        import torch
        from PIL import Image as _PILImage
        from transformers import StoppingCriteria, StoppingCriteriaList
        from ...utils.flux3_layout import pup_reference_size

        model, processor = pup["model"], pup["processor"]
        system = pup["prompts"][key]
        if refs:
            images = []
            for img in refs:
                size = pup_reference_size(img.width, img.height, len(refs))
                images.append(img if img.size == size else img.resize(size, _PILImage.Resampling.LANCZOS))
            content = [{"type": "image", "image": img} for img in images] + [{"type": "text", "text": text}]
            messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        else:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": text}]

        enc = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", enable_thinking=False,
        ).to(model.device)

        max_new = 3584
        progress_fn = inputs.progress_fn

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
            out = model.generate(
                **enc, max_new_tokens=max_new, do_sample=False,
                stopping_criteria=StoppingCriteriaList([_Progress(start)]),
            )
        return processor.decode(out[0, start:], skip_special_tokens=True).strip()


class Flux3KleinPlugin(_Flux3KleinBase):
    MODEL_ID     = "black-forest-labs/flux-3-image-klein"
    DISPLAY_NAME = "FLUX.3 Klein"
    DESCRIPTION  = (
        "FLUX.3 Image Klein (step-distilled, 16 steps): text-to-image, up to 10 reference images, "
        "bounding-box layout and layout edits via the Box Editor. SDNQ 8/4-bit by default (not gated)"
    )
    INPUTS       = InputSpec.PROMPT | InputSpec.IMAGE | InputSpec.LORA | InputSpec.HF_TOKEN
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.IMAGE_STRIP,
        UISection.RESOLUTION, UISection.FRAMES, UISection.STEPS,
        UISection.SEED, UISection.LORA,
    ]
    # Trained for 4-16 steps, no classifier-free guidance.
    PARAMS       = ParamSpec(steps=16, guidance=1.0)
    DISTILLED    = True


class Flux3KleinBasePlugin(_Flux3KleinBase):
    MODEL_ID     = "black-forest-labs/flux-3-image-klein-base"
    DISPLAY_NAME = "FLUX.3 Klein Base"
    DESCRIPTION  = (
        "FLUX.3 Image Klein Base (50 steps, CFG, negative prompt): text-to-image, up to 10 reference "
        "images, bounding-box layout and layout edits. The checkpoint LoRAs are trained on. Gated: "
        "accept the license on the model page and set an HF token"
    )
    INPUTS       = (InputSpec.PROMPT | InputSpec.NEG_PROMPT | InputSpec.IMAGE
                    | InputSpec.LORA | InputSpec.HF_TOKEN)
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.NEG_PROMPT, UISection.IMAGE_STRIP,
        UISection.RESOLUTION, UISection.FRAMES, UISection.STEPS, UISection.GUIDANCE,
        UISection.SEED, UISection.LORA,
    ]
    # Pipeline defaults: 50 steps, guidance 3.5 (BFL recommendation).
    PARAMS       = ParamSpec(steps=50, guidance=3.5)
    DISTILLED    = False
