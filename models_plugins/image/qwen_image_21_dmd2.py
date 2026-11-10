"""Two-step text-to-image and multi-reference generation via Qwen-Image 2.1 (SDNQ 8-bit) with the DMD2 distillation LoRA, with up to 9 reference image slots."""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram


_DMD2_REPO    = "linoyts/Qwen-Image-2.1-DMD2-2steps"
_DMD2_WEIGHTS = "pytorch_lora_weights_v1_5.safetensors"
_DMD2_ADAPTER = "dmd2"
# The fixed schedule the LoRA was trained on. The teacher's guidance is baked
# into the student, so it samples CFG-free (true_cfg_scale=1.0).
_DMD2_SIGMAS  = [1.0, 0.5]


def _peft_compatible_lora(path):
    """Path to a copy of the LoRA whose embedded PEFT config the installed peft can parse.

    The file is saved with peft 0.20, whose adapter metadata carries LoraConfig
    fields (lora_ga_config, monteclora_config, ...) that older peft rejects with
    "unexpected keyword argument". Unknown fields are dropped; the tensors and
    every known field (r, lora_alpha, target_modules) are untouched.
    """
    import dataclasses
    import json
    import os
    import tempfile
    from peft import LoraConfig
    from safetensors import safe_open
    from safetensors.torch import save_file

    with safe_open(path, "pt") as f:
        meta = dict(f.metadata() or {})
        raw = meta.get("lora_adapter_metadata")
        if not raw:
            return path
        cfg = json.loads(raw)
        # Keys are component-prefixed, e.g. "transformer.lora_alpha".
        allowed = {fld.name for fld in dataclasses.fields(LoraConfig)}
        keep = {k: v for k, v in cfg.items() if k.split(".", 1)[-1] in allowed}
        if len(keep) == len(cfg):
            return path
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    meta["lora_adapter_metadata"] = json.dumps(keep, indent=2)
    out_dir = os.path.join(tempfile.gettempdir(), "pallaidium_lora_compat")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, os.path.basename(path))
    save_file(tensors, out, metadata=meta)
    print(f"Qwen-Image 2.1 DMD2: dropped {sorted(set(cfg) - set(keep))} from LoRA metadata (installed peft is older)")
    return out


class QwenImage21DMD2Plugin(ModelPlugin):
    # The LoRA is trained on base Qwen-Image 2.1, NOT on the Turbo finetune.
    MODEL_ID     = "OzzyGT/Qwen_Image_2_1_sdnq_dynamic_8bit"
    DISPLAY_NAME = "Qwen-Image 2.1 DMD2 (2 steps)"
    DESCRIPTION  = ("Two-step text-to-image via Qwen-Image 2.1 (SDNQ 8-bit) with the DMD2 LoRA and up to 9 reference "
                    "images. Qwen Research License: non-commercial research and evaluation only")
    MODEL_TYPE   = "image"
    INPUTS       = InputSpec.PROMPT | InputSpec.IMAGE | InputSpec.LORA
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.IMAGE_STRIP,
        UISection.RESOLUTION, UISection.FRAMES, UISection.SEED,
        UISection.LORA,
    ]
    # Steps and guidance are fixed by the distillation. Trained multi-aspect at ~1 MP.
    PARAMS            = ParamSpec(width=1024, height=1024, steps=2, guidance=1.0)
    REQUIRED_PACKAGES = ["torch", "diffusers", "transformers", "sdnq"]
    supports_inpaint  = False

    def load(self, prefs, scene, **kw):
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline
        from huggingface_hub import hf_hub_download
        # Registers SDNQ's quantizer so from_pretrained recognizes
        # quant_method="sdnq" (see the Qwen-Image 2.1 Turbo plugin).
        from sdnq import SDNQConfig  # noqa: F401

        _cache_dir = prefs.hf_cache_dir or None
        _lfo = prefs.local_files_only
        print(f"Loading {self.MODEL_ID}…")

        pipe = QwenImage21Pipeline.from_pretrained(
            self.MODEL_ID, torch_dtype=torch.bfloat16,
            cache_dir=_cache_dir, local_files_only=_lfo,
        )
        # References go through Qwen3-VL's vision tower: its bf16 Conv3d patch embed
        # is pathologically slow on this torch/cuDNN — run it as a matmul.
        from ...utils.fast_patch_embed import install_fast_patch_embed
        install_fast_patch_embed(pipe.text_encoder, "Qwen-Image 2.1 DMD2")
        # The stored config applies resolution-based shifting and a 0.02 terminal
        # stretch, either of which would move the trained sigmas.
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            pipe.scheduler.config, use_dynamic_shifting=False, shift=1.0, shift_terminal=None,
        )

        lora_path = hf_hub_download(
            repo_id=_DMD2_REPO, filename=_DMD2_WEIGHTS,
            cache_dir=_cache_dir, local_files_only=_lfo,
        )
        # Stays an unmerged adapter: merging into int8 SDNQ weights is not possible.
        pipe.load_lora_weights(_peft_compatible_lora(lora_path), adapter_name=_DMD2_ADAPTER)
        names, weights = [_DMD2_ADAPTER], [1.0]

        enabled_items = kw.get("enabled_items", [])
        if enabled_items:
            from ...utils.helpers import clean_filename, bpy
            lora_folder = getattr(bpy.context.scene, "lora_folder", "")
            for item in enabled_items:
                name = clean_filename(item.name).replace(".", "")
                names.append(name)
                weights.append(item.weight_value)
                pipe.load_lora_weights(
                    bpy.path.abspath(lora_folder),
                    weight_name=item.name + ".safetensors",
                    adapter_name=name,
                )
                print(f"Qwen-Image 2.1 DMD2: user LoRA '{item.name}' loaded (adapter='{name}', weight={item.weight_value:.2f})")
        pipe.set_adapters(names, adapter_weights=weights)
        print(f"Qwen-Image 2.1 DMD2: active adapters={names} weights={weights}")

        # Exact row strips: bounded memory, no tiling seams (see the Turbo plugin).
        try:
            from ._qwen_image_21_vae_strips import patch_vae
            patch_vae(pipe.vae)
        except Exception as e:
            print(f"Qwen-Image 2.1 DMD2: strip VAE decode unavailable ({e}); using tiling (may show seams).")
            pipe.vae.enable_tiling()

        # NOT enable_group_offload: leaf-level streaming can leave SDNQ side
        # buffers (scale / zero_point) on CPU → cross-device dequant → noise.
        if gfx_device == "mps":
            pipe.to("mps")
        elif low_vram():
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.enable_model_cpu_offload()
        # After offloading is set up. Never fatal: without it jobs just read the
        # reference images every time, as they always did.
        try:
            from ._qwen_image_21_vision_cache import patch_vision_cache
            patch_vision_cache(pipe.text_encoder)
        except Exception as e:
            print(f"{self.DISPLAY_NAME}: vision cache unavailable ({e})")
        return {"pipe": pipe, "converter": pipe, "refiner": None, "preprocessor": None}

    def draw_custom_ui(self, col, context) -> bool:
        scene = context.scene
        # Same shared klein_strip_N pickers as the Turbo plugin.
        vse_scene = getattr(context, "sequencer_scene", None) or context.scene
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

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch

        seed = inputs.seed
        # CPU generator: with offloading the pipeline device reports CPU.
        generator = torch.Generator("cpu").manual_seed(seed) if seed != 0 else None

        from PIL import Image as _PILImage
        ref_images = []
        for attr in (f"klein_strip_{i}_path" for i in range(1, 10)):
            path = getattr(scene, attr, None)
            if path:
                try:
                    img = _PILImage.open(path).convert("RGB")
                    ref_images.append(img)
                    print(f"Qwen-Image 2.1 DMD2 ref loaded: {attr} = '{path}' {img.size}")
                except Exception as e:
                    print(f"Qwen-Image 2.1 DMD2 ref failed to open '{path}': {e}")
        # The active input is always the first reference ("Picture 1" in the prompt).
        images = ([inputs.image.convert("RGB")] if inputs.image is not None else []) + ref_images
        print(f"Qwen-Image 2.1 DMD2: {len(images)} reference image(s) loaded, mode={inputs.mode}")

        self.set_phase(inputs, "Generating")
        pipe_key = "converter" if (inputs.mode == "img2img" and inputs.image is not None) else "pipe"
        return pipe_obj[pipe_key](
            prompt=inputs.prompt,
            image=images if images else None,
            true_cfg_scale=1.0,
            num_inference_steps=len(_DMD2_SIGMAS),
            sigmas=_DMD2_SIGMAS,
            height=inputs.height,
            width=inputs.width,
            generator=generator,
            callback_on_step_end=self.step_callback(inputs),
        ).images[0]
