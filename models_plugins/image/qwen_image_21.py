"""Text-to-image and multi-reference generation via Qwen-Image 2.1, with up to 9 reference image slots."""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram


class QwenImage21Plugin(ModelPlugin):
    MODEL_ID     = "Qwen/Qwen-Image-2.1"
    DISPLAY_NAME = "Qwen-Image 2.1"
    DESCRIPTION  = "Text-to-image via Qwen-Image 2.1 (quantized) with up to 9 reference images"
    MODEL_TYPE   = "image"
    INPUTS       = InputSpec.PROMPT | InputSpec.NEG_PROMPT | InputSpec.IMAGE | InputSpec.LORA
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.NEG_PROMPT, UISection.IMAGE_STRIP,
        UISection.RESOLUTION, UISection.FRAMES, UISection.STEPS, UISection.GUIDANCE,
        UISection.SEED,
        UISection.LORA,
    ]
    # true_cfg_scale defaults to 1.0 (CFG-free) upstream; the Guidance slider maps
    # straight onto it and only takes effect once a negative prompt is also set.
    PARAMS            = ParamSpec(steps=40, guidance=1.0)
    REQUIRED_PACKAGES = ["torch", "diffusers", "transformers"]
    supports_inpaint       = False
    # QwenImage21Pipeline has no strength param, no inpaint variant, and no
    # separate img2img class — image= is reference conditioning (VLM context +
    # VAE latent tokens), not a denoise blend, so there is nothing for the
    # Image Strength slider to control.

    def load(self, prefs, scene, **kw):
        import torch
        from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel
        from transformers import Qwen3VLForConditionalGeneration

        _cache_dir = prefs.hf_cache_dir or None
        _lfo = prefs.local_files_only
        dtype = torch.bfloat16
        print(f"Loading {self.MODEL_ID}…")

        # No pre-quantized community checkpoint exists yet for this very new
        # model, so quantize on the fly (same approach as the Qwen-Image-2512
        # img2img path in qwen_image.py) instead of depending on one.
        try:
            from diffusers import BitsAndBytesConfig as DBnB
            from transformers import BitsAndBytesConfig as TBnB
            _q_t = DBnB(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
            _q_e = TBnB(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
        except Exception:
            _q_t = _q_e = None
        _t_kw = {"quantization_config": _q_t} if _q_t is not None else {}
        _e_kw = {"quantization_config": _q_e} if _q_e is not None else {}

        transformer = QwenImage21Transformer2DModel.from_pretrained(
            self.MODEL_ID, subfolder="transformer", torch_dtype=dtype,
            cache_dir=_cache_dir, local_files_only=_lfo, **_t_kw,
        ).to("cpu")
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            self.MODEL_ID, subfolder="text_encoder", torch_dtype=dtype,
            cache_dir=_cache_dir, local_files_only=_lfo, **_e_kw,
        ).to("cpu")
        # vae, scheduler, and processor load from the base repo untouched.
        pipe = QwenImage21Pipeline.from_pretrained(
            self.MODEL_ID,
            transformer=transformer, text_encoder=text_encoder, torch_dtype=dtype,
            cache_dir=_cache_dir, local_files_only=_lfo,
        )

        enabled_items = kw.get("enabled_items", [])
        if enabled_items:
            from ...utils.helpers import clean_filename, bpy
            lora_folder = getattr(bpy.context.scene, "lora_folder", "")
            names, weights = [], []
            for item in enabled_items:
                name = clean_filename(item.name).replace(".", "")
                names.append(name)
                weights.append(item.weight_value)
                pipe.load_lora_weights(
                    bpy.path.abspath(lora_folder),
                    weight_name=item.name + ".safetensors",
                    adapter_name=name,
                )
                print(f"Qwen-Image 2.1: user LoRA '{item.name}' loaded (adapter='{name}', weight={item.weight_value:.2f})")
            pipe.set_adapters(names, adapter_weights=weights)
            print(f"Qwen-Image 2.1: active adapters={names} weights={weights}")

        if gfx_device == "mps":
            pipe.to("mps")
        else:
            pipe.enable_model_cpu_offload()
        return {"pipe": pipe, "converter": pipe, "refiner": None, "preprocessor": None}

    def draw_custom_ui(self, col, context) -> bool:
        scene = context.scene
        # Strip refs live in the scene shown in the VSE (context.sequencer_scene
        # in Blender 5.x), which can differ from the active scene. Reuses the
        # klein_strip_N / klein_visible_strips scene props and add/remove
        # operators shared by every up-to-9-reference plugin in this codebase
        # (FLUX.2 Klein 9B/KV/Schematic) rather than duplicating that plumbing.
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
        generator = (
            torch.Generator("cuda").manual_seed(seed)
            if torch.cuda.is_available() and seed != 0 else None
        )

        from PIL import Image as _PILImage
        ref_images = []
        for attr in (f"klein_strip_{i}_path" for i in range(1, 10)):
            path = getattr(scene, attr, None)
            if path:
                try:
                    img = _PILImage.open(path).convert("RGB")
                    ref_images.append(img)
                    print(f"Qwen-Image 2.1 ref loaded: {attr} = '{path}' {img.size}")
                except Exception as e:
                    print(f"Qwen-Image 2.1 ref failed to open '{path}': {e}")
        # The active input (image/video/scene frame) is always the first
        # reference, ahead of the named ref slots, regardless of txt2img/img2img
        # mode — matching the Klein plugins.
        images = ([inputs.image.convert("RGB")] if inputs.image is not None else []) + ref_images
        print(f"Qwen-Image 2.1: {len(images)} reference image(s) loaded, mode={inputs.mode}")

        # Ignored upstream unless paired with true_cfg_scale > 1 — passing "" would
        # still flip do_true_cfg on and log a wasted negative pass.
        neg_prompt = inputs.neg_prompt or None

        self.set_phase(inputs, "Generating")
        pipe_key = "converter" if (inputs.mode == "img2img" and inputs.image is not None) else "pipe"
        return pipe_obj[pipe_key](
            prompt=inputs.prompt,
            image=images if images else None,
            negative_prompt=neg_prompt,
            true_cfg_scale=inputs.guidance,
            num_inference_steps=inputs.steps,
            height=inputs.height,
            width=inputs.width,
            generator=generator,
            callback_on_step_end=self.step_callback(inputs),
        ).images[0]
