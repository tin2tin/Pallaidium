"""Schematic map → image via FLUX.2 Klein 9B (no schematic LoRA, multi-ref conditioning)."""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram, find_strip_by_name, load_strip_as_pil


class Flux2Klein9BSchematicGuidePlugin(ModelPlugin):
    MODEL_ID     = "black-forest-labs/FLUX.2-klein-base-9B#guide"
    DISPLAY_NAME = "FLUX.2 Klein 9B Schematic → Image"
    DESCRIPTION  = "Generate images guided by schematic maps (depth/pose/normal/seg) via Klein 9B"
    MODEL_TYPE   = "image"
    INPUTS       = InputSpec.PROMPT | InputSpec.IMAGE | InputSpec.LORA
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.IMAGE_STRIP,
        UISection.RESOLUTION, UISection.FRAMES, UISection.STEPS,
        UISection.GUIDANCE, UISection.IMAGE_STRENGTH, UISection.SEED,
        UISection.LORA,
    ]
    PARAMS            = ParamSpec(steps=25, guidance=3.5)
    REQUIRED_PACKAGES = ["torch", "diffusers", "transformers"]
    supports_inpaint  = False
    supports_img2img  = True
    requires_input_strip = True
    requires_no_style    = True

    _BASE_PIPELINE = "ModelsLab/FLUX.2-klein-9B"
    _TRANSFORMER   = "OzzyGT/flux2_klein_9B_bnb_4bit_transformer"
    _TEXT_ENCODER  = "OzzyGT/flux2_klein_9B_bnb_4bit_text_encoder"

    def load(self, prefs, scene, **kw):
        import torch
        from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
        from transformers import Qwen3ForCausalLM

        _cache_dir = prefs.hf_cache_dir or None
        print(f"Loading {self.MODEL_ID}…")

        try:
            from transformers import BitsAndBytesConfig
            _bnb4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        except Exception:
            _bnb4 = None
        _bnb_kw = {"quantization_config": _bnb4} if _bnb4 is not None else {}
        transformer = Flux2Transformer2DModel.from_pretrained(
            self._TRANSFORMER, torch_dtype=torch.bfloat16, device_map="cpu", cache_dir=_cache_dir,
            **_bnb_kw,
        )
        text_encoder = Qwen3ForCausalLM.from_pretrained(
            self._TEXT_ENCODER, torch_dtype=torch.bfloat16, device_map="cpu", cache_dir=_cache_dir,
            **_bnb_kw,
        )
        pipe = Flux2KleinPipeline.from_pretrained(
            self._BASE_PIPELINE,
            transformer=transformer, text_encoder=text_encoder,
            torch_dtype=torch.bfloat16, cache_dir=_cache_dir,
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
            pipe.set_adapters(names, adapter_weights=weights)

        if gfx_device == "mps":
            pipe.to("mps")
        else:
            pipe.enable_model_cpu_offload()

        return {"pipe": pipe, "converter": pipe, "refiner": None, "preprocessor": None}

    def draw_custom_ui(self, col, context) -> bool:
        scene = context.scene
        row = col.row()
        row.enabled = False
        try:
            row.prop(scene, "input_strips", text="Input")
        except Exception:
            pass
        if scene.sequence_editor is None:
            return True
        for attr, action in [
            ("klein_strip_1", "klein_select1"),
            ("klein_strip_2", "klein_select2"),
            ("klein_strip_3", "klein_select3"),
        ]:
            row = col.row(align=True)
            row.prop_search(
                scene, attr, scene.sequence_editor, "strips",
                text="Ref.", icon="FILE_IMAGE",
            )
            row.operator("sequencer.strip_picker", text="", icon="EYEDROPPER").action = action
        return True

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch

        if inputs.image is None:
            raise ValueError("FLUX.2 Klein Schematic → Image requires a schematic map as input image.")

        src = inputs.image.convert("RGB").resize((inputs.width, inputs.height))

        ref_images = []
        for attr in ["klein_strip_1", "klein_strip_2", "klein_strip_3"]:
            strip_name = getattr(scene, attr, None)
            if strip_name:
                strip = find_strip_by_name(scene, strip_name)
                if strip:
                    img = load_strip_as_pil(strip)
                    if img is not None:
                        ref_images.append(img.resize((inputs.width, inputs.height)))

        all_images = [src] + ref_images

        seed = inputs.seed
        generator = (
            torch.Generator("cuda").manual_seed(seed)
            if torch.cuda.is_available() and seed != 0 else None
        )

        self.set_phase(inputs, "Generating")
        return pipe_obj["converter"](
            prompt=inputs.prompt,
            image=all_images,
            num_inference_steps=inputs.steps,
            guidance_scale=inputs.guidance,
            height=inputs.height,
            width=inputs.width,
            generator=generator,
            callback_on_step_end=self.step_callback(inputs),
        ).images[0]
