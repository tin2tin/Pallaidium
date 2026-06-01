"""Text-to-image and img2img via FLUX.2 Klein 9B KV-cache with consistency LoRA."""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram, find_strip_by_name, load_strip_as_pil


class Flux2Klein9BKVPlugin(ModelPlugin):
    MODEL_ID     = "dx8152/Flux2-Klein-9B-Consistency"
    DISPLAY_NAME = "Image: FLUX.2 Klein 9B KV (Consistency)"
    DESCRIPTION  = "Multi-reference image generation via FLUX.2 Klein 9B KV-cache with consistency LoRA"
    MODEL_TYPE   = "image"
    INPUTS       = InputSpec.PROMPT | InputSpec.IMAGE | InputSpec.LORA
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.IMAGE_STRIP,
        UISection.RESOLUTION, UISection.FRAMES, UISection.STEPS, UISection.SEED,
        UISection.LORA,
    ]
    PARAMS            = ParamSpec(steps=4, guidance=1.0)
    REQUIRED_PACKAGES = ["torch", "diffusers", "transformers"]
    supports_inpaint  = True

    _BASE_PIPELINE       = "black-forest-labs/FLUX.2-klein-9b-kv"
    _TRANSFORMER         = "OzzyGT/flux2_klein_9B_bnb_4bit_transformer"
    _TEXT_ENCODER        = "OzzyGT/flux2_klein_9B_bnb_4bit_text_encoder"
    _CONSISTENCY_LORA    = "dx8152/Flux2-Klein-9B-Consistency"
    _CONSISTENCY_WEIGHTS = "Flux2-Klein-9B-consistency-V2.safetensors"

    def load(self, prefs, scene, **kw):
        import torch
        from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
        from transformers import Qwen3ForCausalLM

        _cache_dir = prefs.hf_cache_dir or None
        mode = kw.get("mode", "txt2img")
        print(f"Loading {self.MODEL_ID} ({mode})…")

        if mode == "inpaint":
            from diffusers import Flux2KleinInpaintPipeline, Flux2Transformer2DModel

            transformer = Flux2Transformer2DModel.from_pretrained(
                self._TRANSFORMER, torch_dtype=torch.bfloat16, device_map="cpu", cache_dir=_cache_dir,
            )
            text_encoder = Qwen3ForCausalLM.from_pretrained(
                self._TEXT_ENCODER, torch_dtype=torch.bfloat16, device_map="cpu", cache_dir=_cache_dir,
            )
            pipe = Flux2KleinInpaintPipeline.from_pretrained(
                self._BASE_PIPELINE,
                transformer=transformer, text_encoder=text_encoder,
                torch_dtype=torch.bfloat16, cache_dir=_cache_dir,
            )
            if gfx_device == "mps":
                pipe.to("mps")
            elif low_vram():
                pipe.enable_sequential_cpu_offload()
            else:
                pipe.enable_model_cpu_offload()
            return {"pipe": pipe, "converter": None, "refiner": None, "preprocessor": None}

        dtype = torch.bfloat16
        transformer = Flux2Transformer2DModel.from_pretrained(
            self._TRANSFORMER, torch_dtype=dtype, device_map="cpu", cache_dir=_cache_dir,
        )
        text_encoder = Qwen3ForCausalLM.from_pretrained(
            self._TEXT_ENCODER, torch_dtype=dtype, device_map="cpu", cache_dir=_cache_dir,
        )
        pipe = Flux2KleinPipeline.from_pretrained(
            self._BASE_PIPELINE,
            transformer=transformer, text_encoder=text_encoder, torch_dtype=dtype,
            cache_dir=_cache_dir,
        )

        # Consistency LoRA is always active; user LoRAs are appended alongside it
        pipe.load_lora_weights(
            self._CONSISTENCY_LORA,
            weight_name=self._CONSISTENCY_WEIGHTS,
            adapter_name="consistency",
        )
        names   = ["consistency"]
        weights = [1.0]

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
        pipe.set_adapters(names, adapter_weights=weights)

        # KV-cache requires all tensors to stay on device across denoising steps —
        # enable_model_cpu_offload() evicts layers between steps and breaks the cache.
        if gfx_device == "mps":
            pipe.to("mps")
        else:
            pipe.to(gfx_device)
        return {"pipe": pipe, "converter": pipe, "refiner": None, "preprocessor": None}

    def draw_custom_ui(self, col, context) -> bool:
        scene = context.scene
        try:
            col.prop(scene, "input_strips", text="Input")
        except Exception:
            pass
        if scene.sequence_editor is None:
            return True
        #col.label(text="Reference Images:")
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

        seed = inputs.seed
        generator = (
            torch.Generator("cuda").manual_seed(seed)
            if torch.cuda.is_available() and seed != 0 else None
        )
        common = dict(
            prompt=inputs.prompt,
            max_sequence_length=512,
            guidance_scale=inputs.guidance,
            num_inference_steps=inputs.steps,
            height=inputs.height,
            width=inputs.width,
            generator=generator,
        )

        ref_images = []
        for attr in ["klein_strip_1", "klein_strip_2", "klein_strip_3"]:
            strip_name = getattr(scene, attr, None)
            if strip_name:
                strip = find_strip_by_name(scene, strip_name)
                if strip:
                    img = load_strip_as_pil(strip)
                    if img is not None:
                        ref_images.append(img.resize((inputs.width, inputs.height)))

        self.set_phase(inputs, "Generating")
        if inputs.mode == "inpaint":
            if inputs.inpaint_mask is None:
                print("Inpaint: no valid mask image — check that inpaint_selected_strip points to a valid image strip.")
                raise RuntimeError("Inpaint mask not available. Check inpaint_selected_strip.")
            src  = inputs.image.convert("RGB").resize((inputs.width, inputs.height))
            mask = inputs.inpaint_mask.convert("L").resize((inputs.width, inputs.height))
            result = pipe_obj["pipe"](
                prompt=inputs.prompt,
                image=src,
                mask_image=mask,
                num_inference_steps=inputs.steps,
                guidance_scale=inputs.guidance,
                generator=generator,
                strength=1.0,
                callback_on_step_end=self.step_callback(inputs),
            ).images[0]
            if result.size != (inputs.width, inputs.height):
                from PIL import Image as _PILImage
                result = result.resize((inputs.width, inputs.height), _PILImage.LANCZOS)
            return result
        if inputs.mode == "img2img" and inputs.image is not None:
            img = [inputs.image] + ref_images if ref_images else inputs.image
            return pipe_obj["converter"](**common, image=img,
                                         callback_on_step_end=self.step_callback(inputs)).images[0]
        if ref_images:
            return pipe_obj["pipe"](**common, image=ref_images,
                                    callback_on_step_end=self.step_callback(inputs)).images[0]
        return pipe_obj["pipe"](**common, callback_on_step_end=self.step_callback(inputs)).images[0]
