"""Text-to-image, img2img, and inpaint via FLUX.2 Klein 9B with 3 reference image slots."""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram


class Flux2Klein9BPlugin(ModelPlugin):
    MODEL_ID     = "ModelsLab/FLUX.2-klein-9B"
    DISPLAY_NAME = "Image: FLUX.2 Klein 9B"
    DESCRIPTION  = "Text-to-image via FLUX.2 Klein 9B (quantized) with up to 3 reference images"
    MODEL_TYPE   = "image"
    INPUTS       = InputSpec.PROMPT | InputSpec.IMAGE | InputSpec.LORA
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.IMAGE_STRIP,
        UISection.RESOLUTION, UISection.FRAMES, UISection.STEPS, UISection.GUIDANCE,
        UISection.IMAGE_STRENGTH, UISection.SEED,
        UISection.LORA,
    ]
    PARAMS            = ParamSpec(steps=4, guidance=1.0)
    REQUIRED_PACKAGES = ["torch", "diffusers", "transformers"]
    supports_inpaint       = True
    inpaint_uses_strength  = True

    _BASE_PIPELINE = "ModelsLab/FLUX.2-klein-9B"
    _TRANSFORMER   = "OzzyGT/flux2_klein_9B_bnb_4bit_transformer"
    _TEXT_ENCODER  = "OzzyGT/flux2_klein_9B_bnb_4bit_text_encoder"

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
        try:
            col.prop(scene, "input_strips", text="Input")
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

        from PIL import Image as _PILImage
        ref_images = []
        for attr in ["klein_strip_1_path", "klein_strip_2_path", "klein_strip_3_path"]:
            path = getattr(scene, attr, None)
            if path:
                try:
                    img = _PILImage.open(path).convert("RGB")
                    ref_images.append(img)
                    print(f"Klein ref loaded: {attr} = '{path}' {img.size}")
                except Exception as e:
                    print(f"Klein ref failed to open '{path}': {e}")
            else:
                print(f"Klein ref empty: {attr}")
        print(f"Klein: {len(ref_images)} reference image(s) loaded, mode={inputs.mode}")

        self.set_phase(inputs, "Generating")
        if inputs.mode == "inpaint":
            if inputs.inpaint_mask is None:
                print("Inpaint: no valid mask image — check that inpaint_selected_strip points to a valid image strip.")
                raise RuntimeError("Inpaint mask not available. Check inpaint_selected_strip.")
            if len(ref_images) > 1:
                print(f"Inpaint: {len(ref_images)} reference images set; only the first is supported by the inpaint pipeline.")
            src  = inputs.image.convert("RGB").resize((inputs.width, inputs.height))
            mask = inputs.inpaint_mask.convert("L").resize((inputs.width, inputs.height))
            result = pipe_obj["pipe"](
                prompt=inputs.prompt,
                image=src,
                mask_image=mask,
                image_reference=ref_images[0] if ref_images else None,
                max_sequence_length=512,
                num_inference_steps=inputs.steps,
                guidance_scale=inputs.guidance,
                generator=generator,
                strength=inputs.strength,
                callback_on_step_end=self.step_callback(inputs),
            ).images[0]
            if result.size != (inputs.width, inputs.height):
                result = result.resize((inputs.width, inputs.height), _PILImage.LANCZOS)
            return result
        # Klein was trained with a single horizontally-concatenated reference image per the
        # official BFL sampling code — separate T-coordinate slots (T=20, T=30…) have zero
        # effect. Always concatenate before passing.
        cat = pipe_obj["pipe"].image_processor.concatenate_images
        if inputs.mode == "img2img" and inputs.image is not None:
            src_rgb = inputs.image.convert("RGB")
            if ref_images:
                # Resize input to match the first ref so concatenate_images
                # doesn't introduce white-padding between tiles.
                src_rgb = src_rgb.resize(ref_images[0].size, _PILImage.LANCZOS)
                ref_list = [src_rgb] + ref_images
            else:
                ref_list = [src_rgb]
            ref_input = cat(ref_list)
            print(f"Klein img2img → pipe image={ref_input.size}, refs={len(ref_images)} (+input)")
            return pipe_obj["converter"](**common, image=ref_input,
                                         callback_on_step_end=self.step_callback(inputs)).images[0]
        if ref_images:
            ref_input = cat(ref_images)
            print(f"Klein txt2img → pipe image={ref_input.size}, refs={len(ref_images)}")
            return pipe_obj["pipe"](**common, image=ref_input,
                                    callback_on_step_end=self.step_callback(inputs)).images[0]
        return pipe_obj["pipe"](**common, callback_on_step_end=self.step_callback(inputs)).images[0]
