"""Text-to-image, img2img, and inpaint via FLUX.2 Klein 4B with 3 reference image slots."""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram


class Flux2Klein4BPlugin(ModelPlugin):
    MODEL_ID     = "black-forest-labs/FLUX.2-klein-4B"
    DISPLAY_NAME = "FLUX.2 Klein 4B"
    DESCRIPTION  = "Text-to-image via FLUX.2 Klein 4B with up to 3 reference images"
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

    _BASE_PIPELINE = "black-forest-labs/FLUX.2-klein-4B"

    def load(self, prefs, scene, **kw):
        import torch
        from diffusers import Flux2KleinPipeline

        _cache_dir = prefs.hf_cache_dir or None
        mode = kw.get("mode", "txt2img")
        print(f"Loading {self.MODEL_ID} ({mode})…")

        _lfo = prefs.local_files_only
        if mode == "inpaint":
            from diffusers import Flux2KleinInpaintPipeline

            pipe = Flux2KleinInpaintPipeline.from_pretrained(
                self._BASE_PIPELINE, torch_dtype=torch.bfloat16, cache_dir=_cache_dir,
                local_files_only=_lfo,
            )
            if gfx_device == "mps":
                pipe.to("mps")
            elif low_vram():
                pipe.enable_sequential_cpu_offload()
            else:
                pipe.enable_model_cpu_offload()
            return {"pipe": pipe, "converter": None, "refiner": None, "preprocessor": None}

        dtype = torch.bfloat16
        pipe = Flux2KleinPipeline.from_pretrained(
            self._BASE_PIPELINE, torch_dtype=dtype, cache_dir=_cache_dir, local_files_only=_lfo,
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
        # Strip refs live in the scene shown in the VSE (context.sequencer_scene
        # in Blender 5.x), which can differ from the active scene.
        vse_scene = getattr(context, "sequencer_scene", None) or context.scene
        try:
            col.prop(scene, "input_strips", text="Input")
        except Exception:
            pass
        if vse_scene.sequence_editor is None:
            return True
        #col.label(text="Reference Images:")
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
        for attr in (f"klein_strip_{i}_path" for i in range(1, 10)):
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
        # Pass references as a LIST of separate images: Flux2KleinPipeline VAE-encodes
        # each element on its own and assigns it a distinct T-coordinate reference slot,
        # so the model treats each as a real reference (matching the working FLUX.2 Dev
        # plugin). Concatenating them into one wide image loses that per-reference
        # conditioning. A visual strip on the active input (image/video/scene frame)
        # always becomes the first reference, ahead of the named ref slots, regardless
        # of txt2img/img2img mode.
        images = ([inputs.image.convert("RGB")] if inputs.image is not None else []) + ref_images

        pipe_key = "converter" if (inputs.mode == "img2img" and inputs.image is not None) else "pipe"
        if images:
            print(f"Klein {inputs.mode} → pipe images={len(images)} (list of separate refs)")
            return pipe_obj[pipe_key](**common, image=images,
                                      callback_on_step_end=self.step_callback(inputs)).images[0]
        return pipe_obj["pipe"](**common, callback_on_step_end=self.step_callback(inputs)).images[0]
