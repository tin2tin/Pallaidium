"""Text-to-image, img2img, and inpaint via FLUX.2 Klein 9B KV-cache with consistency LoRA and up to 9 reference image slots."""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram


class Flux2Klein9BKVPlugin(ModelPlugin):
    MODEL_ID     = "dx8152/Flux2-Klein-9B-Consistency"
    DISPLAY_NAME = "FLUX.2 Klein 9B KV (Consistency)"
    DESCRIPTION  = "Multi-reference image generation via FLUX.2 Klein 9B KV-cache with consistency LoRA"
    MODEL_TYPE   = "image"
    INPUTS       = InputSpec.PROMPT | InputSpec.IMAGE | InputSpec.LORA
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.IMAGE_STRIP,
        UISection.RESOLUTION, UISection.FRAMES, UISection.STEPS, UISection.SEED,
        UISection.IMAGE_STRENGTH,
        UISection.LORA,
    ]
    PARAMS            = ParamSpec(steps=4, guidance=1.0)
    REQUIRED_PACKAGES = ["torch", "diffusers", "transformers", "sdnq"]
    supports_inpaint       = True
    inpaint_uses_strength  = True
    strip_power_inpaint_only = True  # Flux2KleinKVPipeline has no guidance_scale/strength
                                      # param — image= is KV-cached reference conditioning,
                                      # not a denoise blend.

    # black-forest-labs/FLUX.2-klein-9b-kv is a separately step-distilled
    # checkpoint (4 steps, no CFG) — NOT just the base 9B model wrapped in a
    # caching pipeline class. It needs its OWN quantized weights; the base
    # model's BNB transformer (used below for inpaint) is the wrong checkpoint
    # for this pipeline and silently produces weak/garbled output.
    _BASE_PIPELINE       = "GeneralShan/FLUX.2-klein-9B-KV-SDNQ-4bit-dynamic-svd-r32"
    # Flux2KleinInpaintPipeline has no -kv variant, so inpaint mode runs the
    # base (non-distilled) architecture instead — same checkpoint pairing as
    # the non-KV plugin.
    _BASE_PIPELINE_INPAINT = "ModelsLab/FLUX.2-klein-9B"
    _TRANSFORMER_INPAINT   = "OzzyGT/flux2_klein_9B_bnb_4bit_transformer"
    _TEXT_ENCODER_INPAINT  = "OzzyGT/flux2_klein_9B_bnb_4bit_text_encoder"
    _CONSISTENCY_LORA    = "dx8152/Flux2-Klein-9B-Consistency"
    _CONSISTENCY_WEIGHTS = "Flux2-Klein-9B-consistency-V2.safetensors"

    def load(self, prefs, scene, **kw):
        import torch
        from diffusers import Flux2KleinKVPipeline, Flux2Transformer2DModel
        from transformers import Qwen3ForCausalLM

        _cache_dir = prefs.hf_cache_dir or None
        mode = kw.get("mode", "txt2img")

        _lfo = prefs.local_files_only
        if mode == "inpaint":
            print(f"Loading {self._BASE_PIPELINE_INPAINT} + consistency LoRA ({mode})…")
            from diffusers import Flux2KleinInpaintPipeline, Flux2Transformer2DModel

            try:
                from transformers import BitsAndBytesConfig
                _bnb4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            except Exception:
                _bnb4 = None
            _bnb_kw = {"quantization_config": _bnb4} if _bnb4 is not None else {}

            transformer = Flux2Transformer2DModel.from_pretrained(
                self._TRANSFORMER_INPAINT, torch_dtype=torch.bfloat16, device_map="cpu", cache_dir=_cache_dir,
                local_files_only=_lfo, **_bnb_kw,
            )
            text_encoder = Qwen3ForCausalLM.from_pretrained(
                self._TEXT_ENCODER_INPAINT, torch_dtype=torch.bfloat16, device_map="cpu", cache_dir=_cache_dir,
                local_files_only=_lfo, **_bnb_kw,
            )
            pipe = Flux2KleinInpaintPipeline.from_pretrained(
                self._BASE_PIPELINE_INPAINT,
                transformer=transformer, text_encoder=text_encoder,
                torch_dtype=torch.bfloat16, cache_dir=_cache_dir, local_files_only=_lfo,
            )
            if gfx_device == "mps":
                pipe.to("mps")
            elif low_vram():
                pipe.enable_sequential_cpu_offload()
            else:
                pipe.enable_model_cpu_offload()
            return {"pipe": pipe, "converter": None, "refiner": None, "preprocessor": None}

        print(f"Loading {self._BASE_PIPELINE} + consistency LoRA ({mode})…")

        # Registers SDNQ's quantizer with diffusers/transformers so from_pretrained
        # recognizes quant_method="sdnq" in the configs and wraps the weights with
        # the dequantizer. Without this import the quantized tensors load raw (no
        # dequant) → pure noise. See krea2_turbo.py for the same pattern.
        from sdnq import SDNQConfig  # noqa: F401

        dtype = torch.bfloat16
        transformer = Flux2Transformer2DModel.from_pretrained(
            self._BASE_PIPELINE, subfolder="transformer", torch_dtype=dtype,
            cache_dir=_cache_dir, local_files_only=_lfo,
        )
        text_encoder = Qwen3ForCausalLM.from_pretrained(
            self._BASE_PIPELINE, subfolder="text_encoder", torch_dtype=dtype,
            cache_dir=_cache_dir, local_files_only=_lfo,
        )
        pipe = Flux2KleinKVPipeline.from_pretrained(
            self._BASE_PIPELINE,
            transformer=transformer, text_encoder=text_encoder, torch_dtype=dtype,
            cache_dir=_cache_dir, local_files_only=_lfo,
        )

        # Consistency LoRA is always active; user LoRAs are appended alongside it
        pipe.load_lora_weights(
            self._CONSISTENCY_LORA,
            weight_name=self._CONSISTENCY_WEIGHTS,
            adapter_name="consistency",
        )
        print(f"Klein KV: consistency LoRA loaded ({self._CONSISTENCY_LORA}, weight=1.00)")
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
                print(f"Klein KV: user LoRA '{item.name}' loaded (adapter='{name}', weight={item.weight_value:.2f})")
        pipe.set_adapters(names, adapter_weights=weights)
        print(f"Klein KV: active adapters={names} weights={weights}")

        # KV-cache requires all tensors to stay on device across denoising steps —
        # enable_model_cpu_offload() evicts layers between steps and breaks the cache.
        if gfx_device == "mps":
            pipe.to("mps")
        else:
            pipe.to(gfx_device)
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
        # Flux2KleinKVPipeline has no guidance_scale param — it's a step-distilled
        # KV-cache pipeline; passing it would raise a TypeError.
        common = dict(
            prompt=inputs.prompt,
            max_sequence_length=512,
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
        print(f"Klein KV: {len(ref_images)} reference image(s) loaded, mode={inputs.mode}")

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
        # Flux2KleinKVPipeline's image= is a reference-conditioning list, not a
        # denoise blend (no strength param) — same as the non-KV Klein pipeline.
        # The active input (image/video/scene frame) is always the first
        # reference, ahead of the named ref slots, regardless of txt2img/img2img.
        images = ([inputs.image.convert("RGB")] if inputs.image is not None else []) + ref_images

        pipe_key = "converter" if (inputs.mode == "img2img" and inputs.image is not None) else "pipe"
        if images:
            print(f"Klein KV {inputs.mode} → pipe images={len(images)} (list of separate refs)")
            return pipe_obj[pipe_key](**common, image=images,
                                      callback_on_step_end=self.step_callback(inputs)).images[0]
        return pipe_obj["pipe"](**common, callback_on_step_end=self.step_callback(inputs)).images[0]
