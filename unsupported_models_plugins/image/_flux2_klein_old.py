"""Text-to-image and img2img via FLUX.2 Klein (9B quantized, two MODEL_ID aliases)."""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram, find_strip_by_name, get_strip_path, load_first_frame


class _Flux2KleinBase(ModelPlugin):
    MODEL_TYPE  = "image"
    INPUTS      = InputSpec.PROMPT | InputSpec.IMAGE | InputSpec.LORA
    UI_SECTIONS = [
        UISection.PROMPT, UISection.IMAGE_STRIP,
        UISection.RESOLUTION, UISection.FRAMES, UISection.STEPS, UISection.GUIDANCE, UISection.SEED,
        UISection.LORA,
    ]
    PARAMS      = ParamSpec(steps=4, guidance=1.0)
    REQUIRED_PACKAGES = ["torch", "diffusers", "transformers"]
    supports_inpaint  = True

    _BASE_PIPELINE = "black-forest-labs/FLUX.2-klein-9b-kv"
    _TRANSFORMER   = "OzzyGT/flux2_klein_9B_bnb_4bit_transformer"
    _TEXT_ENCODER  = "OzzyGT/flux2_klein_9B_bnb_4bit_text_encoder"

    def _build_klein_pipe(self, cache_dir=None):
        import torch
        from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
        from transformers import Qwen3ForCausalLM

        dtype = torch.bfloat16
        transformer = Flux2Transformer2DModel.from_pretrained(
            self._TRANSFORMER, torch_dtype=dtype, device_map="cpu", cache_dir=cache_dir,
        )
        text_encoder = Qwen3ForCausalLM.from_pretrained(
            self._TEXT_ENCODER, torch_dtype=dtype, device_map="cpu", cache_dir=cache_dir,
        )
        pipe = Flux2KleinPipeline.from_pretrained(
            self._BASE_PIPELINE,
            transformer=transformer, text_encoder=text_encoder, torch_dtype=dtype,
            cache_dir=cache_dir,
        )
        if gfx_device == "mps":
            pipe.to("mps")
        else:
            pipe.enable_model_cpu_offload()
        return pipe

    def load(self, prefs, scene, **kw):
        import torch

        _cache_dir = prefs.hf_cache_dir or None
        mode = kw.get("mode", "txt2img")
        print(f"Loading {self.MODEL_ID} ({mode})…")

        if mode == "inpaint":
            from diffusers import DiffusionPipeline, FluxFillPipeline, FluxTransformer2DModel
            from transformers import T5EncoderModel

            orig = DiffusionPipeline.from_pretrained(
                self._BASE_PIPELINE, torch_dtype=torch.bfloat16, cache_dir=_cache_dir,
            )
            transformer = FluxTransformer2DModel.from_pretrained(
                "sayakpaul/FLUX.1-Fill-dev-nf4", subfolder="transformer",
                torch_dtype=torch.bfloat16, cache_dir=_cache_dir,
            )
            text_enc_2 = T5EncoderModel.from_pretrained(
                "sayakpaul/FLUX.1-Fill-dev-nf4", subfolder="text_encoder_2",
                torch_dtype=torch.bfloat16, cache_dir=_cache_dir,
            )
            pipe = FluxFillPipeline.from_pipe(
                orig, transformer=transformer, text_encoder_2=text_enc_2,
                torch_dtype=torch.bfloat16,
            )
            if gfx_device == "mps":
                pipe.to("mps")
            elif low_vram():
                pipe.enable_sequential_cpu_offload()
                pipe.vae.enable_tiling()
            else:
                pipe.enable_model_cpu_offload()
            return {"pipe": pipe, "converter": None, "refiner": None, "preprocessor": None}

        enabled_items = kw.get("enabled_items", [])
        pipe = self._build_klein_pipe(cache_dir=_cache_dir)
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
        return {"pipe": pipe, "converter": pipe, "refiner": None, "preprocessor": None}

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
            guidance_scale=1.0,
            num_inference_steps=4,
            height=inputs.height,
            width=inputs.width,
            generator=generator,
        )
        if inputs.mode == "inpaint" and inputs.image is not None and inputs.inpaint_mask is not None:
            return pipe_obj["pipe"](
                **common, image=inputs.image, mask_image=inputs.inpaint_mask,
            ).images[0]
        if inputs.mode == "img2img" and inputs.image is not None:
            return pipe_obj["converter"](**common, image=inputs.image).images[0]
        return pipe_obj["pipe"](**common).images[0]


class Flux2Klein9BKVPlugin(_Flux2KleinBase):
    MODEL_ID     = "dx8152/Flux2-Klein-9B-Consistency"
    DISPLAY_NAME = "FLUX.2 Klein 9B KV (Consistency)"
    DESCRIPTION  = "Multi-reference image generation via FLUX.2 Klein 9B KV-cache with consistency LoRA"

    _CONSISTENCY_LORA    = "dx8152/Flux2-Klein-9B-Consistency"
    _CONSISTENCY_WEIGHTS = "Flux2-Klein-9B-consistency-V2.safetensors"

    def _build_klein_pipe(self, cache_dir=None):
        import torch
        from diffusers import Flux2KleinKVPipeline, Flux2Transformer2DModel
        from transformers import Qwen3ForCausalLM

        dtype = torch.bfloat16
        transformer = Flux2Transformer2DModel.from_pretrained(
            self._TRANSFORMER, torch_dtype=dtype, device_map="cpu", cache_dir=cache_dir,
        )
        text_encoder = Qwen3ForCausalLM.from_pretrained(
            self._TEXT_ENCODER, torch_dtype=dtype, device_map="cpu", cache_dir=cache_dir,
        )
        pipe = Flux2KleinKVPipeline.from_pretrained(
            self._BASE_PIPELINE,
            transformer=transformer, text_encoder=text_encoder, torch_dtype=dtype,
            cache_dir=cache_dir,
        )
        pipe.load_lora_weights(
            self._CONSISTENCY_LORA,
            weight_name=self._CONSISTENCY_WEIGHTS,
            adapter_name="consistency",
        )
        return pipe

    def load(self, prefs, scene, **kw):
        _cache_dir = prefs.hf_cache_dir or None
        mode = kw.get("mode", "txt2img")
        print(f"Loading {self.MODEL_ID} ({mode})…")

        if mode == "inpaint":
            return super().load(prefs, scene, **kw)

        pipe = self._build_klein_pipe(cache_dir=_cache_dir)

        # Always keep the consistency LoRA active; append any user LoRAs alongside it
        enabled_items = kw.get("enabled_items", [])
        names   = ["consistency"]
        weights = [1.0]
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

        # KV-cache pipelines require all tensors to stay on the same device across
        # denoising steps — enable_model_cpu_offload() breaks this by evicting layers
        # between steps, so the cached K/V tensors are no longer reachable.
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
        if scene.sequence_editor is None or scene.input_strips != "input_strips":
            return True
        col.label(text="Reference Images:")
        for attr, action in [
            ("klein_strip_1", "klein_select1"),
            ("klein_strip_2", "klein_select2"),
            ("klein_strip_3", "klein_select3"),
        ]:
            row = col.row(align=True)
            row.prop_search(
                scene, attr, scene.sequence_editor, "strips",
                text="", icon="FILE_IMAGE",
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

        # Inpaint uses FluxFillPipeline (loaded by base class), which needs guidance_scale
        if inputs.mode == "inpaint" and inputs.image is not None and inputs.inpaint_mask is not None:
            return pipe_obj["pipe"](
                prompt=inputs.prompt,
                max_sequence_length=512,
                guidance_scale=1.0,
                num_inference_steps=4,
                height=inputs.height,
                width=inputs.width,
                generator=generator,
                image=inputs.image,
                mask_image=inputs.inpaint_mask,
            ).images[0]

        # Flux2KleinKVPipeline has no guidance_scale parameter
        common = dict(
            prompt=inputs.prompt,
            max_sequence_length=512,
            num_inference_steps=4,
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
                    img = load_first_frame(get_strip_path(strip))
                    if img is not None:
                        ref_images.append(img.resize((inputs.width, inputs.height)))

        if inputs.mode == "img2img" and inputs.image is not None:
            all_images = [inputs.image] + ref_images if ref_images else inputs.image
            return pipe_obj["converter"](**common, image=all_images).images[0]
        if ref_images:
            return pipe_obj["pipe"](**common, image=ref_images).images[0]
        return pipe_obj["pipe"](**common).images[0]


class Flux2KleinBasePlugin(_Flux2KleinBase):
    MODEL_ID     = "Runware/BFL-FLUX.2-klein-base-4B"
    DISPLAY_NAME = "FLUX.2 Klein 4B"
    DESCRIPTION  = "Fast text-to-image via FLUX.2 Klein 4B (quantized)"


class Flux2Klein9BPlugin(_Flux2KleinBase):
    MODEL_ID     = "black-forest-labs/FLUX.2-klein-9b-kv"
    DISPLAY_NAME = "FLUX.2 Klein 9B"
    DESCRIPTION  = "Fast text-to-image via FLUX.2 Klein 9B (quantized)"

    def draw_custom_ui(self, col, context) -> bool:
        scene = context.scene
        try:
            col.prop(scene, "input_strips", text="Input")
        except Exception:
            pass
        if scene.sequence_editor is None or scene.input_strips != "input_strips":
            return True
        col.label(text="Reference Images:")
        for attr, action in [
            ("klein_strip_1", "klein_select1"),
            ("klein_strip_2", "klein_select2"),
            ("klein_strip_3", "klein_select3"),
        ]:
            row = col.row(align=True)
            row.prop_search(
                scene, attr, scene.sequence_editor, "strips",
                text="", icon="FILE_IMAGE",
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
            guidance_scale=1.0,
            num_inference_steps=4,
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
                    ref_images.append(load_first_frame(get_strip_path(strip)))

        if inputs.mode == "inpaint" and inputs.image is not None and inputs.inpaint_mask is not None:
            return pipe_obj["pipe"](
                **common, image=inputs.image, mask_image=inputs.inpaint_mask,
            ).images[0]
        if inputs.mode == "img2img" and inputs.image is not None:
            img = [inputs.image] + ref_images if ref_images else inputs.image
            return pipe_obj["converter"](**common, image=img).images[0]
        if ref_images:
            return pipe_obj["pipe"](**common, image=ref_images).images[0]
        return pipe_obj["pipe"](**common).images[0]
