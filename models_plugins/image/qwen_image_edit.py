"""Multi-image editing via Qwen-Image-Edit-2511."""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram, find_strip_by_name, get_strip_path, load_first_frame, load_strip_as_pil


class QwenImageEditPlugin(ModelPlugin):
    MODEL_ID     = "Qwen/Qwen-Image-Edit-2511"
    DISPLAY_NAME = "Image: Qwen Image Edit (multi-image)"
    MODEL_TYPE   = "image"
    DESCRIPTION  = "Multi-image instruction editing via Qwen-Image-Edit-2511"

    INPUTS       = InputSpec.PROMPT | InputSpec.NEG_PROMPT | InputSpec.MULTI_IMAGE | InputSpec.LORA
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.NEG_PROMPT, UISection.MULTI_IMAGES,
        UISection.FRAMES, UISection.STEPS, UISection.SEED, UISection.LORA,
    ]
    PARAMS       = ParamSpec(steps=4, max_multi_images=3)
    REQUIRED_PACKAGES          = ["torch", "diffusers", "transformers"]
    supports_inpaint           = False
    supports_img2img           = True
    requires_input_strip       = True
    uses_standard_input_strip  = False

    def load(self, prefs, scene, **kw):
        import torch
        from transformers import BitsAndBytesConfig as TBnB, Qwen2_5_VLForConditionalGeneration
        from diffusers import BitsAndBytesConfig as DBnB, QwenImageEditPlusPipeline, QwenImageTransformer2DModel

        _cache_dir = prefs.hf_cache_dir or None
        model_id = self.MODEL_ID
        dtype = torch.bfloat16
        print(f"Loading {model_id}…")

        q_transformer = DBnB(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=dtype,
                              llm_int8_skip_modules=["transformer_blocks.0.img_mod"])
        q_text = TBnB(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)

        transformer = QwenImageTransformer2DModel.from_pretrained(
            model_id, subfolder="transformer", quantization_config=q_transformer,
            torch_dtype=dtype, cache_dir=_cache_dir,
        ).to("cpu")
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, subfolder="text_encoder", quantization_config=q_text,
            torch_dtype=dtype, cache_dir=_cache_dir,
        ).to("cpu")

        pipe = QwenImageEditPlusPipeline.from_pretrained(
            model_id, transformer=transformer, text_encoder=text_encoder, torch_dtype=dtype,
            cache_dir=_cache_dir,
        )
        pipe.load_lora_weights(
            "lightx2v/Qwen-Image-Edit-2511-Lightning",
            weight_name="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        )
        # from sdnq import SDNQConfig  # noqa: F401
        # from sdnq.common import use_torch_compile as triton_is_available
        # from sdnq.loader import apply_sdnq_options_to_model
        # from transformers import Qwen2_5_VLForConditionalGeneration
        # torch_dtype = torch.bfloat16


        # text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #     "Disty0/Qwen-Image-Edit-2511-SDNQ-uint4-svd-r32",
        #     subfolder="text_encoder",
        #     dtype=torch_dtype,
        #     device_map="cpu",
        # )

        # transformer = QwenImageTransformer2DModel.from_pretrained(
        #     "Disty0/Qwen-Image-Edit-2511-SDNQ-uint4-svd-r32",
        #     subfolder="transformer",
        #     torch_dtype=torch_dtype,
        #     device_map="cpu",
        # )


        # pipe = QwenImageEditPlusPipeline.from_pretrained(
        #     "Qwen/Qwen-Image-Edit-2511", transformer=transformer, text_encoder=text_encoder, torch_dtype=torch_dtype
        # )

        # if triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
        #     pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
        #     pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)

        # Apply user LoRAs
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
        elif low_vram():
            pipe.enable_sequential_cpu_offload()
            pipe.vae.enable_tiling()
        else:
            pipe.enable_model_cpu_offload()
        return {"pipe": pipe, "converter": None, "refiner": None, "preprocessor": None}

    def draw_custom_ui(self, col, context) -> bool:
        scene = context.scene
        try:
            col.prop(scene, "input_strips", text="Input")
        except Exception:
            pass
        if scene.sequence_editor is None:
            return True
        for attr, action in [
            ("qwen_strip_1", "qwen_select1"),
            ("qwen_strip_2", "qwen_select2"),
            ("qwen_strip_3", "qwen_select3"),
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

        pipe = pipe_obj["pipe"]
        seed = inputs.seed
        generator = (
            torch.Generator("cuda").manual_seed(seed)
            if torch.cuda.is_available() and seed != 0 else None
        )

        qwen_images = []
        if inputs.image is not None:
            qwen_images.append(inputs.image)
        for attr in ["qwen_strip_1", "qwen_strip_2", "qwen_strip_3"]:
            strip_name = getattr(scene, attr, None)
            if strip_name:
                strip = find_strip_by_name(scene, strip_name)
                if strip:
                    qwen_images.append(load_strip_as_pil(strip))

        if not qwen_images:
            raise ValueError(
                "Qwen Image Edit requires at least one image input. "
                "Select an image or movie strip and set Input to 'Strip', "
                "or pick strips using the image pickers below."
            )

        self.set_phase(inputs, "Generating")
        with torch.inference_mode():
            return pipe(
                image=qwen_images,
                prompt=inputs.prompt,
                generator=generator,
                true_cfg_scale=4.0,
                negative_prompt=inputs.neg_prompt + " ",
                num_inference_steps=inputs.steps,
                num_images_per_prompt=1,
                callback_on_step_end=self.step_callback(inputs),
            ).images[0]
