"""Text-to-image, img2img, and inpaint via FLUX.1-dev (NF4 quantized, macOS mflux)."""

import os
import platform
from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram


class FluxDevPlugin(ModelPlugin):
    MODEL_ID     = "ChuckMcSneed/FLUX.1-dev"
    DISPLAY_NAME = "Image: FLUX.1 Dev"
    MODEL_TYPE   = "image"
    DESCRIPTION  = "High-quality text-to-image, img2img, and inpaint via FLUX.1-dev (NF4 quantized)"

    INPUTS       = InputSpec.PROMPT | InputSpec.IMAGE | InputSpec.LORA
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.IMAGE_STRIP,
        UISection.RESOLUTION, UISection.STEPS, UISection.GUIDANCE,
        UISection.IMAGE_STRENGTH, UISection.SEED,
        UISection.LORA,
    ]
    PARAMS       = ParamSpec(steps=20, guidance=3.5)
    REQUIRED_PACKAGES = ["torch", "diffusers"]

    def load(self, prefs, scene, **kw):
        import torch

        mode = kw.get("mode", "txt2img")
        print(f"Loading {self.MODEL_ID} ({mode})…")

        if platform.system() == "Darwin":
            from huggingface_hub.commands.user import login
            login(token=prefs.hugginface_token, add_to_git_credential=True)
            from mflux import Flux1
            pipe = Flux1.from_name(model_name="dev", quantize=4)
            return {"pipe": pipe, "converter": pipe, "refiner": None, "preprocessor": None}

        if mode == "inpaint":
            from diffusers import DiffusionPipeline, FluxFillPipeline, FluxTransformer2DModel
            from transformers import T5EncoderModel

            orig = DiffusionPipeline.from_pretrained(self.MODEL_ID, torch_dtype=torch.bfloat16)
            transformer = FluxTransformer2DModel.from_pretrained(
                "sayakpaul/FLUX.1-Fill-dev-nf4", subfolder="transformer",
                torch_dtype=torch.bfloat16,
            )
            text_enc_2 = T5EncoderModel.from_pretrained(
                "sayakpaul/FLUX.1-Fill-dev-nf4", subfolder="text_encoder_2",
                torch_dtype=torch.bfloat16,
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

        from diffusers import BitsAndBytesConfig, FluxTransformer2DModel, FluxPipeline

        nf4 = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        transformer = FluxTransformer2DModel.from_pretrained(
            self.MODEL_ID, subfolder="transformer",
            quantization_config=nf4, torch_dtype=torch.bfloat16,
        )
        pipe = FluxPipeline.from_pretrained(
            self.MODEL_ID, transformer=transformer, torch_dtype=torch.bfloat16
        )
        if gfx_device == "mps":
            pipe.to("mps")
        elif low_vram():
            pipe.enable_model_cpu_offload()
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()
        else:
            pipe.enable_model_cpu_offload()

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

        return {"pipe": pipe, "converter": pipe, "refiner": None, "preprocessor": None}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch

        seed = inputs.seed
        generator = (
            torch.Generator("cuda").manual_seed(seed)
            if torch.cuda.is_available() and seed != 0 else None
        )

        if platform.system() == "Darwin":
            from mflux import Config
            pipe = pipe_obj["pipe"]
            if inputs.mode == "img2img" and inputs.image is not None:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    img_path = f.name
                inputs.image.save(img_path)
                try:
                    return pipe.generate_image(
                        seed=abs(int(seed)),
                        prompt=inputs.prompt,
                        config=Config(
                            num_inference_steps=inputs.steps,
                            height=inputs.height,
                            width=inputs.width,
                            image_path=img_path,
                            image_strength=1.0 - inputs.strength,
                        ),
                    )
                finally:
                    os.unlink(img_path)
            return pipe.generate_image(
                seed=abs(int(seed)),
                prompt=inputs.prompt,
                config=Config(
                    num_inference_steps=inputs.steps,
                    height=inputs.height,
                    width=inputs.width,
                ),
            )

        if inputs.mode == "inpaint" and inputs.image is not None and inputs.inpaint_mask is not None:
            return pipe_obj["pipe"](
                prompt=inputs.prompt,
                max_sequence_length=512,
                image=inputs.image,
                mask_image=inputs.inpaint_mask,
                num_inference_steps=inputs.steps,
                guidance_scale=inputs.guidance,
                height=inputs.height,
                width=inputs.width,
                generator=generator,
            ).images[0]

        if inputs.mode == "img2img" and inputs.image is not None:
            return pipe_obj["converter"](
                prompt=inputs.prompt,
                max_sequence_length=512,
                image=inputs.image,
                strength=1.0 - inputs.strength,
                num_inference_steps=inputs.steps,
                guidance_scale=inputs.guidance,
                height=inputs.height,
                width=inputs.width,
                generator=generator,
            ).images[0]

        return pipe_obj["pipe"](
            prompt=inputs.prompt,
            prompt_2=None,
            max_sequence_length=512,
            num_inference_steps=inputs.steps,
            guidance_scale=inputs.guidance,
            height=inputs.height,
            width=inputs.width,
            generator=generator,
        ).images[0]
