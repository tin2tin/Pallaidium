"""Text-to-image via Chroma (lodestones/Chroma)."""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram


class ChromaPlugin(ModelPlugin):
    MODEL_ID     = "lodestones/Chroma"
    DISPLAY_NAME = "Image: Chroma"
    MODEL_TYPE   = "image"
    DESCRIPTION  = "Text-to-image via Chroma (text-only, no img2img/inpaint)"

    INPUTS       = InputSpec.PROMPT | InputSpec.LORA
    UI_SECTIONS  = [
        UISection.PROMPT,
        UISection.RESOLUTION, UISection.STEPS, UISection.GUIDANCE, UISection.SEED,
        UISection.LORA,
    ]
    PARAMS       = ParamSpec(steps=30, guidance=5.0)
    REQUIRED_PACKAGES = ["torch", "diffusers"]

    def load(self, prefs, scene, **kw):
        import torch
        from diffusers import ChromaPipeline
        from diffusers.quantizers import PipelineQuantizationConfig

        print("Loading Chroma…")
        dtype = torch.bfloat16
        if gfx_device == "mps" or low_vram():
            print("Chroma: 4-bit quant")
            quant_config = PipelineQuantizationConfig(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs={
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": dtype,
                    "llm_int8_skip_modules": ["distilled_guidance_layer"],
                },
                components_to_quantize=["transformer", "text_encoder"],
            )
            pipe = ChromaPipeline.from_pretrained(
                "imnotednamode/Chroma-v36-dc-diffusers",
                quantization_config=quant_config,
                torch_dtype=dtype,
            )
            if gfx_device == "mps":
                pipe.to("mps")
            else:
                pipe.enable_model_cpu_offload()
                pipe.vae.enable_slicing()
                pipe.vae.enable_tiling()
        else:
            print("Chroma: 8-bit quant")
            quant_config = PipelineQuantizationConfig(
                quant_backend="bitsandbytes_8bit",
                quant_kwargs={"load_in_8bit": True},
                components_to_quantize=["transformer", "text_encoder_2"],
            )
            pipe = ChromaPipeline.from_pretrained(
                "imnotednamode/Chroma-v36-dc-diffusers",
                quantization_config=quant_config,
                torch_dtype=dtype,
            )
            pipe.to("cuda")
        return {"pipe": pipe, "converter": None, "refiner": None, "preprocessor": None}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch

        pipe = pipe_obj["pipe"]
        seed = inputs.seed
        generator = (
            torch.Generator("cuda").manual_seed(seed)
            if torch.cuda.is_available() and seed != 0
            else (torch.Generator(device=gfx_device).manual_seed(seed) if seed != 0 else None)
        )
        return pipe(
            prompt=inputs.prompt,
            num_inference_steps=inputs.steps,
            guidance_scale=inputs.guidance,
            height=inputs.height,
            width=inputs.width,
            generator=generator,
        ).images[0]
