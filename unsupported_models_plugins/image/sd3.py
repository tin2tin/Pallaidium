"""Text-to-image via Stable Diffusion 3.5 (large and medium variants)."""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram


class _SD3Base(ModelPlugin):
    MODEL_TYPE = "image"
    INPUTS     = InputSpec.PROMPT | InputSpec.NEG_PROMPT
    UI_SECTIONS = [
        UISection.PROMPT, UISection.NEG_PROMPT,
        UISection.RESOLUTION, UISection.STEPS, UISection.GUIDANCE, UISection.SEED,
    ]
    REQUIRED_PACKAGES = ["torch", "diffusers"]
    supports_inpaint  = False

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch

        pipe = pipe_obj["pipe"]
        seed = inputs.seed
        generator = (
            torch.Generator("cuda").manual_seed(seed)
            if torch.cuda.is_available() and seed != 0 else None
        )
        return pipe(
            prompt_3=inputs.prompt,
            negative_prompt=inputs.neg_prompt,
            num_inference_steps=inputs.steps,
            guidance_scale=inputs.guidance,
            height=inputs.height,
            width=inputs.width,
            max_sequence_length=512,
            generator=generator,
        ).images[0]


class SD3MediumPlugin(_SD3Base):
    MODEL_ID     = "adamo1139/stable-diffusion-3.5-medium-ungated"
    DISPLAY_NAME = "Image: Stable Diffusion 3.5 Medium"
    DESCRIPTION  = "SD 3.5 Medium (4-bit quantized)"
    PARAMS       = ParamSpec(steps=28, guidance=4.5)

    def load(self, prefs, scene, **kw):
        import torch
        from diffusers import BitsAndBytesConfig, SD3Transformer2DModel, StableDiffusion3Pipeline

        print(f"Loading {self.MODEL_ID}…")
        nf4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.bfloat16)
        transformer = SD3Transformer2DModel.from_pretrained(
            self.MODEL_ID, subfolder="transformer",
            quantization_config=nf4, torch_dtype=torch.bfloat16,
        )
        pipe = StableDiffusion3Pipeline.from_pretrained(
            self.MODEL_ID, transformer=transformer, torch_dtype=torch.bfloat16
        )
        pipe.enable_model_cpu_offload()
        return {"pipe": pipe, "converter": None, "refiner": None, "preprocessor": None}


class SD3LargePlugin(_SD3Base):
    MODEL_ID     = "adamo1139/stable-diffusion-3.5-large-ungated"
    DISPLAY_NAME = "Image: Stable Diffusion 3.5 Large"
    DESCRIPTION  = "SD 3.5 Large (4-bit quantized, HF token required)"
    PARAMS       = ParamSpec(steps=28, guidance=4.5)
    INPUTS       = InputSpec.PROMPT | InputSpec.NEG_PROMPT | InputSpec.HF_TOKEN

    def load(self, prefs, scene, **kw):
        import torch
        from huggingface_hub.commands.user import login
        from diffusers import BitsAndBytesConfig, SD3Transformer2DModel, StableDiffusion3Pipeline

        login(token=prefs.hugginface_token, add_to_git_credential=True)
        print(f"Loading {self.MODEL_ID}…")
        nf4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.bfloat16)
        transformer = SD3Transformer2DModel.from_pretrained(
            self.MODEL_ID, subfolder="transformer",
            quantization_config=nf4, torch_dtype=torch.bfloat16,
        )
        pipe = StableDiffusion3Pipeline.from_pretrained(
            self.MODEL_ID, transformer=transformer, torch_dtype=torch.bfloat16
        )
        if gfx_device == "mps":
            pipe.to("mps")
        else:
            pipe.enable_model_cpu_offload()
        return {"pipe": pipe, "converter": None, "refiner": None, "preprocessor": None}
