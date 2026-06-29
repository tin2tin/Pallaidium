"""AI relighting via FLUX Kontext + relight LoRA."""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram, ILLUMINATION_OPTIONS


class KontextRelightPlugin(ModelPlugin):
    MODEL_ID     = "kontext-community/relighting-kontext-dev-lora-v3"
    DISPLAY_NAME = "Kontext Relight"
    MODEL_TYPE   = "image"
    DESCRIPTION  = "AI image relighting via FLUX Kontext + relight LoRA"

    INPUTS       = InputSpec.PROMPT | InputSpec.IMAGE
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.IMAGE_STRIP,
        UISection.RESOLUTION, UISection.FRAMES, UISection.STEPS, UISection.GUIDANCE,
        UISection.ILLUMINATION, UISection.SEED,
    ]
    PARAMS       = ParamSpec(steps=28, guidance=3.5)
    REQUIRED_PACKAGES          = ["torch", "diffusers"]
    supports_inpaint           = False
    supports_img2img           = False
    uses_standard_input_strip  = False

    _BASE_MODEL = "yuvraj108c/FLUX.1-Kontext-dev"

    def load(self, prefs, scene, **kw):
        import torch
        from diffusers import BitsAndBytesConfig, FluxTransformer2DModel, FluxKontextPipeline

        _cache_dir = prefs.hf_cache_dir or None
        print(f"Loading {self.MODEL_ID}…")
        nf4 = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        transformer = FluxTransformer2DModel.from_pretrained(
            self._BASE_MODEL, subfolder="transformer",
            quantization_config=nf4, torch_dtype=torch.bfloat16,
            cache_dir=_cache_dir, local_files_only=prefs.local_files_only,
        )
        pipe = FluxKontextPipeline.from_pretrained(
            self._BASE_MODEL, transformer=transformer, torch_dtype=torch.bfloat16,
            cache_dir=_cache_dir, local_files_only=prefs.local_files_only,
        )
        pipe.load_lora_weights(
            "kontext-community/relighting-kontext-dev-lora-v3",
            weight_name="relighting-kontext-dev-lora-v3.safetensors",
            adapter_name="lora",
        )
        pipe.set_adapters(["lora"], adapter_weights=[0.75])

        if gfx_device == "mps":
            pipe.to("mps")
        elif low_vram():
            pipe.enable_sequential_cpu_offload()
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()
        else:
            pipe.enable_model_cpu_offload()
        return {"pipe": pipe, "converter": pipe, "refiner": None, "preprocessor": None}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch

        if inputs.image is None:
            raise ValueError("Kontext Relight requires an input image.")

        seed = inputs.seed
        generator = (
            torch.Generator("cuda").manual_seed(seed)
            if torch.cuda.is_available() and seed != 0 else None
        )

        if inputs.prompt:
            prompt_desc = inputs.prompt
            style_parts = ["with custom lighting"]
        else:
            illum_style = getattr(scene, "illumination_style", "")
            prompt_desc = ILLUMINATION_OPTIONS.get(illum_style, "")
            style_parts = [f"with {illum_style} lighting"]

        light_dir = getattr(scene, "light_direction", "auto")
        if light_dir != "auto":
            style_parts.append(f"coming from the {light_dir}")

        final_prompt = (
            f"Relight the image {' '.join(style_parts)}. "
            f"{prompt_desc} "
            "Maintain the identity of the foreground subjects."
        )
        print(f"Relight prompt: {final_prompt}")

        self.set_phase(inputs, "Generating")
        return pipe_obj["converter"](
            image=inputs.image,
            prompt=final_prompt,
            num_inference_steps=inputs.steps,
            guidance_scale=inputs.guidance,
            width=inputs.width,
            height=inputs.height,
            generator=generator,
            callback_on_step_end=self.step_callback(inputs),
        ).images[0]
