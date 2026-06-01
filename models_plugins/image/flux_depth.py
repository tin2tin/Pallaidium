"""FLUX Depth ControlNet (romanfratric234/FLUX.1-Depth-dev-lora)."""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram


class FluxDepthPlugin(ModelPlugin):
    MODEL_ID     = "romanfratric234/FLUX.1-Depth-dev-lora"
    DISPLAY_NAME = "Image: FLUX Depth ControlNet"
    MODEL_TYPE   = "image"
    DESCRIPTION  = "Depth-guided generation via FLUX.1 Depth ControlNet"

    INPUTS       = InputSpec.PROMPT | InputSpec.IMAGE | InputSpec.LORA
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.IMAGE_STRIP,
        UISection.RESOLUTION, UISection.FRAMES, UISection.STEPS, UISection.GUIDANCE,
        UISection.IMAGE_STRENGTH, UISection.SEED,
        UISection.LORA,
    ]
    PARAMS       = ParamSpec(steps=28, guidance=3.5)
    REQUIRED_PACKAGES = ["torch", "diffusers", "image_gen_aux"]
    supports_inpaint  = False
    supports_img2img  = True

    def load(self, prefs, scene, **kw):
        import torch
        from diffusers import BitsAndBytesConfig, FluxTransformer2DModel, FluxControlPipeline
        from image_gen_aux import DepthPreprocessor

        _cache_dir = prefs.hf_cache_dir or None
        print("Loading FLUX Depth ControlNet…")
        pipecard = "ChuckMcSneed/FLUX.1-dev"
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_nf4 = FluxTransformer2DModel.from_pretrained(
            pipecard, subfolder="transformer",
            quantization_config=nf4_config, torch_dtype=torch.bfloat16,
            cache_dir=_cache_dir,
        )
        pipe = FluxControlPipeline.from_pretrained(
            pipecard, transformer=model_nf4, torch_dtype=torch.bfloat16,
            local_files_only=prefs.local_files_only, cache_dir=_cache_dir,
        )
        pipe.load_lora_weights(self.MODEL_ID, adapter_name="depth_control")

        enabled_items = kw.get("enabled_items", [])
        names, weights = ["depth_control"], [1.0]
        if enabled_items:
            from ...utils.helpers import clean_filename, bpy
            lora_folder = getattr(bpy.context.scene, "lora_folder", "")
            for item in enabled_items:
                name = clean_filename(item.name).replace(".", "")
                try:
                    pipe.load_lora_weights(
                        bpy.path.abspath(lora_folder),
                        weight_name=item.name + ".safetensors",
                        adapter_name=name,
                    )
                    names.append(name)
                    weights.append(item.weight_value)
                except Exception as e:
                    print(f"LoRA '{item.name}': load error — {e}")
        pipe.set_adapters(names, adapter_weights=weights)

        if gfx_device == "mps":
            pipe.to("mps")
        elif low_vram():
            pipe.enable_model_cpu_offload()
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()
        else:
            pipe.enable_model_cpu_offload()
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()

        processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf", cache_dir=_cache_dir)
        return {"pipe": pipe, "converter": None, "refiner": None, "preprocessor": processor}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch

        pipe      = pipe_obj["pipe"]
        processor = pipe_obj["preprocessor"]
        image     = inputs.image
        if image is None:
            raise ValueError("FLUX Depth requires an input image.")

        seed = inputs.seed
        generator = (
            torch.Generator("cuda").manual_seed(seed)
            if torch.cuda.is_available() and seed != 0 else None
        )
        control_image = processor(image)[0].convert("RGB")
        self.set_phase(inputs, "Generating")
        return pipe(
            prompt=inputs.prompt,
            control_image=control_image,
            controlnet_conditioning_scale=1.0 - inputs.strength,
            num_inference_steps=inputs.steps,
            guidance_scale=inputs.guidance,
            height=inputs.height,
            width=inputs.width,
            generator=generator,
            callback_on_step_end=self.step_callback(inputs),
        ).images[0]
