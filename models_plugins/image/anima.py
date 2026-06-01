"""Text-to-image and image-to-image via Anima (mrfatso/anima-preview3-diffusers)."""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram, clean_filename


class AnimaPlugin(ModelPlugin):
    MODEL_ID     = "mrfatso/anima-preview3-diffusers"
    DISPLAY_NAME = "Image: Anima"
    MODEL_TYPE   = "image"
    DESCRIPTION  = "Anime-style generation via Anima with txt2img, img2img, and LoRA support"

    INPUTS       = InputSpec.PROMPT | InputSpec.NEG_PROMPT | InputSpec.IMAGE | InputSpec.LORA
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.NEG_PROMPT, UISection.IMAGE_STRIP,
        UISection.RESOLUTION, UISection.FRAMES, UISection.STEPS, UISection.GUIDANCE,
        UISection.IMAGE_STRENGTH, UISection.SEED, UISection.LORA,
    ]
    PARAMS            = ParamSpec(steps=25, guidance=4.0)
    REQUIRED_PACKAGES = ["torch", "diffusers"]
    supports_inpaint  = False
    supports_img2img  = True

    def load(self, prefs, scene, **kw):
        import torch
        from diffusers import AnimaAutoBlocks
        from diffusers.guiders import ClassifierFreeGuidance

        _cache_dir = prefs.hf_cache_dir or None
        print(f"Loading {self.MODEL_ID}…")

        pipe = AnimaAutoBlocks().init_pipeline(self.MODEL_ID)
        pipe.load_components(torch_dtype=torch.bfloat16, cache_dir=_cache_dir)
        pipe.update_components(
            guider=ClassifierFreeGuidance(guidance_scale=4.0),
        )

        # Apply user LoRAs before moving to device
        enabled_items = kw.get("enabled_items", [])
        if enabled_items:
            import bpy
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

        pipe.to("mps" if gfx_device == "mps" else gfx_device)

        return {"pipe": pipe, "converter": pipe, "refiner": None, "preprocessor": None}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch

        pipe  = pipe_obj["pipe"]
        seed  = inputs.seed
        generator = (
            torch.Generator("cuda").manual_seed(seed)
            if torch.cuda.is_available() and seed != 0 else None
        )

        self.set_phase(inputs, "Generating")

        shared_kwargs = dict(
            prompt=inputs.prompt,
            negative_prompt=inputs.neg_prompt or "",
            width=inputs.width,
            height=inputs.height,
            num_inference_steps=inputs.steps,
            guidance_scale=inputs.guidance,
            generator=generator,
            callback_on_step_end=self.step_callback(inputs),
        )

        if inputs.image is not None:
            # img2img: pass the init image and denoise strength
            return pipe(
                image=inputs.image,
                strength=1.0 - inputs.strength,
                **shared_kwargs,
            ).images[0]

        return pipe(**shared_kwargs).images[0]
