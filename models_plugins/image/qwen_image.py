"""Text-to-image and img2img via Qwen-Image-2512."""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram


class QwenImagePlugin(ModelPlugin):
    MODEL_ID     = "Qwen/Qwen-Image-2512"
    DISPLAY_NAME = "Image: Qwen Image 2512"
    MODEL_TYPE   = "image"
    DESCRIPTION  = "High-quality text-to-image and img2img via Qwen-Image-2512"

    INPUTS       = InputSpec.PROMPT | InputSpec.IMAGE | InputSpec.LORA
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.IMAGE_STRIP,
        UISection.RESOLUTION, UISection.FRAMES, UISection.STEPS, UISection.IMAGE_STRENGTH, UISection.SEED,
        UISection.LORA,
    ]
    PARAMS            = ParamSpec(steps=4, guidance=1.0)
    REQUIRED_PACKAGES = ["torch", "diffusers", "transformers"]
    supports_inpaint  = False

    def load(self, prefs, scene, **kw):
        import torch
        from transformers import BitsAndBytesConfig as TBnB, Qwen2_5_VLForConditionalGeneration
        from diffusers import BitsAndBytesConfig as DBnB

        _cache_dir = prefs.hf_cache_dir or None
        model_id = self.MODEL_ID
        dtype = torch.bfloat16
        mode = kw.get("mode", "txt2img")
        print(f"Loading {model_id} ({mode})…")

        _lfo = prefs.local_files_only
        if mode == "img2img":
            from diffusers import QwenImageImg2ImgPipeline, QwenImageTransformer2DModel
            q_t = DBnB(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
                       llm_int8_skip_modules=["transformer_blocks.0.img_mod"])
            q_e = TBnB(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
            transformer = QwenImageTransformer2DModel.from_pretrained(
                model_id, subfolder="transformer", quantization_config=q_t, torch_dtype=dtype,
                cache_dir=_cache_dir, local_files_only=_lfo,
            ).to("cpu")
            text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, subfolder="text_encoder", quantization_config=q_e, torch_dtype=dtype,
                cache_dir=_cache_dir, local_files_only=_lfo,
            ).to("cpu")
            pipe = QwenImageImg2ImgPipeline.from_pretrained(
                model_id, transformer=transformer, text_encoder=text_encoder, torch_dtype=dtype,
                cache_dir=_cache_dir, local_files_only=_lfo,
            )
        else:
            from diffusers import QwenImagePipeline, QwenImageTransformer2DModel
            transformer = QwenImageTransformer2DModel.from_pretrained(
                "OzzyGT/Qwen-Image-2512-bnb-4bit-transformer", torch_dtype=dtype, device_map="cpu",
                cache_dir=_cache_dir, local_files_only=_lfo,
            )
            text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "OzzyGT/Qwen-Image-2512-bnb-4bit-text-encoder", torch_dtype=dtype, device_map="cpu",
                cache_dir=_cache_dir, local_files_only=_lfo,
            )
            pipe = QwenImagePipeline.from_pretrained(
                model_id, transformer=transformer, text_encoder=text_encoder, torch_dtype=dtype,
                cache_dir=_cache_dir, local_files_only=_lfo,
            )

        # Lightning distillation LoRA → 4-step inference. Version-matched to
        # Qwen-Image-2512 and stored with diffusers-native keys (transformer.
        # prefix), so it attaches cleanly — unlike the Wuli turbo LoRA, whose
        # ComfyUI-style keys matched zero transformer params and left output
        # undercooked. Name it as an adapter and include it in set_adapters below
        # so it STAYS active when the user also enables their own LoRAs (a bare
        # load with no adapter_name gets dropped the moment set_adapters()
        # activates only the user names). Runs CFG-free (guidance_scale=1.0).
        _lora_names, _lora_weights = [], []
        try:
            pipe.load_lora_weights(
                "lightx2v/Qwen-Image-2512-Lightning",
                weight_name="Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors",
                adapter_name="lightning",
                cache_dir=_cache_dir,
                local_files_only=_lfo,
            )
            _lora_names.append("lightning")
            _lora_weights.append(1.0)
        except Exception as e:
            print(f"Qwen Image 2512: Lightning LoRA unavailable ({e}); "
                  "raise steps for usable results.")

        # Apply user LoRAs on top of Turbo
        enabled_items = kw.get("enabled_items", [])
        if enabled_items:
            from ...utils.helpers import clean_filename, bpy
            lora_folder = getattr(bpy.context.scene, "lora_folder", "")
            for item in enabled_items:
                name = clean_filename(item.name).replace(".", "")
                pipe.load_lora_weights(
                    bpy.path.abspath(lora_folder),
                    weight_name=item.name + ".safetensors",
                    adapter_name=name,
                )
                _lora_names.append(name)
                _lora_weights.append(item.weight_value)

        # Only activate adapters that actually registered. Loading a LoRA onto
        # the bnb-4bit transformer can no-op on the PEFT adapter registry without
        # raising (key/prefix mismatch), which would make set_adapters() fail on
        # a name that isn't present. Intersect our intended names with what the
        # pipe reports so this never crashes and still stacks whatever did load.
        if _lora_names:
            _present = set()
            try:
                for _v in pipe.get_list_adapters().values():
                    _present.update(_v)
            except Exception:
                pass
            _active = [(n, w) for n, w in zip(_lora_names, _lora_weights) if n in _present]
            if _active:
                pipe.set_adapters([n for n, _ in _active],
                                  adapter_weights=[w for _, w in _active])

        if gfx_device == "mps":
            pipe.to("mps")
        elif low_vram():
            pipe.enable_model_cpu_offload()
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()
        else:
            pipe.enable_model_cpu_offload()
        return {"pipe": pipe, "converter": None, "refiner": None, "preprocessor": None}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch

        pipe = pipe_obj["pipe"]
        seed = inputs.seed
        generator = (
            torch.Generator("cuda").manual_seed(seed)
            if torch.cuda.is_available() and seed != 0 else None
        )
        self.set_phase(inputs, "Generating")
        if inputs.mode == "img2img" and inputs.image is not None:
            return pipe(
                prompt=inputs.prompt,
                image=inputs.image,
                strength=1.0 - inputs.strength,
                num_inference_steps=inputs.steps,
                guidance_scale=inputs.guidance,
                generator=generator,
                callback_on_step_end=self.step_callback(inputs),
            ).images[0]
        else:
            return pipe(
                prompt=inputs.prompt,
                num_inference_steps=inputs.steps,
                guidance_scale=inputs.guidance,
                height=inputs.height,
                width=inputs.width,
                generator=generator,
                callback_on_step_end=self.step_callback(inputs),
            ).images[0]
