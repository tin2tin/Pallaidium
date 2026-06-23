"""Text-to-image via Krea 2 (base, CFG-guided)."""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram


class Krea2BasePlugin(ModelPlugin):
    MODEL_ID     = "ethanfel/Krea-2-Base-Diffusers"
    DISPLAY_NAME = "Image: Krea 2"
    DESCRIPTION  = "High-quality text-to-image via Krea 2 (12.9B MMDiT, CFG-guided)"
    MODEL_TYPE   = "image"
    INPUTS       = InputSpec.PROMPT | InputSpec.NEG_PROMPT | InputSpec.LORA
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.NEG_PROMPT,
        UISection.RESOLUTION, UISection.FRAMES, UISection.STEPS, UISection.GUIDANCE, UISection.SEED,
        UISection.LORA,
    ]
    PARAMS            = ParamSpec(steps=28, guidance=4.5)
    REQUIRED_PACKAGES = ["torch", "diffusers", "transformers"]
    supports_inpaint  = False
    supports_img2img  = False

    def load(self, prefs, scene, **kw):
        import torch
        from diffusers import Krea2Pipeline, Krea2Transformer2DModel
        from diffusers import BitsAndBytesConfig as DBnB
        from transformers import BitsAndBytesConfig as TBnB, Qwen3VLModel

        _cache_dir = prefs.hf_cache_dir or None
        _lfo  = prefs.local_files_only
        dtype = torch.bfloat16
        mode  = kw.get("mode", "txt2img")
        print(f"Loading {self.MODEL_ID} ({mode})…")

        # No pre-quantized repo exists for Krea 2 yet, and the full bf16 checkpoint
        # is ~36 GB (12.9B transformer + Qwen3-VL-4B text encoder). Quantize the two
        # heavy components to 4-bit nf4 on load so it fits consumer VRAM, mirroring
        # the Qwen-Image img2img path. The VAE/scheduler stay in bf16.
        q_t = DBnB(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
        q_e = TBnB(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)

        # On-the-fly bnb quant: use a chained .to("cpu"), NOT device_map="cpu".
        # With device_map="cpu", accelerate quantizes each weight on the CPU via
        # bitsandbytes' pure-PyTorch default backend, whose argmin step allocates
        # multi-GB temporaries per shard (OOM/crash, ~132 s/it). .to("cpu") leaves
        # the params unquantized on CPU and defers quantization to the fast native
        # CUDA kernel when enable_model_cpu_offload moves each module to the GPU.
        # (Klein uses device_map="cpu" safely only because its repos are PRE-
        # quantized — no kernel runs there. This mirrors the Qwen-Image img2img path.)
        transformer = Krea2Transformer2DModel.from_pretrained(
            self.MODEL_ID, subfolder="transformer", quantization_config=q_t,
            torch_dtype=dtype, cache_dir=_cache_dir, local_files_only=_lfo,
        ).to("cpu")
        text_encoder = Qwen3VLModel.from_pretrained(
            self.MODEL_ID, subfolder="text_encoder", quantization_config=q_e,
            torch_dtype=dtype, cache_dir=_cache_dir, local_files_only=_lfo,
        ).to("cpu")
        pipe = Krea2Pipeline.from_pretrained(
            self.MODEL_ID, transformer=transformer, text_encoder=text_encoder,
            torch_dtype=dtype, cache_dir=_cache_dir, local_files_only=_lfo,
        )

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
        return pipe(
            prompt=inputs.prompt,
            negative_prompt=inputs.neg_prompt,
            num_inference_steps=inputs.steps,
            guidance_scale=inputs.guidance,
            height=inputs.height,
            width=inputs.width,
            max_sequence_length=512,
            generator=generator,
            callback_on_step_end=self.step_callback(inputs),
        ).images[0]
