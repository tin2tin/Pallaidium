"""Image-to-video via Wan2.2-I2V-A14B (dual NF4 transformers + Lightx2v LoRA)."""

import shutil
from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram, solve_path, clean_filename, load_first_frame


def _resize_for_wan(image, max_dim: int = 832):
    from PIL import Image
    w, h = image.size
    scale = max_dim / max(w, h)
    new_w = (int(w * scale) // 16) * 16
    new_h = (int(h * scale) // 16) * 16
    return image.resize((new_w, new_h), Image.LANCZOS)


class WanI2VPlugin(ModelPlugin):
    MODEL_ID     = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    DISPLAY_NAME = "Video: Wan2.2 I2V"
    MODEL_TYPE   = "video"
    DESCRIPTION  = "Image-to-video via Wan2.2-I2V-A14B (dual 4-bit transformers, Lightx2v LoRA)"

    INPUTS       = InputSpec.PROMPT | InputSpec.NEG_PROMPT | InputSpec.IMAGE | InputSpec.LORA
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.NEG_PROMPT, UISection.VIDEO_STRIP,
        UISection.RESOLUTION, UISection.FRAMES, UISection.SEED, UISection.LORA,
    ]
    PARAMS       = ParamSpec(width=832, height=480, frames=81, steps=8, guidance=1.0)
    REQUIRED_PACKAGES = ["torch", "diffusers", "transformers"]

    def load(self, prefs, scene, **kw):
        import torch
        from diffusers import WanImageToVideoPipeline, WanTransformer3DModel, FlowMatchEulerDiscreteScheduler
        from transformers import BitsAndBytesConfig

        _cache_dir = prefs.hf_cache_dir or None
        enabled_items = kw.get("enabled_items", [])
        print(f"Loading {self.MODEL_ID}…")

        nf4 = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        )
        _lfo = prefs.local_files_only
        transformer_high = WanTransformer3DModel.from_pretrained(
            self.MODEL_ID, subfolder="transformer",
            quantization_config=nf4, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            cache_dir=_cache_dir, local_files_only=_lfo,
        )
        transformer_low = WanTransformer3DModel.from_pretrained(
            self.MODEL_ID, subfolder="transformer_2",
            quantization_config=nf4, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            cache_dir=_cache_dir, local_files_only=_lfo,
        )
        pipe = WanImageToVideoPipeline.from_pretrained(
            self.MODEL_ID,
            transformer=transformer_high, transformer_2=transformer_low,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            cache_dir=_cache_dir, local_files_only=_lfo,
        )

        if gfx_device == "mps":
            pipe.to("mps")
        else:
            pipe.enable_model_cpu_offload()

        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            pipe.scheduler.config, shift=5.0,
        )

        lora_file = "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors"
        try:
            pipe.load_lora_weights("Kijai/WanVideo_comfy", weight_name=lora_file, adapter_name="lightx2v")
            pipe.load_lora_weights(
                "Kijai/WanVideo_comfy", weight_name=lora_file,
                adapter_name="lightx2v_2", load_into_transformer_2=True,
            )
            pipe.set_adapters(["lightx2v", "lightx2v_2"], adapter_weights=[1.0, 1.0])
        except Exception as e:
            print(f"Lightx2v I2V LoRA load failed (continuing): {e}")

        if enabled_items:
            from ...utils.helpers import clean_filename as _cf, bpy
            lora_folder = getattr(bpy.context.scene, "lora_folder", "")
            names, weights = [], []
            for item in enabled_items:
                name = _cf(item.name).replace(".", "")
                names.append(name)
                weights.append(item.weight_value)
                pipe.load_lora_weights(
                    bpy.path.abspath(lora_folder),
                    weight_name=item.name + ".safetensors",
                    adapter_name=name,
                )
            pipe.set_adapters(names, adapter_weights=weights)

        return {"pipe": pipe, "refiner": None, "last_model_card": self.MODEL_ID}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch
        import gc
        from diffusers.utils import export_to_video

        pipe = pipe_obj["pipe"]
        seed = inputs.seed
        generator = (
            torch.Generator("cuda").manual_seed(seed)
            if torch.cuda.is_available() and seed != 0 else None
        )

        image = inputs.image
        if image is None and inputs.video_path:
            image = load_first_frame(inputs.video_path)
        if image is None:
            raise RuntimeError("Wan2.2-I2V requires an image or video input.")

        image = _resize_for_wan(image)
        gc.collect()
        torch.cuda.empty_cache()

        self.set_phase(inputs, "Generating")
        video_frames = pipe(
            image=image,
            prompt=inputs.prompt,
            negative_prompt=inputs.neg_prompt,
            num_frames=inputs.frames,
            num_inference_steps=8,
            guidance_scale=1.0,
            guidance_scale_2=1.0,
            height=inputs.height,
            width=inputs.width,
            generator=generator,
            max_sequence_length=512,
            callback_on_step_end=self.step_callback(inputs),
        ).frames[0]

        self.set_phase(inputs, "Saving")
        src_path = export_to_video(video_frames, fps=16)
        dst_path = solve_path(clean_filename(str(seed) + "_" + inputs.prompt) + ".mp4")
        shutil.move(src_path, dst_path)
        return dst_path
