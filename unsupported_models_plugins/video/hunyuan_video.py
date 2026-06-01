"""Text-to-video and img2vid via HunyuanVideo (GGUF-quantized transformer)."""

import shutil
from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram, solve_path, clean_filename, load_first_frame


class HunyuanVideoPlugin(ModelPlugin):
    MODEL_ID     = "hunyuanvideo-community/HunyuanVideo"
    DISPLAY_NAME = "Video: HunyuanVideo"
    MODEL_TYPE   = "video"
    DESCRIPTION  = "High-quality text-to-video and img2vid via HunyuanVideo (GGUF quantized)"

    INPUTS       = InputSpec.PROMPT | InputSpec.IMAGE | InputSpec.LORA
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.VIDEO_STRIP,
        UISection.RESOLUTION, UISection.FRAMES, UISection.STEPS, UISection.GUIDANCE, UISection.SEED,
        UISection.LORA,
    ]
    PARAMS       = ParamSpec(width=848, height=480, frames=61, steps=50, guidance=6.0)
    REQUIRED_PACKAGES = ["torch", "diffusers", "transformers"]

    def load(self, prefs, scene, **kw):
        import torch
        from diffusers.models import HunyuanVideoTransformer3DModel
        from diffusers import GGUFQuantizationConfig, BitsAndBytesConfig
        from transformers import LlamaModel, CLIPTextModel

        mode = kw.get("mode", "txt2vid")
        enabled_items = kw.get("enabled_items", [])
        print(f"Loading {self.MODEL_ID} ({mode})…")

        img2vid = mode in ("img2vid", "vid2vid")  # vid2vid falls back to I2V

        if img2vid:
            model_id = "hunyuanvideo-community/HunyuanVideo-I2V"
            if low_vram():
                transformer_path = "https://huggingface.co/city96/HunyuanVideo-I2V-gguf/blob/main/hunyuan-video-i2v-720p-Q4_K_S.gguf"
            else:
                transformer_path = "https://huggingface.co/city96/HunyuanVideo-I2V-gguf/blob/main/hunyuan-video-i2v-720p-Q4_K_S.gguf"
            from diffusers import HunyuanVideoImageToVideoPipeline
        else:
            model_id = "hunyuanvideo-community/HunyuanVideo"
            if low_vram():
                transformer_path = "https://huggingface.co/city96/HunyuanVideo-gguf/blob/main/hunyuan-video-t2v-720p-Q3_K_S.gguf"
            else:
                transformer_path = "https://huggingface.co/city96/HunyuanVideo-gguf/blob/main/hunyuan-video-t2v-720p-Q4_K_S.gguf"
            from diffusers import HunyuanVideoPipeline

        quant_nf4 = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
        transformer = HunyuanVideoTransformer3DModel.from_single_file(
            transformer_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
        )

        if img2vid:
            pipe = HunyuanVideoImageToVideoPipeline.from_pretrained(
                model_id, transformer=transformer, torch_dtype=torch.float16,
            )
        else:
            text_encoder = LlamaModel.from_pretrained(
                model_id, subfolder="text_encoder",
                quantization_config=quant_nf4, torch_dtype=torch.float16,
            )
            text_encoder_2 = CLIPTextModel.from_pretrained(
                model_id, subfolder="text_encoder_2",
                quantization_config=quant_nf4, torch_dtype=torch.float16,
            )
            pipe = HunyuanVideoPipeline.from_pretrained(
                model_id,
                text_encoder=text_encoder, text_encoder_2=text_encoder_2,
                transformer=transformer, torch_dtype=torch.float16,
            )

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

        if gfx_device == "mps":
            pipe.to("mps")
        elif low_vram():
            pipe.vae.enable_tiling()
            pipe.enable_model_cpu_offload()
        else:
            pipe.enable_model_cpu_offload()

        return {"pipe": pipe, "refiner": None, "last_model_card": self.MODEL_ID}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch
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

        common = dict(
            prompt=inputs.prompt,
            num_inference_steps=inputs.steps,
            guidance_scale=inputs.guidance,
            height=inputs.height,
            width=inputs.width,
            num_frames=inputs.frames,
            generator=generator,
        )

        if image is not None:
            video_frames = pipe(image=image, **common).frames[0]
        else:
            video_frames = pipe(num_videos_per_prompt=1, **common).frames[0]

        import bpy
        render = bpy.context.scene.render
        fps = round(render.fps / render.fps_base, 3)
        src_path = export_to_video(video_frames, fps=fps)
        dst_path = solve_path(clean_filename(str(seed) + "_" + inputs.prompt) + ".mp4")
        shutil.move(src_path, dst_path)
        return dst_path
