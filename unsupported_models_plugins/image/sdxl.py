"""Text-to-image, img2img, and inpaint via Stable Diffusion XL (IP Adapter, LCM, LoRA, refiner)."""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import (
    gfx_device, low_vram, NoWatermark, load_images_from_folder,
)


class SDXLPlugin(ModelPlugin):
    MODEL_ID     = "stabilityai/stable-diffusion-xl-base-1.0"
    DISPLAY_NAME = "Image: Stable Diffusion XL"
    MODEL_TYPE   = "image"
    DESCRIPTION  = "SDXL with IP Adapter, LCM, LoRA, img2img, inpaint, and refiner"

    INPUTS       = InputSpec.PROMPT | InputSpec.NEG_PROMPT | InputSpec.IMAGE | InputSpec.LORA
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.NEG_PROMPT, UISection.IMAGE_STRIP,
        UISection.RESOLUTION, UISection.STEPS, UISection.GUIDANCE,
        UISection.IMAGE_STRENGTH, UISection.SEED,
        UISection.LORA, UISection.IP_ADAPTER, UISection.ENHANCE,
    ]
    PARAMS       = ParamSpec(steps=20, guidance=7.5)
    REQUIRED_PACKAGES = ["torch", "diffusers", "transformers"]

    def load(self, prefs, scene, **kw):
        import torch
        from diffusers import DiffusionPipeline, AutoencoderKL

        mode        = kw.get("mode", "txt2img")
        use_lcm     = kw.get("use_lcm", False)
        use_refine  = kw.get("use_refine", False)
        enabled_items = kw.get("enabled_items", [])
        ip_face     = kw.get("ip_adapter_face_folder", "")
        ip_style    = kw.get("ip_adapter_style_folder", "")
        local_only  = kw.get("local_files_only", False)

        print(f"Loading {self.MODEL_ID} ({mode})…")
        pipe = converter = refiner = None

        # --- Inpaint ---
        if mode == "inpaint":
            from diffusers import AutoPipelineForInpainting

            pipe = AutoPipelineForInpainting.from_pretrained(
                "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                torch_dtype=torch.float16, variant="fp16",
                local_files_only=local_only,
            )
            if use_lcm:
                from diffusers import LCMScheduler
                pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
                if enabled_items:
                    pipe.load_lora_weights(
                        "latent-consistency/lcm-lora-sdxl",
                        weight_name="pytorch_lora_weights.safetensors",
                        adapter_name="lcm-lora-sdxl",
                    )
                else:
                    pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
            else:
                from diffusers import DPMSolverMultistepScheduler
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.watermark = NoWatermark()
            if gfx_device == "mps":
                pipe.to("mps")
            elif low_vram():
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(gfx_device)

        # --- Img2img converter ---
        elif mode == "img2img":
            if ip_face or ip_style:
                from diffusers import AutoPipelineForImage2Image
                from transformers import CLIPVisionModelWithProjection

                image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                    "h94/IP-Adapter", subfolder="models/image_encoder",
                    torch_dtype=torch.float16, local_files_only=local_only,
                )
                converter = AutoPipelineForImage2Image.from_pretrained(
                    self.MODEL_ID, torch_dtype=torch.float16,
                    image_encoder=image_encoder, local_files_only=local_only,
                )
                self._attach_ip_adapter(converter, ip_face, ip_style, local_only)
            else:
                from diffusers import StableDiffusionXLImg2ImgPipeline

                vae = AutoencoderKL.from_pretrained(
                    "madebyollin/sdxl-vae-fp16-fix",
                    torch_dtype=torch.float16, local_files_only=local_only,
                )
                converter = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    "thingthatis/stable-diffusion-xl-refiner-1.0",
                    vae=vae, torch_dtype=torch.float16, variant="fp16",
                    local_files_only=local_only,
                )
            converter.watermark = NoWatermark()
            if gfx_device == "mps":
                converter.to("mps")
            elif low_vram():
                converter.enable_model_cpu_offload()
            else:
                converter.to(gfx_device)

        # --- Txt2img ---
        else:
            if ip_face or ip_style:
                from diffusers import AutoPipelineForText2Image
                from transformers import CLIPVisionModelWithProjection

                image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                    "h94/IP-Adapter", subfolder="models/image_encoder",
                    torch_dtype=torch.float16, local_files_only=local_only,
                )
                pipe = AutoPipelineForText2Image.from_pretrained(
                    self.MODEL_ID, torch_dtype=torch.float16,
                    image_encoder=image_encoder, local_files_only=local_only,
                )
                self._attach_ip_adapter(pipe, ip_face, ip_style, local_only)
            else:
                vae = AutoencoderKL.from_pretrained(
                    "madebyollin/sdxl-vae-fp16-fix",
                    torch_dtype=torch.float16, local_files_only=local_only,
                )
                pipe = DiffusionPipeline.from_pretrained(
                    self.MODEL_ID, vae=vae,
                    torch_dtype=torch.float16, variant="fp16",
                    local_files_only=local_only,
                )
            # LCM
            if use_lcm:
                from diffusers import LCMScheduler
                if enabled_items:
                    pipe.load_lora_weights(
                        "latent-consistency/lcm-lora-sdxl",
                        weight_name="pytorch_lora_weights.safetensors",
                        adapter_name="lcm-lora-sdxl",
                    )
                else:
                    pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
                pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            # Segmind-VegaRT style guidance tweak is handled by segmind_vega.py
            if gfx_device == "mps":
                pipe.to("mps")
            elif low_vram():
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(gfx_device)

        # LoRA (non-LCM, for txt2img/inpaint)
        if enabled_items and not use_lcm and pipe is not None:
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

        # Refiner
        if use_refine:
            from diffusers import StableDiffusionXLImg2ImgPipeline

            vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix",
                torch_dtype=torch.float16, local_files_only=local_only,
            )
            refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                "thingthatis/stable-diffusion-xl-refiner-1.0",
                vae=vae, torch_dtype=torch.float16, variant="fp16",
                local_files_only=local_only,
            )
            refiner.watermark = NoWatermark()
            if gfx_device == "mps":
                refiner.to("mps")
            elif low_vram():
                refiner.enable_model_cpu_offload()
            else:
                refiner.to(gfx_device)

        return {"pipe": pipe, "converter": converter, "refiner": refiner, "preprocessor": None}

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _attach_ip_adapter(pipeline, ip_face, ip_style, local_only):
        from diffusers import DDIMScheduler

        if ip_face and ip_style:
            pipeline.load_ip_adapter(
                "h94/IP-Adapter", subfolder="sdxl_models",
                weight_name=[
                    "ip-adapter-plus_sdxl_vit-h.safetensors",
                    "ip-adapter-plus-face_sdxl_vit-h.safetensors",
                ],
                local_files_only=local_only,
            )
            pipeline.set_ip_adapter_scale([0.7, 0.5])
        elif ip_face:
            pipeline.load_ip_adapter(
                "h94/IP-Adapter", subfolder="sdxl_models",
                weight_name=["ip-adapter-plus-face_sdxl_vit-h.safetensors"],
                local_files_only=local_only,
            )
            pipeline.set_ip_adapter_scale([0.8])
        else:
            pipeline.load_ip_adapter(
                "h94/IP-Adapter", subfolder="sdxl_models",
                weight_name=["ip-adapter-plus_sdxl_vit-h.safetensors"],
                local_files_only=local_only,
            )
            pipeline.set_ip_adapter_scale([1.0])
            pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    @staticmethod
    def _build_ip_images(ip_face, ip_style):
        if ip_face and ip_style:
            return [load_images_from_folder(ip_style), load_images_from_folder(ip_face)]
        if ip_face:
            return [load_images_from_folder(ip_face)]
        return [load_images_from_folder(ip_style)]

    # ------------------------------------------------------------------ generate

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch

        pipe      = pipe_obj["pipe"]
        converter = pipe_obj["converter"]
        seed      = inputs.seed
        generator = (
            torch.Generator("cuda").manual_seed(seed)
            if torch.cuda.is_available() and seed != 0 else None
        )
        ip_face  = getattr(scene, "ip_adapter_face_folder", "")
        ip_style = getattr(scene, "ip_adapter_style_folder", "")

        # --- Inpaint ---
        if inputs.mode == "inpaint" and inputs.image is not None and inputs.inpaint_mask is not None:
            if ip_face or ip_style:
                ip_img = self._build_ip_images(ip_face, ip_style)
                return pipe(
                    inputs.prompt,
                    negative_prompt=inputs.neg_prompt,
                    image=inputs.image,
                    mask_image=inputs.inpaint_mask,
                    ip_adapter_image=ip_img,
                    num_inference_steps=inputs.steps,
                    guidance_scale=inputs.guidance,
                    height=inputs.height,
                    width=inputs.width,
                    generator=generator,
                ).images[0]
            return pipe(
                prompt=inputs.prompt,
                negative_prompt=inputs.neg_prompt,
                image=inputs.image,
                mask_image=inputs.inpaint_mask,
                num_inference_steps=inputs.steps,
                guidance_scale=inputs.guidance,
                height=inputs.height,
                width=inputs.width,
                padding_mask_crop=42,
                strength=0.99,
                generator=generator,
            ).images[0]

        # --- Img2img ---
        if inputs.mode == "img2img" and inputs.image is not None and converter is not None:
            if ip_face or ip_style:
                ip_img = self._build_ip_images(ip_face, ip_style)
                return converter(
                    inputs.prompt,
                    image=inputs.image,
                    negative_prompt=inputs.neg_prompt,
                    ip_adapter_image=ip_img,
                    num_inference_steps=inputs.steps,
                    guidance_scale=inputs.guidance,
                    height=inputs.height,
                    width=inputs.width,
                    generator=generator,
                ).images[0]
            return converter(
                prompt=inputs.prompt,
                image=inputs.image,
                strength=1.0 - inputs.strength,
                negative_prompt=inputs.neg_prompt,
                num_inference_steps=inputs.steps,
                guidance_scale=inputs.guidance,
                generator=generator,
            ).images[0]

        # --- Txt2img ---
        if ip_face or ip_style:
            ip_img = self._build_ip_images(ip_face, ip_style)
            return pipe(
                inputs.prompt,
                negative_prompt=inputs.neg_prompt,
                ip_adapter_image=ip_img,
                num_inference_steps=inputs.steps,
                guidance_scale=inputs.guidance,
                height=inputs.height,
                width=inputs.width,
                generator=generator,
            ).images[0]

        return pipe(
            inputs.prompt,
            negative_prompt=inputs.neg_prompt,
            num_inference_steps=inputs.steps,
            guidance_scale=inputs.guidance,
            height=inputs.height,
            width=inputs.width,
            cross_attention_kwargs={"scale": 1.0},
            generator=generator,
        ).images[0]
