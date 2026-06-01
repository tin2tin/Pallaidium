"""Img2vid via FramePack I2V (HunyuanVideo backbone + Siglip image encoder)."""

import shutil
from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram, solve_path, clean_filename, load_first_frame, find_strip_by_name


class FramePackPlugin(ModelPlugin):
    MODEL_ID     = "lllyasviel/FramePackI2V_HY"
    DISPLAY_NAME = "Video: FramePack I2V"
    MODEL_TYPE   = "video"
    DESCRIPTION  = "Image-to-video via FramePack (HunyuanVideo backbone, Siglip encoder)"

    INPUTS       = InputSpec.PROMPT | InputSpec.IMAGE
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.VIDEO_STRIP,
        UISection.RESOLUTION, UISection.FRAMES, UISection.STEPS, UISection.GUIDANCE, UISection.SEED,
    ]
    PARAMS                    = ParamSpec(width=848, height=480, frames=61, steps=25, guidance=1.0)
    REQUIRED_PACKAGES         = ["torch", "diffusers", "transformers"]
    uses_standard_input_strip = False

    def draw_custom_ui(self, col, context) -> bool:
        scene = context.scene
        if getattr(scene, "input_strips", "") != "input_strips" or scene.sequence_editor is None:
            return False
        row = col.row(align=True)
        row.prop_search(
            scene, "out_frame", scene.sequence_editor, "strips",
            text="End Frame", icon="RENDER_RESULT",
        )
        row.operator("sequencer.strip_picker", text="", icon="EYEDROPPER").action = "out_frame_select"
        return False

    def load(self, prefs, scene, **kw):
        import torch
        from diffusers import (
            BitsAndBytesConfig, HunyuanVideoFramepackPipeline,
            HunyuanVideoFramepackTransformer3DModel,
        )
        from transformers import SiglipImageProcessor, SiglipVisionModel

        mode = kw.get("mode", "img2vid")
        if mode not in ("img2vid", "vid2vid"):
            print("FramePack: txt2vid is not supported — requires an image input.")
            return {"pipe": None, "refiner": None, "last_model_card": self.MODEL_ID}

        print(f"Loading {self.MODEL_ID}…")

        nf4 = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        )
        transformer = HunyuanVideoFramepackTransformer3DModel.from_pretrained(
            "lllyasviel/FramePack_F1_I2V_HY_20250503",
            quantization_config=nf4, torch_dtype=torch.bfloat16,
        )
        feature_extractor = SiglipImageProcessor.from_pretrained(
            "lllyasviel/flux_redux_bfl", subfolder="feature_extractor",
        )
        image_encoder = SiglipVisionModel.from_pretrained(
            "lllyasviel/flux_redux_bfl", subfolder="image_encoder", torch_dtype=torch.float16,
        )
        pipe = HunyuanVideoFramepackPipeline.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo",
            transformer=transformer,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            torch_dtype=torch.float16,
        )

        if gfx_device == "mps":
            pipe.to("mps")
        else:
            pipe.vae.enable_tiling()
            pipe.enable_model_cpu_offload()

        return {"pipe": pipe, "refiner": None, "last_model_card": self.MODEL_ID}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch
        from diffusers.utils import export_to_video

        pipe = pipe_obj["pipe"]
        if pipe is None:
            raise RuntimeError("FramePack requires an image input (img2vid mode).")

        seed = inputs.seed
        generator = (
            torch.Generator("cuda").manual_seed(seed)
            if torch.cuda.is_available() and seed != 0 else None
        )

        image = inputs.image
        if image is None and inputs.video_path:
            image = load_first_frame(inputs.video_path)
        if image is None:
            raise RuntimeError("FramePack requires an image input.")

        # Optional end-frame strip
        last_image = None
        out_frame = getattr(scene, "out_frame", "")
        if out_frame:
            import bpy, os
            strip = find_strip_by_name(scene, out_frame)
            if strip and strip.type == "IMAGE":
                img_path = bpy.path.abspath(
                    os.path.join(strip.directory, strip.elements[0].filename)
                )
                if os.path.isfile(img_path):
                    from diffusers.utils import load_image
                    last_image = load_image(img_path).resize(image.size)

        video_frames = pipe(
            image=image,
            last_image=last_image,
            prompt=inputs.prompt,
            num_inference_steps=inputs.steps,
            guidance_scale=inputs.guidance,
            height=inputs.height,
            width=inputs.width,
            num_frames=inputs.frames,
            generator=generator,
            sampling_type="vanilla",
        ).frames[0]

        import bpy
        render = bpy.context.scene.render
        fps = round(render.fps / render.fps_base, 3)
        src_path = export_to_video(video_frames, fps=fps)
        dst_path = solve_path(clean_filename(str(seed) + "_" + inputs.prompt) + ".mp4")
        shutil.move(src_path, dst_path)
        return dst_path
