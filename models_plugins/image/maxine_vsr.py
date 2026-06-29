"""NVIDIA Maxine Video Super Resolution via nvvfx."""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs


def _pil_to_chw(image):
    import torch
    import numpy as np

    arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    return (
        torch.from_numpy(arr)
        .permute(2, 0, 1)
        .contiguous()
        .to(device="cuda", dtype=torch.float32)
    )


def _chw_to_pil(tensor):
    import numpy as np
    from PIL import Image

    arr = (
        tensor.clamp(0, 1)
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
    )
    return Image.fromarray((arr * 255).astype(np.uint8), "RGB")


def _get_quality_level(scene):
    from nvvfx.effects.video_super_res import QualityLevel
    name = getattr(scene, "maxine_quality", "HIGH")
    return getattr(QualityLevel, name, QualityLevel.HIGH)


class MaxineVSRPlugin(ModelPlugin):
    MODEL_ID     = "nvidia/maxine-vsr"
    DISPLAY_NAME = "Maxine Super Resolution"
    MODEL_TYPE   = "image"
    DESCRIPTION  = "AI super-resolution + denoise + deblur via NVIDIA Maxine"

    INPUTS      = InputSpec.IMAGE
    UI_SECTIONS = [UISection.RESOLUTION, UISection.FRAMES, UISection.SEED]
    PARAMS      = ParamSpec(width=1920, height=1080)

    REQUIRED_PACKAGES          = ["torch", "nvvfx"]
    supports_inpaint           = False
    supports_img2img           = True
    requires_input_strip       = True
    uses_standard_input_strip  = False
    show_enhance               = False
    supports_batch             = False

    def is_available(self):
        try:
            import nvvfx  # noqa: F401
        except ImportError:
            return False
        try:
            import torch
            if not torch.cuda.is_available():
                return False
        except ImportError:
            return False
        return True

    def draw_post_seed_ui(self, col, context):
        col.prop(context.scene, "maxine_quality")

    def load(self, prefs, scene, **kw):
        return {"pipe": None, "converter": None, "refiner": None}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch
        from nvvfx import VideoSuperRes

        image = inputs.image
        if image is None:
            raise ValueError("Maxine VSR requires an input image.")

        quality = _get_quality_level(scene)
        self.set_phase(inputs, "Upscaling")
        tensor = _pil_to_chw(image)

        with VideoSuperRes(quality=quality) as sr:
            sr.output_width = inputs.width
            sr.output_height = inputs.height
            sr.load()
            result = sr.run(tensor)
            output = torch.from_dlpack(result.image).clone()

        return _chw_to_pil(output)
