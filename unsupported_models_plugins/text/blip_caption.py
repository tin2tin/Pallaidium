"""Image captioning via BLIP (Salesforce/blip-image-captioning-large)."""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, clean_string


class BlipCaptionPlugin(ModelPlugin):
    MODEL_ID     = "Salesforce/blip-image-captioning-large"
    DISPLAY_NAME = "Image Captioning: Blip"
    MODEL_TYPE   = "text"
    DESCRIPTION  = "Image Captioning"

    INPUTS       = InputSpec.IMAGE
    UI_SECTIONS  = []   # no prompt needed — image comes from the active strip
    PARAMS       = ParamSpec()
    REQUIRED_PACKAGES = ["torch", "PIL", "transformers"]

    def load(self, prefs, scene, **kw):
        import torch
        from transformers import BlipProcessor, BlipForConditionalGeneration

        local = prefs.local_files_only
        processor = BlipProcessor.from_pretrained(self.MODEL_ID, local_files_only=local)
        model = BlipForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.float16,
            local_files_only=local,
        ).to(gfx_device)
        return {"model": model, "processor": processor, "tokenizer": None}

    def generate(self, pipe, inputs: ModelInputs, scene, prefs) -> str:
        import torch

        model     = pipe["model"]
        processor = pipe["processor"]

        inp = processor(inputs.image, "", return_tensors="pt").to(gfx_device, torch.float16)
        out = model.generate(**inp, max_new_tokens=256)
        text = processor.decode(out[0], skip_special_tokens=True)
        text = clean_string(text)
        print("BLIP generated text:", text)
        return text
