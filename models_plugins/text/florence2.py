"""Detailed image captioning via Florence-2 (florence-community/Florence-2-large)."""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs


class Florence2Plugin(ModelPlugin):
    MODEL_ID     = "florence-community/Florence-2-large"
    DISPLAY_NAME = "Image Captioning: Florence-2"
    MODEL_TYPE   = "text"
    DESCRIPTION  = "Image Captioning"

    INPUTS       = InputSpec.IMAGE
    UI_SECTIONS  = []   # no prompt needed — image comes from the active strip
    PARAMS       = ParamSpec()
    REQUIRED_PACKAGES = ["torch", "PIL", "transformers"]

    _CAPTION_PROMPT = "<MORE_DETAILED_CAPTION>"

    def load(self, prefs, scene, **kw):
        from transformers import AutoProcessor, Florence2ForConditionalGeneration

        _cache_dir = prefs.hf_cache_dir or None
        model = Florence2ForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            device_map="auto",
            cache_dir=_cache_dir,
        )
        processor = AutoProcessor.from_pretrained(self.MODEL_ID, cache_dir=_cache_dir)
        return {"model": model, "processor": processor, "tokenizer": None}

    def generate(self, pipe, inputs: ModelInputs, scene, prefs) -> str:
        model     = pipe["model"]
        processor = pipe["processor"]
        image     = inputs.image

        if image is None:
            raise ValueError("Florence-2 requires an image input — select an IMAGE or MOVIE strip before adding to queue.")
        if image.mode != "RGB":
            image = image.convert("RGB")

        prompt = self._CAPTION_PROMPT

        # Prepare inputs on the model's device
        proc_inputs = processor(text=prompt, images=image, return_tensors="pt")
        proc_inputs = {k: v.to(model.device) for k, v in proc_inputs.items()}

        self.set_phase(inputs, "Captioning")
        generated_ids = model.generate(
            **proc_inputs,
            max_new_tokens=1024,
            num_beams=3,
            repetition_penalty=1.10,
        )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            generated_text,
            task=prompt,
            image_size=(image.width, image.height),
        )
        text = parsed[prompt]
        print("Florence-2 generated text:", text)
        return text
