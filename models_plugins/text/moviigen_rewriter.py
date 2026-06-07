"""Cinematic prompt enhancer via MoviiGen (ZuluVision/MoviiGen1.1_Prompt_Rewriter)."""

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import remove_duplicate_phrases

_SYSTEM_MSG = (
    "As a cinematic prompt engineer, be creative, rewrite the following into a "
    "comma-separated list of visual details, starting with camera angle, camera "
    "motion and progressing through subject, setting, lighting, atmosphere, style."
)


class MoviiGenRewriterPlugin(ModelPlugin):
    MODEL_ID     = "ZuluVision/MoviiGen1.1_Prompt_Rewriter"
    DISPLAY_NAME = "Prompt Enhancer: MoviiGen"
    MODEL_TYPE   = "text"
    DESCRIPTION  = "MoviiGen Prompt Rewriter"

    INPUTS       = InputSpec.PROMPT   # text-only; no image input
    UI_SECTIONS  = [UISection.PROMPT]
    PARAMS       = ParamSpec()
    REQUIRED_PACKAGES = ["torch", "transformers"]

    def load(self, prefs, scene, **kw):
        import torch
        from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
        from torchao.quantization.quant_api import Int8WeightOnlyConfig

        _cache_dir = prefs.hf_cache_dir or None
        print("Loading MoviiGen Prompt Rewriter…")
        quantization_config = TorchAoConfig(Int8WeightOnlyConfig())

        model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quantization_config,
            cache_dir=_cache_dir,
            local_files_only=prefs.local_files_only,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID, cache_dir=_cache_dir, local_files_only=prefs.local_files_only)
        return {"model": model, "processor": None, "tokenizer": tokenizer}

    def generate(self, pipe, inputs: ModelInputs, scene, prefs) -> str:
        model     = pipe["model"]
        tokenizer = pipe["tokenizer"]

        messages = [
            {"role": "system", "content": _SYSTEM_MSG},
            {"role": "user",   "content": inputs.prompt},
        ]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([formatted], return_tensors="pt").to(model.device)

        self.set_phase(inputs, "Generating")
        generated_ids = model.generate(**model_inputs, max_new_tokens=512)
        # Strip the prompt tokens from the output
        trimmed = [
            out[len(inp):]
            for inp, out in zip(model_inputs.input_ids, generated_ids)
        ]
        text = tokenizer.batch_decode(trimmed, skip_special_tokens=True)[0]
        text = remove_duplicate_phrases(text)
        print("MoviiGen enhanced prompt:", text)
        return text
