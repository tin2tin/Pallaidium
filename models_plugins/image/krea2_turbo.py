"""Text-to-image via Krea 2 Turbo (distilled 8-step, CFG-free) with SDNQ 8-bit weights."""

import os

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram


def _registered_adapters(pipe):
    """Names PEFT has actually attached to the pipeline (across all components)."""
    present = set()
    try:
        for v in pipe.get_list_adapters().values():
            present.update(v)
    except Exception:
        pass
    return present


def _load_krea2_lora(pipe, file_path, adapter_name):
    """Attach one LoRA to a Krea2Pipeline, returning True only if it registered.

    diffusers' Krea2LoraLoaderMixin (added in huggingface/diffusers#14046) does
    NO non-diffusers key conversion — it requires diffusers-native keys
    ('transformer.<block>...lora_A/lora_B.weight'). A LoRA exported by ComfyUI
    carries a 'diffusion_model.' prefix (or bare block keys) that matches zero
    transformer params, so load_lora_weights no-ops with only a warning and a
    later set_adapters() would raise. We try the native loader first (correct for
    LoRAs from the Krea 2 DreamBooth trainer), then remap common layouts and
    retry from an in-memory state dict.
    """
    # 1) Native diffusers loader — the supported path for trainer-saved LoRAs.
    try:
        pipe.load_lora_weights(
            os.path.dirname(file_path),
            weight_name=os.path.basename(file_path),
            adapter_name=adapter_name,
        )
    except Exception as e:
        print(f"Krea 2 Turbo: native LoRA load of '{adapter_name}' failed ({e}); trying key remap.")
    if adapter_name in _registered_adapters(pipe):
        return True

    # 2) Remap non-diffusers keys to the 'transformer.' prefix and retry. kohya /
    #    sd-scripts LoRAs (lora_down/lora_up + .alpha) need a different conversion
    #    entirely and the mixin's is_correct_format check rejects the '.alpha'
    #    keys, so detect and bail with a clear message instead of mangling them.
    try:
        from safetensors.torch import load_file
        sd = load_file(file_path)
    except Exception as e:
        print(f"Krea 2 Turbo: could not read '{file_path}' for remap ({e}).")
        return False

    if any(k.endswith(".alpha") or "lora_down" in k or "lora_up" in k for k in sd):
        print(f"Krea 2 Turbo: '{adapter_name}' is a kohya-format LoRA "
              "(lora_down/lora_up/alpha), unsupported by the Krea2 diffusers "
              "loader; re-export in diffusers/PEFT format.")
        return False

    if not any(k.startswith("transformer.") for k in sd):
        if any(k.startswith("diffusion_model.") for k in sd):
            sd = {"transformer." + k[len("diffusion_model."):]: v for k, v in sd.items()}
        else:
            sd = {"transformer." + k: v for k, v in sd.items()}
        try:
            pipe.load_lora_weights(sd, adapter_name=adapter_name)
        except Exception as e:
            print(f"Krea 2 Turbo: remapped LoRA load of '{adapter_name}' failed ({e}).")

    return adapter_name in _registered_adapters(pipe)


class Krea2TurboPlugin(ModelPlugin):
    MODEL_ID     = "OzzyGT/Krea_2_Turbo_sdnq_dynamic_8bit"
    DISPLAY_NAME = "Krea 2 Turbo"
    DESCRIPTION  = "Fast text-to-image via Krea 2 Turbo (12.9B MMDiT, distilled to 8 steps, CFG-free, SDNQ 8-bit)"
    MODEL_TYPE   = "image"
    INPUTS       = InputSpec.PROMPT | InputSpec.NEG_PROMPT | InputSpec.LORA
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.NEG_PROMPT,
        UISection.RESOLUTION, UISection.FRAMES, UISection.STEPS, UISection.GUIDANCE, UISection.SEED,
        UISection.LORA,
    ]
    # Distilled for few-step sampling. The current Krea2Pipeline enables CFG for
    # any guidance_scale > 0 (pred = cond + scale*(cond-uncond)), so guidance=1.0
    # runs the uncond pass and honors negative_prompt. NOT 0.0 — that disables CFG
    # and uses the empty-prompt prediction → noise on this setup.
    PARAMS            = ParamSpec(steps=8, guidance=1.0)
    REQUIRED_PACKAGES = ["torch", "diffusers", "transformers", "sdnq"]
    supports_inpaint  = False
    supports_img2img  = False

    def load(self, prefs, scene, **kw):
        import torch
        from diffusers import Krea2Pipeline, Krea2Transformer2DModel
        from transformers import Qwen3VLModel
        # Registers SDNQ's quantizer with diffusers/transformers so from_pretrained
        # recognizes quant_method="sdnq" in the configs and wraps the int4 weights
        # with the dequantizer. WITHOUT this import the quantized tensors load raw
        # (no dequant) → pure noise. Do NOT also call apply_sdnq_options_to_model:
        # that enables the int8 quantized-matmul / compiled-dequant path, which is
        # broken on this Triton build (cluster_dims) and itself produced noise.
        from sdnq import SDNQConfig  # noqa: F401
        from ...utils.helpers import suppress_text_encoder_warnings

        _cache_dir = prefs.hf_cache_dir or None
        _lfo  = prefs.local_files_only
        dtype = torch.bfloat16
        mode  = kw.get("mode", "txt2img")
        print(f"Loading {self.MODEL_ID} ({mode})…")

        # SDNQ stores int8-quantized weights with the quantization config baked into
        # each subfolder, so from_pretrained loads them as real tensors — no bnb
        # on-the-fly quantization (which leaves meta tensors when offloaded to CPU).
        # Do NOT pass device_map="cpu": enable_model_cpu_offload() handles placement,
        # and device_map="cpu" trips a device-mismatch with newer Transformers.
        print("Loading transformer...")
        transformer = Krea2Transformer2DModel.from_pretrained(
            self.MODEL_ID, subfolder="transformer", torch_dtype=dtype,
            cache_dir=_cache_dir, local_files_only=_lfo,
        )
        # Krea 2 uses the Qwen3-VL text encoder for text-only conditioning, so its
        # vision tower is loaded but never run; silence the resulting "missing
        # weights" / group-offload "unexecuted layers" noise (benign, same as LTX).
        print("Loading text encoder...")
        with suppress_text_encoder_warnings():
            text_encoder = Qwen3VLModel.from_pretrained(
                self.MODEL_ID, subfolder="text_encoder", torch_dtype=dtype,
                cache_dir=_cache_dir, local_files_only=_lfo,
            )
        print("Building pipeline...")
        pipe = Krea2Pipeline.from_pretrained(
            self.MODEL_ID, transformer=transformer, text_encoder=text_encoder,
            torch_dtype=dtype, cache_dir=_cache_dir, local_files_only=_lfo,
        )

        enabled_items = kw.get("enabled_items", [])
        if enabled_items:
            from ...utils.helpers import clean_filename, bpy
            lora_folder = bpy.path.abspath(getattr(bpy.context.scene, "lora_folder", ""))
            names, weights = [], []
            for item in enabled_items:
                name = clean_filename(item.name).replace(".", "")
                fpath = os.path.join(lora_folder, item.name + ".safetensors")
                if _load_krea2_lora(pipe, fpath, name):
                    names.append(name)
                    weights.append(item.weight_value)
                else:
                    print(f"Krea 2 Turbo: LoRA '{item.name}' did not attach to the "
                          "transformer; skipping it.")
            # Only adapters that actually registered are activated, so set_adapters
            # never raises on a name the key/prefix mismatch left unregistered.
            if names:
                pipe.set_adapters(names, adapter_weights=weights)
            else:
                print("Krea 2 Turbo: no enabled LoRA matched the transformer; "
                      "generating without LoRA.")

        # NOTE: deliberately NOT calling apply_sdnq_options_to_model(use_quantized_matmul=True).
        # The repo ships these weights with use_quantized_matmul=False; forcing the
        # quantized int8-matmul path here produced pure-noise output on this setup
        # (the model card's recipe relies on plain SDNQ dequant-on-the-fly, which is
        # numerically correct and what ernie effectively runs). int8 weights still
        # give the memory win; the matmul itself dequantizes per layer.

        # VAE tiling/slicing keeps the final decode of the large latent off the
        # VRAM ceiling — the 12.9B transformer plus a full-frame Qwen-Image VAE
        # decode is what spikes memory at the end of an 8-step run.
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

        # NOTE: do NOT use enable_group_offload here. Leaf-level group offload with
        # streaming moves a module's int weight to GPU but can leave the SDNQ side
        # buffers (scale / zero_point / svd_*) on CPU, so the dequant runs across
        # devices and produces pure noise. enable_model_cpu_offload moves whole
        # components together and is the SDNQ-correct path (same as ernie).
        if gfx_device == "mps":
            pipe.to("mps")
        elif low_vram():
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.enable_model_cpu_offload()
        return {"pipe": pipe, "converter": None, "refiner": None, "preprocessor": None}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch
        from ...utils.helpers import suppress_text_encoder_warnings

        pipe = pipe_obj["pipe"]
        seed = inputs.seed
        # CPU generator: with offloading the component .device reports CPU.
        generator = torch.Generator("cpu").manual_seed(seed) if seed != 0 else None
        self.set_phase(inputs, "Generating")
        # suppress the vision-tower "unexecuted layers" group-offload trace warning.
        with torch.inference_mode(), suppress_text_encoder_warnings():
            return pipe(
                prompt=inputs.prompt,
                # Honored only while guidance_scale > 0 (the pipeline enables the
                # uncond/CFG pass for any positive scale); ignored at 0.0.
                negative_prompt=inputs.neg_prompt,
                num_inference_steps=inputs.steps,
                guidance_scale=inputs.guidance,
                height=inputs.height,
                width=inputs.width,
                max_sequence_length=512,
                generator=generator,
                callback_on_step_end=self.step_callback(inputs),
            ).images[0]
