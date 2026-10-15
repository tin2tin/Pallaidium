"""Text-to-image and reference-image generation via Krea 2 Turbo (distilled 8-step) with SDNQ 8-bit weights."""

import os

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import gfx_device, low_vram


# Reference conditioning runs through OzzyGT's Modular Diffusers blocks, which
# feed the references through Qwen3-VL's vision tower (Krea 2's text encoder is
# the full VLM) and, in the LoRA modes, into the transformer as clean latents.
_REF_BLOCKS_ID = "OzzyGT/krea2_reference_blocks"

# reference_mode -> (adapter name, LoRA repo, weight file). "off" needs none.
_REF_MODE_LORAS = {
    "append":  ("krea2_ref_style", "ostris/krea2_turbo_style_reference",
                "krea2_style_reference.safetensors"),
    "prepend": ("krea2_ref_identity", "conradlocke/krea2-identity-edit",
                "krea2_identity_edit_v1_2.safetensors"),
}


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
    DESCRIPTION  = ("Fast text-to-image via Krea 2 Turbo (12.9B MMDiT, distilled to 8 steps, SDNQ 8-bit) "
                    "with up to 9 reference images (vision, style or identity edit)")
    MODEL_TYPE   = "image"
    INPUTS       = InputSpec.PROMPT | InputSpec.NEG_PROMPT | InputSpec.IMAGE | InputSpec.LORA
    UI_SECTIONS  = [
        UISection.PROMPT, UISection.NEG_PROMPT, UISection.IMAGE_STRIP,
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
    # image= is reference conditioning (vision tokens + optional clean latents),
    # not a denoise blend — there is no strength for the Image Strength slider.
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
        # Text-only prompts never run the Qwen3-VL vision tower (only reference
        # jobs do); silence the resulting "missing weights" / group-offload
        # "unexecuted layers" noise (benign, same as LTX).
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
                # Remembered so each job can re-activate them alongside (or
                # without) a reference-mode LoRA — see _activate_adapters.
                pipe._krea2_user_adapters = (names, weights)
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
        # krea2_ref_pipe is built on the first reference job (_reference_pipe).
        # The explicit None resets a stale one: the queue reloads for a LoRA
        # change via model_cache.update() without releasing the old keys.
        return {"pipe": pipe, "converter": None, "refiner": None, "preprocessor": None,
                "krea2_ref_pipe": None}

    def draw_custom_ui(self, col, context) -> bool:
        scene = context.scene
        # Strip refs live in the scene shown in the VSE (context.sequencer_scene
        # in Blender 5.x), which can differ from the active scene. Reuses the
        # klein_strip_N / klein_visible_strips pickers shared by every
        # up-to-9-reference plugin (Klein, Qwen-Image 2.1), whose strip names and
        # rendered paths already flow through the queue, metadata and Redo.
        vse_scene = getattr(context, "sequencer_scene", None) or context.scene
        try:
            col.prop(scene, "input_strips", text="Input")
        except Exception:
            pass
        if vse_scene.sequence_editor is None:
            return True
        for i in range(1, scene.klein_visible_strips + 1):
            row = col.row(align=True)
            row.prop_search(
                vse_scene, f"klein_strip_{i}", vse_scene.sequence_editor, "strips",
                text="Ref.", icon="FILE_IMAGE",
            )
            row.operator("sequencer.strip_picker", text="", icon="EYEDROPPER").action = f"klein_select{i}"
            if i == scene.klein_visible_strips and scene.klein_visible_strips < 9:
                if scene.klein_visible_strips > 3:
                    row.operator("object.klein_hide_strip", text="", icon="REMOVE").strip_index = i
                row.operator("object.klein_add_strip", text="", icon="ADD")
        if hasattr(scene, "krea2_ref_mode"):
            col.prop(scene, "krea2_ref_mode", text="Mode")
            row = col.row(align=True)
            row.prop(scene, "krea2_ref_subject")
            row.prop(scene, "krea2_ref_style")
            if scene.krea2_ref_mode == "prepend":
                col.prop(scene, "krea2_ref_grounding_px")
        return True

    @staticmethod
    def _collect_references(inputs, scene):
        """The active input (strip/scene frame) first, then the filled reference slots."""
        from PIL import Image as _PILImage
        images = [inputs.image.convert("RGB")] if inputs.image is not None else []
        for attr in (f"klein_strip_{i}_path" for i in range(1, 10)):
            path = getattr(scene, attr, None)
            if path and os.path.isfile(path):
                try:
                    images.append(_PILImage.open(path).convert("RGB"))
                except Exception as e:
                    print(f"Krea 2 Turbo: reference '{path}' failed to open: {e}")
        return images

    @staticmethod
    def _activate_adapters(pipe, mode_adapter=None):
        """Enable the user's LoRAs plus, for a LoRA reference mode, its adapter.

        Mode LoRAs stay attached to the shared transformer once downloaded, so
        every job sets the active list: plain text-to-image and the other modes
        must run with them switched off.
        """
        if not _registered_adapters(pipe):
            return
        names, weights = getattr(pipe, "_krea2_user_adapters", ([], []))
        names, weights = list(names), list(weights)
        if mode_adapter:
            names.append(mode_adapter)
            weights.append(1.0)
        if names:
            pipe.enable_lora()
            pipe.set_adapters(names, adapter_weights=weights)
        else:
            pipe.disable_lora()

    @staticmethod
    def _ensure_mode_lora(pipe, ref_mode, prefs):
        """Download and attach the LoRA a reference mode needs; returns its adapter name."""
        entry = _REF_MODE_LORAS.get(ref_mode)
        if entry is None:
            return None
        adapter, repo_id, weight_name = entry
        if adapter not in _registered_adapters(pipe):
            print(f"Krea 2 Turbo: loading {repo_id} for reference mode '{ref_mode}'…")
            pipe.load_lora_weights(
                repo_id, weight_name=weight_name, adapter_name=adapter,
                cache_dir=prefs.hf_cache_dir or None, local_files_only=prefs.local_files_only,
            )
            if adapter not in _registered_adapters(pipe):
                raise RuntimeError(f"Krea 2 Turbo: the {repo_id} LoRA did not attach to the transformer.")
        return adapter

    @staticmethod
    def _reference_pipe(pipe_obj, prefs):
        """The modular reference pipeline, built once over the loaded components.

        Shares the transformer/text encoder/VAE (with their offload hooks and
        LoRA adapters) with the stock pipeline, so no weights load twice.
        """
        ref_pipe = pipe_obj.get("krea2_ref_pipe")
        if ref_pipe is not None:
            return ref_pipe
        from diffusers import ModularPipelineBlocks
        pipe = pipe_obj["pipe"]
        print(f"Krea 2 Turbo: building reference pipeline from {_REF_BLOCKS_ID}…")
        blocks = ModularPipelineBlocks.from_pretrained(
            _REF_BLOCKS_ID, trust_remote_code=True,
            cache_dir=prefs.hf_cache_dir or None, local_files_only=prefs.local_files_only,
        )
        ref_pipe = blocks.init_pipeline()
        ref_pipe.update_components(
            transformer=pipe.transformer, text_encoder=pipe.text_encoder,
            tokenizer=pipe.tokenizer, vae=pipe.vae, scheduler=pipe.scheduler,
        )
        pipe_obj["krea2_ref_pipe"] = ref_pipe
        return ref_pipe

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs):
        import torch
        from ...utils.helpers import suppress_text_encoder_warnings

        pipe = pipe_obj["pipe"]
        seed = inputs.seed
        # CPU generator: with offloading the component .device reports CPU.
        generator = torch.Generator("cpu").manual_seed(seed) if seed != 0 else None

        references = self._collect_references(inputs, scene)
        if references:
            return self._generate_with_references(pipe_obj, inputs, scene, prefs, references, generator)

        self._activate_adapters(pipe)
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

    def _generate_with_references(self, pipe_obj, inputs, scene, prefs, references, generator):
        import torch
        from ...utils.helpers import suppress_text_encoder_warnings

        pipe = pipe_obj["pipe"]
        ref_mode = getattr(scene, "krea2_ref_mode", "off") or "off"
        if ref_mode not in ("off", "append", "prepend"):
            ref_mode = "off"
        self.set_phase(inputs, "Preparing references")
        self._activate_adapters(pipe, self._ensure_mode_lora(pipe, ref_mode, prefs))
        ref_pipe = self._reference_pipe(pipe_obj, prefs)
        print(f"Krea 2 Turbo: {len(references)} reference image(s), mode={ref_mode}")

        # Krea 2 latents are packed in 16 px patches.
        width = max(16, int(inputs.width) // 16 * 16)
        height = max(16, int(inputs.height) // 16 * 16)

        # The reference-latent step encodes with pixels.to(vae.device), which is
        # the CPU (or meta) under model/sequential offload while the offload hook
        # runs the VAE on the GPU. Report the execution device for the run.
        vae = ref_pipe.vae
        exec_device = ref_pipe._execution_device
        vae_cls = vae.__class__
        vae.__class__ = type(vae_cls.__name__, (vae_cls,), {"device": property(lambda self: exec_device)})

        # Modular pipelines take no callback_on_step_end, so report progress
        # from the scheduler step, which runs once per denoising step.
        scheduler = ref_pipe.scheduler
        original_step = scheduler.step
        progress_fn = inputs.progress_fn
        total = max(1, inputs.steps)
        counter = [0]

        def _step(*args, **kwargs):
            out = original_step(*args, **kwargs)
            counter[0] += 1
            if progress_fn is not None:
                progress_fn(counter[0], total)
            return out

        kwargs = dict(
            prompt=inputs.prompt,
            reference_images=references,
            reference_mode=ref_mode,
            reference_subject_strength=float(getattr(scene, "krea2_ref_subject", 1.0)),
            reference_style_strength=float(getattr(scene, "krea2_ref_style", 1.0)),
            height=height,
            width=width,
            num_inference_steps=inputs.steps,
            max_sequence_length=512,
            generator=generator,
            output="images",
        )
        if ref_mode == "prepend":
            # Identity-Edit recipe from the model card's example script.
            kwargs["grounding_px"] = int(getattr(scene, "krea2_ref_grounding_px", 768))
            kwargs["mask_reference_latents"] = True

        scheduler.step = _step
        self.set_phase(inputs, "Generating")
        try:
            # The distilled reference blocks run CFG-free: no negative prompt.
            with torch.inference_mode(), suppress_text_encoder_warnings():
                return ref_pipe(**kwargs)[0]
        finally:
            scheduler.step = original_step
            vae.__class__ = vae_cls
