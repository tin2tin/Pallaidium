"""Optional low-VRAM text encoders for the MiniMax-H3 family (H3 Image, H3 video, Pika H3 Flex).

Qwen3.5 mapped into the native Qwen3-VL-32B conditioner's hidden_states[50] space by a closed-form
"Plato's Key" (Alissonerdx, Ridge v3, calibrated on 20k prompts). Text only: the key maps text, not the
image/video tokens of edits, keyframes or references.

Measured 2026-11-27: cosine vs native 0.85-0.91 on image and video prompts (shot timings, dialogue,
Sound: lines). Same composition as native at the same seed; exact lettering, object counts and small
details are less reliable ("OPEN 24/7" -> "OPEN 24/5"). The 9B is ~18 GB in bf16 — more than the 4-bit
native encoder — and is offered for comparison.
"""

PK_KEY_REPO = "Alissonerdx/TE-PK-Minimax-H3"
PK_ENCODERS = {  # scene value: (donor repo, files to check in the cache, key file, download GB)
    "qwen35_2b": ("Qwen/Qwen3.5-2B",
                  ("config.json", "tokenizer.json", "model.safetensors-00001-of-00001.safetensors"),
                  "qwen3.5/minimax_h3_32b_pk_ridge_qwen3.5_2b.safetensors", 4.6),
    "qwen35_9b": ("Qwen/Qwen3.5-9B",
                  ("config.json", "tokenizer.json") + tuple(
                      f"model.safetensors-0000{i}-of-00004.safetensors" for i in range(1, 5)),
                  "qwen3.5/minimax_h3_32b_pk_ridge_qwen3.5_9b.safetensors", 19.5),
}

TEXT_ENCODER_ITEMS = [
    ("native", "Native (Qwen3-VL-32B)", "The model's own conditioner (~15 GB VRAM). Needed for image/video inputs"),
    ("qwen35_2b", "Qwen3.5-2B (Plato's Key)",
     "Qwen3.5-2B mapped into the native encoder's space (~4 GB VRAM, 4.6 GB download). Prompt-only "
     "generation (image inputs switch to Native automatically); text in the image, counts and small details are less exact"),
    ("qwen35_9b", "Qwen3.5-9B (Plato's Key)",
     "Qwen3.5-9B mapped into the native encoder's space (~18 GB VRAM, 19.5 GB download): more memory than "
     "Native, for comparison. Prompt-only generation"),
]


def lite_choice(scene, prop: str):
    """The Plato's Key encoder chosen on the scene, or None for the native conditioner."""
    choice = getattr(scene, prop, "native")
    return choice if choice in PK_ENCODERS else None


def lite_in_cache(choice, in_cache, cache_dir) -> bool:
    repo, files, key_file, _gb = PK_ENCODERS[choice]
    return in_cache(repo, files, cache_dir) and in_cache(PK_KEY_REPO, (key_file,), cache_dir)


def lite_download_note(choice) -> str:
    repo, _files, _key, gb = PK_ENCODERS[choice]
    return f"{repo.split('/')[-1]} text encoder + key (~{gb} GB)"


def platos_key_class():
    import torch

    class PlatosKeyTextEncoder(torch.nn.Module):
        """Qwen3.5 + a linear map standing in for the native Qwen3-VL-32B conditioner (text only).

        The key was fitted on raw prompts (no chat template, no special tokens) against the native
        encoder's unnormalized hidden_states[50], from the donor's final-norm last hidden state."""

        def __init__(self, donor, tokenizer, key):
            super().__init__()
            self.model = donor
            self.tokenizer = tokenizer
            for name in ("R", "scale", "donor_mean", "target_mean"):
                self.register_buffer(name, key[name].float(), persistent=False)

        @property
        def dtype(self):
            return torch.bfloat16

        @torch.no_grad()
        def encode(self, text: str, length: int, device, dtype):
            ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
            ids = ids[:, : max(1, length * 2)].to(device)
            x = self.model(input_ids=ids, output_hidden_states=True).hidden_states[-1].float()
            key = {n: getattr(self, n).to(device) for n in ("R", "scale", "donor_mean", "target_mean")}
            y = (x - key["donor_mean"]) @ key["R"] * key["scale"] + key["target_mean"]
            if y.shape[1] != length:
                # The two vocabularies split non-English text differently; the transformer tags one
                # row per native token, so resample the sequence to that length.
                print(f"Plato's Key: Qwen3.5 tokens {y.shape[1]} vs native {length} — resampling")
                y = torch.nn.functional.interpolate(y.transpose(1, 2), size=length, mode="linear",
                                                    align_corners=True).transpose(1, 2)
            return y.to(device=device, dtype=dtype)

    return PlatosKeyTextEncoder


def load_lite_text_encoder(choice, cache_dir, local_files_only, label):
    """Qwen3.5 + Plato's Key, from the HF cache first (online only for what isn't cached)."""
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    repo, _files, key_file, _gb = PK_ENCODERS[choice]
    print(f"{label}: text encoder {repo} + Plato's Key ({PK_KEY_REPO})")

    def cache_first(fn):
        try:
            return fn(True)
        except (OSError, ValueError):
            if local_files_only:
                raise
            return fn(False)

    hub = lambda lfo: dict(cache_dir=cache_dir or None, local_files_only=lfo)  # noqa: E731
    tokenizer = cache_first(lambda lfo: AutoTokenizer.from_pretrained(repo, **hub(lfo)))
    donor = cache_first(lambda lfo: AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, **hub(lfo)))
    key_path = cache_first(lambda lfo: hf_hub_download(PK_KEY_REPO, key_file, **hub(lfo)))
    # The LM head is never used (hidden states only); drop it to save ~1 GB.
    donor = getattr(donor, "model", donor)
    return platos_key_class()(donor.eval(), tokenizer, load_file(key_path)).eval()


def patch_lite_embeds(encoders_module, native_tokenizer):
    """Route the encoder module's get_qwen3vl_prompt_embeds through the lite encoder for this call.
    Returns an undo callable. Text only: any vision input is an error."""
    import torch
    native_fn = encoders_module.get_qwen3vl_prompt_embeds

    def _lite_embeds(text_encoder, processor, token_ids, vision_inputs=None, *a, device=None, dtype=None, **kw):
        if vision_inputs:
            raise RuntimeError("The Qwen3.5 text encoders take text only — set Text Encoder to Native for "
                               "image or video inputs.")
        text = native_tokenizer.decode(token_ids)
        return text_encoder.encode(text, len(token_ids), device, dtype or torch.bfloat16)

    encoders_module.get_qwen3vl_prompt_embeds = _lite_embeds
    return lambda: setattr(encoders_module, "get_qwen3vl_prompt_embeds", native_fn)


def swap_text_encoder(pipe, choice, load_native, load_lite, place, label):
    """Make pipe.text_encoder the wanted encoder (choice=None → native), loading it if needed.

    The pipeline is cached per scene setting, so a Qwen3.5 pipeline that met an image job holds the native
    encoder afterwards (and the other way round): swap on demand. place(te, lite) puts it on the device."""
    import gc
    import torch
    current = getattr(pipe, "_h3_te_choice", None)
    if current == choice:
        return False
    print(f"{label}: switching text encoder {current or 'native'} -> {choice or 'native'}")
    te = load_lite(choice) if choice else load_native()
    place(te, bool(choice))
    pipe.update_components(text_encoder=te)  # drops the old encoder
    pipe._h3_te_choice = choice
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True


def switch_note(wanted, used) -> str:
    """Queue note shown on the finished job when the encoder differs from the panel choice."""
    if wanted and not used:
        return f"Text Encoder {PK_ENCODERS[wanted][0].split('/')[-1]} is prompt-only: used Native for the reference input"
    return ""


def hide_keyframes_from_conditioner(step_cls, undo_lite):
    """Run an fl2va text step as if there were no keyframes (text-only presentation). The keyframes still
    reach the clip through the keyframe VAE step, which reads them from the state afterwards.

    Measured 2026-11-27 (H3, first frame + dialogue prompt, same seed): Qwen3.5-2B with the keyframe through
    the VAE only gave nearly the same clip as native (same identity, same motion beats)."""
    orig = step_cls.__call__

    def _call(self, components, state):
        keyframes = state.get("keyframes")
        state.set("keyframes", None)
        try:
            return orig(self, components, state)
        finally:
            state.set("keyframes", keyframes)

    step_cls.__call__ = _call

    def _undo():
        step_cls.__call__ = orig
        undo_lite()
    return _undo
