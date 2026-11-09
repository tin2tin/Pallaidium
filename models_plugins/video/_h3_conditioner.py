"""Memory-lean Qwen3-VL conditioner pass shared by the MiniMax-H3 family plugins
(minimax_h3.py, pika_h3_flex.py, open_h3_regenerate_2k.py).

diffusers' get_qwen3vl_prompt_embeds runs the whole conditioner with output_hidden_states=True and lets
Qwen3-VL's GQA attention fall to SDPA's math kernel. With an image/video reference in the presentation that
is a ~23 GB transient on a 24 GB card (spill into shared memory = minutes per job). conditioner_hidden()
returns the same `hidden_states[layer]` bit-for-bit at ~7 GB peak.
"""


class _StopForward(Exception):
    pass


class ChunkedSDPA:
    """Run torch's scaled_dot_product_attention in query-row chunks while active.

    Qwen3-VL's GQA attention over a ~6k-row presentation lands on SDPA's math kernel (flash and
    memory-efficient both refuse it), which materialises 64 heads x rows^2 scores in fp32: a ~23 GB transient
    per layer. Softmax is per query row, so slicing the queries (and the mask rows) gives the same result in
    a fraction of the memory. The causal flag is turned into an explicit mask per chunk."""

    def __init__(self, rows=1024):
        self.rows = rows
        self._orig = None

    def __enter__(self):
        import torch
        import torch.nn.functional as F
        orig = self._orig = F.scaled_dot_product_attention
        rows = self.rows

        def sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
            length, keys = query.shape[-2], key.shape[-2]
            if length <= rows:
                return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal,
                            scale=scale, **kwargs)
            outs = []
            for i in range(0, length, rows):
                q = query[..., i:i + rows, :]
                mask = attn_mask
                if mask is not None and mask.dim() >= 2 and mask.shape[-2] != 1:
                    mask = mask[..., i:i + rows, :]
                if is_causal:
                    causal = torch.ones(q.shape[-2], keys, dtype=torch.bool, device=query.device).tril(
                        diagonal=i + keys - length)
                    if mask is None:
                        mask = causal
                    elif mask.dtype == torch.bool:
                        mask = mask & causal
                    else:
                        mask = mask.masked_fill(~causal, float("-inf"))
                outs.append(orig(q, key, value, attn_mask=mask, dropout_p=dropout_p, is_causal=False, scale=scale,
                                 **kwargs))
            return torch.cat(outs, dim=-2)

        F.scaled_dot_product_attention = sdpa
        return self

    def __exit__(self, *exc):
        import torch.nn.functional as F
        F.scaled_dot_product_attention = self._orig
        return False


def conditioner_hidden(text_encoder, processor, token_ids, vision_inputs, layer, device, dtype):
    """`hidden_states[layer]` of the Qwen3-VL conditioner for one tokenized presentation — what diffusers'
    get_qwen3vl_prompt_embeds returns — without `output_hidden_states=True`, which keeps every layer's output
    (~51 x 31k rows x 5120 = ~16 GB for a 5 s clip read as video) and runs the layers past the one used.
    A forward hook on decoder layer `layer - 1` captures its output and stops the forward."""
    import torch
    lm = text_encoder.model.language_model
    if len(lm.layers) <= layer:
        raise ValueError(f"the conditioner has {len(lm.layers)} decoder layers, need more than {layer}")
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    mm_token_type_ids = torch.tensor(processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=device)
    vision_kwargs = {name: (value.to(device, text_encoder.dtype) if name.startswith("pixel_") else value.to(device))
                     for name, value in (vision_inputs or {}).items()}
    # Same as diffusers: fire a top-level accelerate offload hook by hand (no-op under group offload).
    hf_hook = getattr(text_encoder, "_hf_hook", None)
    if hf_hook is not None and hasattr(hf_hook, "pre_forward"):
        hf_hook.pre_forward(text_encoder)
    captured = {}

    def hook(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = hidden.detach().to(device=device, dtype=dtype)
        raise _StopForward

    handle = lm.layers[layer - 1].register_forward_hook(hook)
    try:
        with ChunkedSDPA():
            try:
                text_encoder.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                   mm_token_type_ids=mm_token_type_ids, use_cache=False, **vision_kwargs)
            except _StopForward:
                pass
    finally:
        handle.remove()
    if "hidden" not in captured:
        raise RuntimeError("the conditioner never reached the requested layer")
    return captured["hidden"]


def _lean_get_qwen3vl_prompt_embeds(text_encoder, processor, token_ids, vision_inputs=None, text_encoder_layer=50,
                                    device=None, dtype=None):
    return conditioner_hidden(text_encoder, processor, token_ids, vision_inputs, text_encoder_layer, device, dtype)


def install_lean_conditioner(*module_names):
    """Swap diffusers' get_qwen3vl_prompt_embeds for the lean version in the given modular-pipeline encoder
    modules (the encoder blocks look the function up as a module global at call time). Idempotent."""
    import importlib
    for name in module_names:
        try:
            mod = importlib.import_module(name)
        except Exception as exc:  # pipeline not available in this diffusers build — nothing to patch
            print(f"H3 conditioner: {name} not patched ({exc})")
            continue
        if hasattr(mod, "get_qwen3vl_prompt_embeds"):
            mod.get_qwen3vl_prompt_embeds = _lean_get_qwen3vl_prompt_embeds
