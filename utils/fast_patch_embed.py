"""Run vision-tower patch embeddings (a Conv3d whose kernel equals its stride and its input) as one matmul.

Qwen2-VL / Qwen2.5-VL / Qwen3-VL (and the vision towers built on them) patchify images and video into
(N, C, T, P, P) blocks and project each with a Conv3d of kernel (T, P, P) — mathematically a linear layer.
With torch 2.9.1 / cuDNN 9.10 on Windows the bf16 Conv3d takes ~14 ms per patch: 248 s for the 17.7k
patches of two 1024-px references (measured 2026-11-10 in MiniMax H3 Image), against 2.2 ms as F.linear
(difference: 1 bf16 ulp). Every plugin that feeds images or video through such a tower pays that cost.

Overriding `_conv_forward` (which Conv3d.forward calls) keeps offload hooks on the module in charge of
moving its weights. Idempotent; any other geometry falls back to the real convolution.
"""


def _patch_conv3d(conv) -> bool:
    import torch.nn.functional as F
    if getattr(conv, "_pallaidium_fast", False):
        return False
    kernel, stride, padding = tuple(conv.kernel_size), tuple(conv.stride), conv.padding
    no_padding = padding == "valid" if isinstance(padding, str) else not any(padding)
    if stride != kernel or conv.groups != 1 or not no_padding or any(d != 1 for d in conv.dilation):
        return False
    original = conv._conv_forward

    def _conv_forward(input, weight, bias):
        if input.dim() == 5 and tuple(input.shape[2:]) == kernel:
            out = F.linear(input.reshape(input.shape[0], -1), weight.reshape(weight.shape[0], -1), bias)
            return out.view(input.shape[0], weight.shape[0], 1, 1, 1)
        return original(input, weight, bias)

    conv._conv_forward = _conv_forward
    conv._pallaidium_fast = True
    return True


def install_fast_patch_embed(model, label: str = "") -> int:
    """Patch every stride==kernel Conv3d inside `model` (a VLM, its vision tower, or a pipeline component).
    Returns how many were patched; safe to call on models without one (returns 0)."""
    if model is None:
        return 0
    try:
        import torch.nn as nn
        count = sum(_patch_conv3d(m) for m in model.modules() if isinstance(m, nn.Conv3d))
    except Exception as exc:
        print(f"fast patch embed{' (' + label + ')' if label else ''}: skipped ({exc})")
        return 0
    if count and label:
        print(f"{label}: vision patch embedding runs as a matmul ({count} Conv3d)")
    return count
