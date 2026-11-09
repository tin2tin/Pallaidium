"""Re-use the vision tower output when consecutive jobs send the same reference images.

Reading the references is by far the slowest part of a Qwen-Image 2.1 job (about 50 s
per image at the default conditioning size, against seconds for the denoising), and
the result depends only on the pixels. It does not depend on the prompt or the seed.

Safe for batches with changing images:
  - The key is the exact input (pixels, order, grid, dtype, device), so different
    references always run the vision tower as usual.
  - Only the latest output is kept, so memory cannot grow during a batch.
  - The entry lives on the patched module and is freed with the pipeline.
  - Any problem in here falls back to the normal, uncached call.

Set the environment variable PALLAIDIUM_NO_VISION_CACHE to switch it off.
"""

import functools
import hashlib
import os

_SIMPLE_TYPES = (type(None), bool, int, float, str)


def _update_digest(digest, value):
    """Add one call argument to the key. Raises on anything it cannot fingerprint."""
    import torch

    if isinstance(value, torch.Tensor):
        digest.update(f"{value.dtype}|{tuple(value.shape)}|{value.device}|".encode())
        data = value.detach()
        if data.is_floating_point():
            data = data.float()  # numpy has no bfloat16; float32 holds it exactly
        digest.update(data.cpu().numpy().tobytes())
    elif isinstance(value, _SIMPLE_TYPES):
        digest.update(repr(value).encode())
    else:
        raise TypeError(f"Cannot fingerprint {type(value).__name__}")


def _make_key(args, kwargs):
    import torch

    if torch.is_grad_enabled():  # cached tensors must never join a training graph
        return None
    digest = hashlib.sha1()
    for value in args:
        _update_digest(digest, value)
    for name in sorted(kwargs):
        digest.update(name.encode())
        _update_digest(digest, kwargs[name])
    return digest.hexdigest()


def _copy_output(output):
    # transformers' get_image_features overwrites pooler_output on the object it
    # receives, so the stored entry and each caller need their own object.
    return type(output)(**dict(output.items()))


def patch_vision_cache(text_encoder):
    """Wrap text_encoder.model.visual.forward. Call after offloading has been set up."""
    visual = text_encoder.model.visual
    if getattr(visual, "_pallaidium_vision_cache", False):
        return
    original = visual.forward
    entry = {"key": None, "output": None}

    @functools.wraps(original)
    def forward(*args, **kwargs):
        key = None
        if not os.environ.get("PALLAIDIUM_NO_VISION_CACHE"):
            try:
                key = _make_key(args, kwargs)
            except Exception as e:
                print(f"Qwen-Image 2.1: vision cache skipped ({e})")
        if key is not None and key == entry["key"]:
            try:
                output = _copy_output(entry["output"])
                print("Qwen-Image 2.1: re-using the reading of the reference image(s)")
                return output
            except Exception as e:
                print(f"Qwen-Image 2.1: vision cache skipped ({e})")
        output = original(*args, **kwargs)
        entry.update(key=None, output=None)
        if key is not None:
            try:
                entry.update(key=key, output=_copy_output(output))
            except Exception as e:
                entry.update(key=None, output=None)
                print(f"Qwen-Image 2.1: vision cache skipped ({e})")
        return output

    visual.forward = forward
    visual._pallaidium_vision_cache = True
