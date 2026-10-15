# This file is licensed under the GNU General Public License v3.0 (https://www.gnu.org/licenses/gpl-3.0.html),
# not the Apache-2.0 license of the rest of this repository, because it is derived from ComfyUI
# (https://github.com/Comfy-Org/ComfyUI), which is GPL-3.0.
# Vendored verbatim from https://github.com/asomoza/diffusers-recipes/blob/main/models/qwen_image_2_1/scripts/qwen_image_21_vae_strips.py

"""Seam-free, memory-bounded decode and encode for the Qwen-Image-2.1 VAE.

NOTE: this is a workaround for the current version of diffusers (0.41.0.dev0). Its `AutoencoderKLQwenImage21`
either decodes the full frame at once, which runs out of memory at the model's native resolutions, or tiles,
which leaves visible seams. Drop this file once diffusers ships a seam-free low-memory decode for this VAE.

Usage: `patch_vae(pipe.vae)` after loading the pipeline, instead of `pipe.vae.enable_tiling()`.

Credits: the strip approach and `strip_apply` come from ComfyUI
(https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/wan/vae2_2.py), adapted here to the diffusers modules.

The layers run in row strips. Above the latent resolution the VAE is all 3x3 convolutions and a per-pixel norm,
so a strip carrying one extra row per 3x3 convolution on each side computes its interior rows exactly as the full
frame would. The output matches the untiled decode with no blending, so there is nothing to seam. The mid block,
whose attention spans the whole frame, runs unstripped at latent resolution, where it is small.
"""

import torch
from diffusers.models.autoencoders.vae import DecoderOutput


# Output elements per strip. Bounds each layer's working set to a slice of the frame.
STRIP_ELEMS = 2**24


def strip_apply(fn, x, scale=1, halo=1, out=None):
    """Apply `fn` over row strips of `x`, each widened by `halo` input rows per side. `scale` is `fn`'s upscale."""
    strips = -(-x.numel() * scale * scale // STRIP_ELEMS)
    if strips <= 1 and out is None:
        return fn(x)
    add = out is not None
    size = x.shape[-2]
    step = -(-size // max(strips, 1))
    for start in range(0, size, step):
        stop = min(size, start + step)
        lo = max(0, start - halo)
        hi = min(size, stop + halo)
        y = fn(x.narrow(-2, lo, hi - lo)).narrow(-2, (start - lo) * scale, (stop - start) * scale)
        if out is None:
            out = y.new_empty(*y.shape[:-2], size * scale, y.shape[-1])
        dst = out.narrow(-2, start * scale, (stop - start) * scale)
        if add:
            dst.add_(y)
        else:
            dst.copy_(y)
    return out


def strip_apply_down(fn, x, bottom=1, out=None):
    """Apply a 2x downsample `fn` over row strips of `x`; output rows `[a, b)` read input rows `[2a, 2b+bottom)`."""
    size = x.shape[-2]
    out_size = size // 2
    strips = -(-x.numel() // 4 // STRIP_ELEMS)
    if strips <= 1 and out is None:
        return fn(x)
    add = out is not None
    step = -(-out_size // max(strips, 1))
    for start in range(0, out_size, step):
        stop = min(out_size, start + step)
        lo, hi = 2 * start, min(size, 2 * stop + bottom)
        y = fn(x.narrow(-2, lo, hi - lo)).narrow(-2, 0, stop - start)
        if out is None:
            out = y.new_empty(*y.shape[:-2], out_size, y.shape[-1])
        dst = out.narrow(-2, start, stop - start)
        if add:
            dst.add_(y)
        else:
            dst.copy_(y)
    return out


def _down_block(block, x):
    shortcut_input = x
    for resnet in block.resnets:
        x = strip_apply(resnet, x, halo=2)
    shortcut = block.avg_shortcut
    if block.downsampler is not None:
        x = strip_apply_down(block.downsampler, x, bottom=1)
        strip_apply_down(shortcut, shortcut_input, bottom=0, out=x)
    else:
        strip_apply(shortcut, shortcut_input, halo=0, out=x)
    return x


def _up_block(block, x):
    shortcut_input = x
    for resnet in block.resnets:
        x = strip_apply(resnet, x, halo=2)
    if block.upsampler is not None:
        x = strip_apply(block.upsampler, x, scale=2, halo=1)
    if block.avg_shortcut is not None:
        shortcut = block.avg_shortcut
        strip_apply(lambda s: shortcut(s, first_chunk=True), shortcut_input, scale=shortcut.factor_s, halo=0, out=x)
    return x


def strip_encode(vae, x):
    """Encode a single-frame image `(B, C, 1, H, W)` to the raw moments `vae._encode` returns."""
    encoder = vae.encoder
    h = strip_apply(encoder.conv_in, x, halo=1)
    for block in encoder.down_blocks:
        h = _down_block(block, h)
    h = encoder.mid_block(h)
    h = encoder.conv_out(encoder.nonlinearity(encoder.norm_out(h)))
    return vae.quant_conv(h)


def strip_decode(vae, z):
    """Decode single-frame latents `(B, C, 1, H, W)` to `(B, C_out, 1, H*16, W*16)`, clamped to [-1, 1]."""
    decoder = vae.decoder
    x = vae.post_quant_conv(z)
    x = decoder.conv_in(x)
    x = decoder.mid_block(x)
    for block in decoder.up_blocks:
        x = _up_block(block, x)

    def head(s):
        return decoder.conv_out(decoder.nonlinearity(decoder.norm_out(s)))

    x = strip_apply(head, x, halo=1)
    return torch.clamp(x, min=-1.0, max=1.0)


def patch_vae(vae):
    """Route `vae.encode` / `vae.decode` through the strip versions for single-frame inputs.

    Patches the private `_encode` / `_decode`, so the public methods keep their offload hooks.
    """
    if getattr(vae.config, "patch_size", None) is not None:
        raise ValueError("patch_vae needs an AutoencoderKLQwenImage21 without patch_size")
    original_encode, original_decode = vae._encode, vae._decode

    def _encode(x):
        return strip_encode(vae, x) if x.shape[2] == 1 else original_encode(x)

    def _decode(z, return_dict=True):
        if z.shape[2] != 1:
            return original_decode(z, return_dict=return_dict)
        sample = strip_decode(vae, z)
        return DecoderOutput(sample=sample) if return_dict else (sample,)

    vae._encode = _encode
    vae._decode = _decode
    return vae
