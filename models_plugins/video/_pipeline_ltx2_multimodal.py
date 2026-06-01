import copy
from dataclasses import dataclass
from typing import Any, Callable

import av
import numpy as np
import PIL.Image
import torch
import torchaudio
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import (
    LTX2ConditionPipeline,
    LTX2VideoCondition,
    calculate_shift,
    rescale_noise_cfg,
    retrieve_latents,
    retrieve_timesteps,
)
from diffusers.pipelines.ltx2.pipeline_output import LTX2PipelineOutput
from diffusers.utils import is_torch_xla_available, logging
from diffusers.utils.torch_utils import randn_tensor


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)


@dataclass
class LTX2ImageCondition:
    """
    Image conditioning for LTX-2.3 — a single image anchored at a specific pixel frame.

    By convention, `frame=0` is treated as a **replace** condition (hard constraint: the first-frame
    token is set to the image latent and locked via the denoise mask). Any other frame is treated
    as a **keyframe** condition (soft guidance: the image latent is appended as a reference token
    with RoPE coordinates pointing at the snapped latent block, so the model sees it but the
    surrounding frames can interpolate smoothly around it).

    The LTX-2 VAE only allows anchors at pixel positions {0, 1, 9, 17, 25, ...} (i.e. `8N+1` for
    `N>=1`, plus 0). The given `frame` is snapped DOWN to the nearest valid position internally;
    if a snap occurs an INFO line is logged so callers can spot it. Examples (with VAE temporal
    ratio 8): `frame=5` → pixel 1, `frame=10` → pixel 9, `frame=24` → pixel 17.

    Attributes:
        image: The image. Accepts any type handled by `VideoProcessor.preprocess_video`
            (`PIL.Image`, `np.ndarray` of shape `(H, W, C)`, `torch.Tensor` of shape `(C, H, W)`).
        frame: Pixel frame index (0-based). `0` = first frame (replace). Negative indices count
            from the end (e.g. `-1` = last pixel frame). Must be within `[-num_frames, num_frames)`.
        strength: Conditioning strength in `[0, 1]`. `1.0` = fully applied.
    """

    image: Any
    frame: int = 0
    strength: float = 1.0


@dataclass
class LTX2AudioCondition:
    """
    Audio conditioning for LTX-2.3 — single audio segment applied over a time range on the output timeline.

    Attributes:
        audio: Raw waveform tensor of shape `(channels, samples)`. Mono is duplicated to stereo before
            the audio VAE (which expects `[B, 2, time, 64]`). Sample rate is assumed to match the audio
            VAE's `sample_rate`; resample beforehand if it doesn't.
        start_time: Seconds on the OUTPUT timeline where this segment begins. Defaults to 0.0.
        end_time: Seconds on the OUTPUT timeline where this segment ends. `None` = run to the natural
            end of `audio`.
        strength: Conditioning strength in `[0, 1]`. 1.0 = fully preserved (audio is locked, not
            regenerated). 0.0 = ignored. Mask convention matches `LTX2VideoCondition.strength`.
    """

    audio: torch.Tensor
    start_time: float = 0.0
    end_time: float | None = None
    strength: float = 1.0


def load_audio(path: str, target_sample_rate: int, seconds: float | None = None) -> torch.Tensor:
    """
    Load an audio file via torchaudio and resample it to `target_sample_rate`.
    Mono stays mono; stereo stays stereo — the pipeline handles mono→stereo duplication before the audio VAE.

    Args:
        path: Path to an audio file (mp3, wav, flac, etc.).
        target_sample_rate: Sample rate to resample to. Typically `pipe.audio_vae.config.sample_rate`
            (16000 for LTX-2.3).
        seconds: If set, trim the waveform to at most this many seconds.

    Returns:
        Tensor of shape `(channels, samples)` at `target_sample_rate`.
    """
    waveform, sr = torchaudio.load(path)
    if sr != target_sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)(waveform)
    if seconds is not None:
        waveform = waveform[:, : int(seconds * target_sample_rate)]
    return waveform


def load_video(path: str) -> list[PIL.Image.Image]:
    """
    Load a video file (or URL) into a list of RGB PIL images via PyAV.

    Args:
        path: Local path or HTTP(S) URL to a video file. PyAV opens URLs directly via ffmpeg.

    Returns:
        List of `PIL.Image.Image` in RGB, one per decoded frame.
    """
    frames: list[PIL.Image.Image] = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            frames.append(frame.to_image())
    return frames


class LTX2MultiModalPipeline(LTX2ConditionPipeline):
    """
    LTX-2.3 pipeline with arbitrary frame/time conditioning for video + audio, and IC-LoRA control.

    Inherits the base `__init__`, text encoding, prompt-embed preparation, video conditioning mask
    machinery, audio/video packing, and guidance math from `LTX2ConditionPipeline`. Adds:

    - `audio_conditions`: list of `LTX2AudioCondition` — time-ranged **source** audio locks
      (replace / inpaint on the OUTPUT timeline). Independent of IC-LoRA.
    - `control_video` / `control_video_latents`: reference video for a video IC-LoRA. Tokens are
      appended with target-space spatial RoPE.
    - `control_downscale_factor`: spatial scale factor for a low-res reference video (from the LoRA card).
    - `control_strength`: how clean the video reference tokens are held (1.0 = fully clean).
    - `control_audio` / `control_audio_latents`: **reference** audio for an audio IC-LoRA. Tokens are
      PREPENDED to the audio sequence with RoPE coords shifted to negative time so the reference sits
      at `t ∈ [-T_ref, 0)` while the target sits at `t ∈ [0, T_target)`. Independent from
      `audio_conditions` (source) — the two can be combined: ref tokens are prepended (mask=1.0)
      and the existing source mask is kept on the target portion.
    - `control_audio_strength`: how clean the audio reference tokens are held (1.0 = fully clean).
    - `identity_guidance_scale`: optional, **opt-in** extra forward pass per step on the
      target-only audio (ref tokens stripped). The delta amplifies the contribution of the audio
      reference on the target tokens — useful for ID-LoRA-style audio identity transfer. `0.0`
      (default) disables the extra pass entirely so audio LoRAs that don't need it pay nothing.
      Only takes effect when `control_audio` / `control_audio_latents` is also set.
    """

    def _waveform_to_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert a raw waveform to the log-mel spectrogram format expected by the audio VAE.

        Args:
            waveform: Tensor of shape `(channels, samples)`. Mono (`channels == 1`) is duplicated
                to stereo. Sample rate must match `self.audio_vae.config.sample_rate`.

        Returns:
            Tensor of shape `(1, 2, time, n_mels)` — the leading 1 is the batch dim.
        """
        if waveform.ndim != 2:
            raise ValueError(f"Expected waveform of shape (channels, samples), got shape {tuple(waveform.shape)}.")

        if waveform.size(0) == 1:
            waveform = waveform.repeat(2, 1)  # mono → stereo
        elif waveform.size(0) != 2:
            raise ValueError(f"Expected 1 (mono) or 2 (stereo) channels, got {waveform.size(0)}.")

        sample_rate = self.audio_vae.config.sample_rate
        mel_hop_length = self.audio_vae.config.mel_hop_length
        n_mels = self.audio_vae.config.mel_bins
        n_fft = 1024

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=mel_hop_length,
            f_min=0.0,
            f_max=sample_rate / 2.0,
            n_mels=n_mels,
            window_fn=torch.hann_window,
            center=True,
            pad_mode="reflect",
            power=1.0,
            mel_scale="slaney",
            norm="slaney",
        ).to(device=waveform.device, dtype=waveform.dtype)

        mel = mel_transform(waveform)  # [2, n_mels, time]
        mel = torch.log(torch.clamp(mel, min=1e-5))
        mel = mel.unsqueeze(0)  # [1, 2, n_mels, time]
        mel = mel.permute(0, 1, 3, 2).contiguous()  # [1, 2, time, n_mels]
        return mel

    def prepare_audio_latents_with_conditioning(
        self,
        audio_conditions: list[LTX2AudioCondition] | None,
        batch_size: int,
        num_channels_latents: int,
        audio_latent_length: int,
        num_mel_bins: int,
        audio_noise_scale: float,
        dtype: torch.dtype | None,
        device: torch.device | None,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build audio latents together with a per-token conditioning mask and clean latents.

        The returned tensors are packed to `[B, audio_seq_len, C * latent_mel_bins]`, matching the
        shape that `self._pack_audio_latents` produces on the audio stream.

        Args:
            audio_conditions: List of `LTX2AudioCondition`. Empty / None → pure noise + zero mask +
                zero clean (equivalent to the inherited `prepare_audio_latents` behavior).
            audio_latent_length: Length of the audio latent in time (before packing).
            audio_noise_scale: Initial noise level applied where the mask is zero.

        Returns:
            `(audio_latents, audio_conditioning_mask, audio_clean_latents)`.
        """
        latent_mel_bins = num_mel_bins // self.audio_vae_mel_compression_ratio
        shape_4d = (batch_size, num_channels_latents, audio_latent_length, latent_mel_bins)

        noise = randn_tensor(shape_4d, generator=generator, device=device, dtype=dtype)

        clean_4d = torch.zeros(shape_4d, device=device, dtype=dtype)
        mask_4d = torch.zeros((batch_size, 1, audio_latent_length, latent_mel_bins), device=device, dtype=dtype)

        if audio_conditions:
            duration_s = audio_latent_length / float(self.audio_latents_per_second)
            sample_rate = self.audio_vae.config.sample_rate
            target_samples = round(duration_s * sample_rate)
            for cond in audio_conditions:
                end_time = cond.end_time if cond.end_time is not None else duration_s
                start_idx = max(0, round(cond.start_time * self.audio_latents_per_second))
                end_idx = min(audio_latent_length, round(end_time * self.audio_latents_per_second))
                if end_idx <= start_idx:
                    continue

                # Zero-pad the waveform to the full target duration so the VAE sees in-distribution
                # silence context on either side of the clip. Without this, the VAE's last latent
                # token at the clip boundary encodes "audio terminates abruptly here" — a click /
                # discontinuity when decoded.
                wave = cond.audio.to(device=device)
                if wave.ndim != 2:
                    raise ValueError(
                        f"`LTX2AudioCondition.audio` must be (channels, samples); got {tuple(wave.shape)}."
                    )
                leading_samples = max(0, round(cond.start_time * sample_rate))
                max_user_samples = max(0, target_samples - leading_samples)
                wave = wave[:, :max_user_samples]
                padded = torch.zeros((wave.size(0), target_samples), device=device, dtype=wave.dtype)
                padded[:, leading_samples : leading_samples + wave.size(1)] = wave

                mel = self._waveform_to_mel_spectrogram(padded)
                mel = mel.to(dtype=self.audio_vae.dtype)
                cond_latent = retrieve_latents(self.audio_vae.encode(mel), generator=generator, sample_mode="argmax")
                cond_latent = cond_latent.to(device=device, dtype=dtype)
                # cond_latent shape: [1, C, L_full, latent_mel_bins] — full timeline, NOT yet normalized.

                available = min(end_idx - start_idx, cond_latent.size(2) - start_idx)
                if available <= 0:
                    continue

                # Broadcast batch dim: conditioning is applied identically across the batch.
                # Only the original-audio window is locked by the mask; the surrounding VAE-encoded
                # silence outside that window is harmless because mask=0 there (clean is unused).
                clean_4d[:, :, start_idx : start_idx + available, :] = cond_latent[
                    :, :, start_idx : start_idx + available, :
                ]
                mask_4d[:, :, start_idx : start_idx + available, :] = cond.strength

        # Pack everything the same way as `prepare_audio_latents` does, THEN normalize.
        # `_normalize_audio_latents` expects packed 3D `[B, L, C*M]`; the stored stats are shaped for that.
        clean_latents = self._pack_audio_latents(clean_4d)
        if audio_conditions:
            clean_latents = self._normalize_audio_latents(
                clean_latents, self.audio_vae.latents_mean, self.audio_vae.latents_std
            )
        conditioning_mask = self._pack_audio_latents(mask_4d)  # [B, L, latent_mel_bins]
        # Reduce the mask to per-token scalars (all latent_mel_bins entries are identical).
        conditioning_mask = conditioning_mask[..., :1]

        noise_packed = self._pack_audio_latents(noise)
        # Blend: locked tokens take clean_latents; free tokens get noise scaled by audio_noise_scale
        # (matching the video-side formula at `prepare_latents`).
        scaled_mask = (1.0 - conditioning_mask) * audio_noise_scale
        latents = noise_packed * scaled_mask + clean_latents * (1 - scaled_mask)
        # Where the mask is zero and audio_noise_scale is zero, use pure noise (reproduce parent behavior,
        # and ignore any `-mean/std` residue from normalizing zeros outside the conditioned region).
        no_cond_region = conditioning_mask == 0
        if audio_noise_scale == 0.0:
            latents = torch.where(no_cond_region, noise_packed, latents)

        return latents, conditioning_mask, clean_latents

    def prepare_control_latents(
        self,
        control_video: PipelineImageInput | list[PIL.Image.Image] | torch.Tensor | None,
        control_video_latents: torch.Tensor | None,
        height: int,
        width: int,
        num_frames: int,
        control_downscale_factor: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None,
        frame_rate: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build IC-LoRA reference tokens and their RoPE coordinates.

        The reference video is optionally lower resolution than the target (`control_downscale_factor > 1`).
        Returns packed reference tokens `[B, N_ref, C]` and `video_coords`-shaped positions
        `[B, 3, N_ref, 2]` with spatial axes scaled by `control_downscale_factor` to map into the
        target coordinate space.
        """
        if control_video is None and control_video_latents is None:
            return None, None

        ref_height = height // control_downscale_factor
        ref_width = width // control_downscale_factor

        if control_video_latents is not None:
            ref_latent = control_video_latents.to(device=device, dtype=dtype)
            if ref_latent.ndim != 5:
                raise ValueError(
                    "`control_video_latents` must be an unpacked 5D tensor of shape "
                    f"[B, C, F, H, W], got shape {tuple(ref_latent.shape)}."
                )
            ref_latent = self._normalize_latents(
                ref_latent, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
            )
        else:
            control_pixels = self.video_processor.preprocess_video(
                control_video, ref_height, ref_width, resize_mode="crop"
            )
            # Pad or truncate to `num_frames` on the temporal dim (dim=2 of [B, C, F, H, W])
            cur_frames = control_pixels.size(2)
            if cur_frames < num_frames:
                pad = torch.zeros(
                    control_pixels.size(0),
                    control_pixels.size(1),
                    num_frames - cur_frames,
                    control_pixels.size(3),
                    control_pixels.size(4),
                    dtype=control_pixels.dtype,
                )
                control_pixels = torch.cat([control_pixels, pad], dim=2)
                logger.warning(
                    f"control_video has {cur_frames} frames; padding with zeros to match num_frames={num_frames}."
                )
            elif cur_frames > num_frames:
                logger.warning(f"control_video has {cur_frames} frames; truncating to num_frames={num_frames}.")
                control_pixels = control_pixels[:, :, :num_frames]

            control_pixels = control_pixels.to(dtype=self.vae.dtype, device=device)
            ref_latent = retrieve_latents(self.vae.encode(control_pixels), generator=generator, sample_mode="argmax")
            # Cast to float32 BEFORE normalizing — matches frame-artisan (`encoded.float()` before
            # `normalize_latents`). Normalizing in VAE dtype (bfloat16) loses precision on near-zero
            # latent values, which matters for the outpaint IC-LoRA's pure-black sentinel.
            ref_latent = ref_latent.float()
            ref_latent = self._normalize_latents(ref_latent, self.vae.latents_mean, self.vae.latents_std).to(
                device=device, dtype=dtype
            )

        _, _, f_ref, h_ref, w_ref = ref_latent.shape

        # Broadcast reference latent to the requested batch size (reference is the same across batch)
        if ref_latent.size(0) == 1 and batch_size > 1:
            ref_latent = ref_latent.expand(batch_size, -1, -1, -1, -1)
        elif ref_latent.size(0) != batch_size:
            raise ValueError(
                f"control_video batch size {ref_latent.size(0)} must be 1 or match batch_size={batch_size}."
            )

        control_tokens = self._pack_latents(
            ref_latent, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
        )

        # Build RoPE positions on the reference grid, then scale spatial axes into target space
        control_coords = self.transformer.rope.prepare_video_coords(
            batch_size, f_ref, h_ref, w_ref, device, fps=frame_rate
        ).to(dtype=torch.float32)
        if control_downscale_factor != 1:
            control_coords[:, 1, ...] *= control_downscale_factor  # height
            control_coords[:, 2, ...] *= control_downscale_factor  # width

        return control_tokens, control_coords

    def prepare_control_audio_latents(
        self,
        control_audio: torch.Tensor | None,
        control_audio_latents: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build audio IC-LoRA reference tokens and their RoPE coords shifted to negative time.

        Mirrors the video IC-LoRA pattern in `prepare_control_latents`, but on the audio stream:
        a raw waveform (or pre-encoded latents) is encoded by the audio VAE, packed, and given
        audio-RoPE coords shifted to `t ∈ [-T_ref, 0)` so the reference sits before the target on
        the audio time axis. Caller PREPENDS the returned tokens to `audio_latents` (and matching
        clean / mask / coords) and STRIPS them again before audio decode.

        `control_audio_latents` may be passed either packed `[B, L_ref, C*M]` (no extra processing)
        or unpacked `[B, C, L_ref, M]` (auto packed + normalized) — same convention as
        `prepare_audio_latents`.

        Returns `(packed_ref_latents [B, L_ref, C*M], ref_coords [B, 1, L_ref, 2])`. Both `None`
        when neither input is given.
        """
        if control_audio is None and control_audio_latents is None:
            return None, None

        if control_audio_latents is not None:
            ref_latents = control_audio_latents.to(device=device, dtype=dtype)
            if ref_latents.ndim == 4:
                ref_latents = self._pack_audio_latents(ref_latents)
                ref_latents = self._normalize_audio_latents(
                    ref_latents, self.audio_vae.latents_mean, self.audio_vae.latents_std
                )
            elif ref_latents.ndim != 3:
                raise ValueError(
                    "`control_audio_latents` must be 3D packed `[B, L, C*M]` (assumed already "
                    "normalized) or 4D unpacked `[B, C, L, M]`, got shape "
                    f"{tuple(ref_latents.shape)}."
                )
        else:
            mel = self._waveform_to_mel_spectrogram(control_audio.to(device=device))
            mel = mel.to(dtype=self.audio_vae.dtype)
            ref_latents_4d = retrieve_latents(self.audio_vae.encode(mel), generator=generator, sample_mode="argmax")
            ref_latents_4d = ref_latents_4d.to(device=device, dtype=dtype)
            ref_latents = self._pack_audio_latents(ref_latents_4d)
            ref_latents = self._normalize_audio_latents(
                ref_latents, self.audio_vae.latents_mean, self.audio_vae.latents_std
            )

        if ref_latents.size(0) == 1 and batch_size > 1:
            ref_latents = ref_latents.expand(batch_size, -1, -1)
        elif ref_latents.size(0) != batch_size:
            raise ValueError(
                f"control_audio batch size {ref_latents.size(0)} must be 1 or match batch_size={batch_size}."
            )

        ref_seq_len = ref_latents.shape[1]

        # Build audio coords on a "positive" grid for the ref length, then shift the whole grid
        # left by (ref_duration + one_latent_step) so the ref sits in `[-T_ref, 0)` and the target
        # (which still uses its own `[0, T_target)` grid) is placed strictly after.
        ref_coords = self.transformer.audio_rope.prepare_audio_coords(batch_size, ref_seq_len, device).to(
            dtype=torch.float32
        )
        time_per_latent = 1.0 / float(self.audio_latents_per_second)
        ref_duration = ref_coords[..., -1, 1].max().item()
        ref_coords = ref_coords - ref_duration - time_per_latent

        return ref_latents, ref_coords

    def _split_image_conditions(
        self,
        image_conditions: list[LTX2ImageCondition] | None,
        num_frames: int,
        latent_num_frames: int,
    ) -> tuple[list[LTX2VideoCondition], list[tuple[Any, int, float]]]:
        """
        Snap each pixel `frame` to its containing latent slot, then route:
        - latent_idx == 0 → "replace" path via LTX2VideoCondition (hard constraint at frame 0).
        - latent_idx > 0 → "keyframe" path (sequence-concat with target-space RoPE).

        Returns `(replace_conditions, [(image, latent_idx, strength), ...])`. The keyframe entries
        carry the resolved latent index so `prepare_keyframe_latents` doesn't re-snap.
        """
        replace_list: list[LTX2VideoCondition] = []
        keyframe_list: list[tuple[Any, int, float]] = []
        if not image_conditions:
            return replace_list, keyframe_list

        vae_t = int(self.vae_temporal_compression_ratio)
        for cond in image_conditions:
            frame = cond.frame
            if frame < 0:
                frame = num_frames + frame
            if frame < 0 or frame >= num_frames:
                raise ValueError(f"LTX2ImageCondition.frame {cond.frame} is out of range for num_frames={num_frames}.")
            # Snap pixel frame → latent slot. ceil(frame / vae_t) gives:
            #   frame=0 → latent 0 (causal slot), frame 1..vae_t → latent 1, ..vae_t+1..2*vae_t → latent 2, ...
            latent_idx = (frame + vae_t - 1) // vae_t
            latent_idx = min(latent_idx, latent_num_frames - 1)
            anchored_pixel = 0 if latent_idx == 0 else (latent_idx - 1) * vae_t + 1
            if anchored_pixel != frame:
                logger.info(
                    f"LTX2ImageCondition: frame={cond.frame} snapped to pixel {anchored_pixel} "
                    f"(latent slot {latent_idx})."
                )
            if latent_idx == 0:
                replace_list.append(LTX2VideoCondition(frames=cond.image, index=0, strength=cond.strength))
            else:
                keyframe_list.append((cond.image, latent_idx, cond.strength))
        return replace_list, keyframe_list

    def prepare_keyframe_latents(
        self,
        keyframe_conditions: list[tuple[Any, int, float]],
        height: int,
        width: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None,
        frame_rate: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build keyframe tokens + their RoPE coords + per-token strength mask.

        Each keyframe image is VAE-encoded (as a 1-frame video), packed, and given RoPE coords
        anchored at the resolved latent slot — so the model treats it as a reference at that
        position. Returns packed tokens `[B, N_kf, C]`, coords `[B, 3, N_kf, 2]`, and strengths
        `[B, N_kf, 1]`.

        `keyframe_conditions` items are `(image, latent_idx, strength)` tuples produced by
        `_split_image_conditions` (pixel→latent snap already applied).
        """
        if not keyframe_conditions:
            return None, None, None

        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio

        all_tokens = []
        all_coords = []
        all_strengths = []
        for image, latent_idx, strength in keyframe_conditions:
            # Preprocess → encode a single-frame "video"
            image_pixels = self.video_processor.preprocess_video(image, height, width, resize_mode="crop")
            # ensure exactly one temporal frame (image inputs already yield F=1)
            image_pixels = image_pixels[:, :, :1]
            image_pixels = image_pixels.to(dtype=self.vae.dtype, device=device)
            image_latent = retrieve_latents(self.vae.encode(image_pixels), generator=generator, sample_mode="argmax")
            # float32 before normalize — matches frame-artisan and preserves precision near zero.
            image_latent = image_latent.float()
            image_latent = self._normalize_latents(image_latent, self.vae.latents_mean, self.vae.latents_std).to(
                device=device, dtype=dtype
            )  # [1, C, 1, lh, lw]

            if image_latent.size(0) == 1 and batch_size > 1:
                image_latent = image_latent.expand(batch_size, -1, -1, -1, -1)

            tokens = self._pack_latents(
                image_latent, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
            )  # [B, lh*lw, C]

            # Build coords for 1 latent frame placed at target latent slot `latent_idx`.
            # `prepare_video_coords` is called with num_frames=1 only to get the spatial axes;
            # the temporal axis is overridden below to anchor the keyframe at the correct pixel
            # frame via the VAE temporal scale + causal offset (otherwise a naive offset would
            # land at pixel N instead of ~N*8).
            coords = self.transformer.rope.prepare_video_coords(
                batch_size, 1, latent_height, latent_width, device, fps=frame_rate
            ).to(dtype=torch.float32)
            # coords shape: [B, 3, N_patches, 2]; axis 1 = (frame, height, width); last axis = (start, end).
            # Temporal mapping mirrors `prepare_video_coords`:
            #   pixel_start = (latent_idx * vae_t + causal_offset - vae_t).clamp(0)
            # For a single-image keyframe (num_pixel_frames == 1) the end is narrowed to
            # `start + 1 pixel frame` so the RoPE midpoint lands on the anchored pixel frame
            # instead of the VAE block centre. Matches upstream LTX-2 keyframe_cond.py.
            vae_t = self.transformer.rope.scale_factors[0]
            causal_offset = self.transformer.rope.causal_offset
            pixel_t_start = max(0.0, float(latent_idx) * vae_t + causal_offset - vae_t)
            coords[:, 0, ..., 0] = pixel_t_start / frame_rate
            coords[:, 0, ..., 1] = (pixel_t_start + 1.0) / frame_rate

            strength_mask = torch.full(
                (batch_size, tokens.shape[1], 1), fill_value=strength, device=device, dtype=dtype
            )

            all_tokens.append(tokens)
            all_coords.append(coords)
            all_strengths.append(strength_mask)

        keyframe_tokens = torch.cat(all_tokens, dim=1)
        keyframe_coords = torch.cat(all_coords, dim=2)
        keyframe_strengths = torch.cat(all_strengths, dim=1)
        return keyframe_tokens, keyframe_coords, keyframe_strengths

    @property
    def audio_latents_per_second(self) -> float:
        return self.audio_sampling_rate / self.audio_hop_length / float(self.audio_vae_temporal_compression_ratio)

    def _assert_ic_lora_loaded(self) -> None:
        try:
            active = self.get_active_adapters()
        except Exception:
            active = []
        if not active:
            raise ValueError(
                "control_video / control_video_latents requires an IC-LoRA to be loaded. "
                "Call `pipe.load_lora_weights(...)` with an LTX-2.3 IC-LoRA checkpoint first."
            )

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_on_step_end_tensor_inputs=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
        latents=None,
        audio_latents=None,
        spatio_temporal_guidance_blocks=None,
        stg_scale=None,
        audio_stg_scale=None,
        # New — validated here
        audio_conditions=None,
        image_conditions=None,
        control_video=None,
        control_video_latents=None,
        control_downscale_factor=1,
        control_audio=None,
        control_audio_latents=None,
        control_audio_strength=1.0,
        num_frames=None,
        frame_rate=None,
    ):
        super().check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            latents=latents,
            audio_latents=audio_latents,
            spatio_temporal_guidance_blocks=spatio_temporal_guidance_blocks,
            stg_scale=stg_scale,
            audio_stg_scale=audio_stg_scale,
        )

        if audio_conditions is not None:
            if not isinstance(audio_conditions, list):
                audio_conditions = [audio_conditions]
            duration_s = num_frames / frame_rate if (num_frames and frame_rate) else None
            for i, cond in enumerate(audio_conditions):
                if not isinstance(cond, LTX2AudioCondition):
                    raise ValueError(f"audio_conditions[{i}] must be an LTX2AudioCondition, got {type(cond)}.")
                if cond.audio.ndim != 2:
                    raise ValueError(
                        f"audio_conditions[{i}].audio must be 2D (channels, samples); got shape "
                        f"{tuple(cond.audio.shape)}."
                    )
                end_time = cond.end_time if cond.end_time is not None else duration_s
                if cond.start_time < 0:
                    raise ValueError(f"audio_conditions[{i}].start_time must be >= 0.")
                if end_time is not None and end_time <= cond.start_time:
                    raise ValueError(
                        f"audio_conditions[{i}]: end_time ({end_time}) must be > start_time "
                        f"({cond.start_time}). start_time/end_time refer to the OUTPUT timeline."
                    )
                if duration_s is not None and end_time is not None and end_time > duration_s + 1e-6:
                    raise ValueError(
                        f"audio_conditions[{i}].end_time={end_time} exceeds output duration "
                        f"{duration_s:.3f}s (= num_frames/frame_rate)."
                    )

        if control_video is not None or control_video_latents is not None:
            if control_video is not None and control_video_latents is not None:
                raise ValueError("Pass only one of `control_video` or `control_video_latents`, not both.")
            if not isinstance(control_downscale_factor, int) or control_downscale_factor < 1:
                raise ValueError(f"control_downscale_factor must be a positive int, got {control_downscale_factor}.")
            self._assert_ic_lora_loaded()

        if control_audio is not None or control_audio_latents is not None:
            if control_audio is not None and control_audio_latents is not None:
                raise ValueError("Pass only one of `control_audio` or `control_audio_latents`, not both.")
            if control_audio is not None and control_audio.ndim != 2:
                raise ValueError(
                    f"control_audio must be 2D (channels, samples); got shape {tuple(control_audio.shape)}."
                )
            if not (0.0 <= float(control_audio_strength) <= 1.0):
                raise ValueError(f"control_audio_strength must be in [0, 1], got {control_audio_strength}.")
            self._assert_ic_lora_loaded()

        if image_conditions is not None:
            if not isinstance(image_conditions, list):
                image_conditions = [image_conditions]
            for i, cond in enumerate(image_conditions):
                if not isinstance(cond, LTX2ImageCondition):
                    raise ValueError(f"image_conditions[{i}] must be an LTX2ImageCondition, got {type(cond)}.")
                if not (0.0 <= cond.strength <= 1.0):
                    raise ValueError(f"image_conditions[{i}].strength must be in [0, 1], got {cond.strength}.")

    @torch.no_grad()
    def __call__(
        self,
        video_conditions: LTX2VideoCondition | list[LTX2VideoCondition] | None = None,
        prompt: str | list[str] = None,
        negative_prompt: str | list[str] | None = None,
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        frame_rate: float = 24.0,
        num_inference_steps: int = 40,
        sigmas: list[float] | None = None,
        timesteps: list[float] | None = None,
        guidance_scale: float = 4.0,
        stg_scale: float = 0.0,
        modality_scale: float = 1.0,
        guidance_rescale: float = 0.0,
        audio_guidance_scale: float | None = None,
        audio_stg_scale: float | None = None,
        audio_modality_scale: float | None = None,
        audio_guidance_rescale: float | None = None,
        spatio_temporal_guidance_blocks: list[int] | None = None,
        noise_scale: float | None = None,
        audio_noise_scale: float = 0.0,
        num_videos_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        audio_latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        decode_timestep: float | list[float] = 0.0,
        decode_noise_scale: float | list[float] | None = None,
        use_cross_timestep: bool = False,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 1024,
        # New parameters
        audio_conditions: LTX2AudioCondition | list[LTX2AudioCondition] | None = None,
        image_conditions: LTX2ImageCondition | list[LTX2ImageCondition] | None = None,
        control_video: PipelineImageInput | list[PIL.Image.Image] | torch.Tensor | None = None,
        control_video_latents: torch.Tensor | None = None,
        control_downscale_factor: int = 1,
        control_strength: float = 1.0,
        control_audio: torch.Tensor | None = None,
        control_audio_latents: torch.Tensor | None = None,
        control_audio_strength: float = 1.0,
        identity_guidance_scale: float = 0.0,
    ):
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        audio_guidance_scale = audio_guidance_scale or guidance_scale
        audio_stg_scale = audio_stg_scale or stg_scale
        audio_modality_scale = audio_modality_scale or modality_scale
        audio_guidance_rescale = audio_guidance_rescale or guidance_rescale

        # 1. Check inputs.
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            latents=latents,
            audio_latents=audio_latents,
            spatio_temporal_guidance_blocks=spatio_temporal_guidance_blocks,
            stg_scale=stg_scale,
            audio_stg_scale=audio_stg_scale,
            audio_conditions=audio_conditions,
            image_conditions=image_conditions,
            control_video=control_video,
            control_video_latents=control_video_latents,
            control_downscale_factor=control_downscale_factor,
            control_audio=control_audio,
            control_audio_latents=control_audio_latents,
            control_audio_strength=control_audio_strength,
            num_frames=num_frames,
            frame_rate=frame_rate,
        )

        self._guidance_scale = guidance_scale
        self._stg_scale = stg_scale
        self._modality_scale = modality_scale
        self._guidance_rescale = guidance_rescale
        self._audio_guidance_scale = audio_guidance_scale
        self._audio_stg_scale = audio_stg_scale
        self._audio_modality_scale = audio_modality_scale
        self._audio_guidance_rescale = audio_guidance_rescale

        self._attention_kwargs = attention_kwargs
        self._interrupt = False
        self._current_timestep = None

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if video_conditions is not None and not isinstance(video_conditions, list):
            video_conditions = [video_conditions]
        if audio_conditions is not None and not isinstance(audio_conditions, list):
            audio_conditions = [audio_conditions]
        if image_conditions is not None and not isinstance(image_conditions, list):
            image_conditions = [image_conditions]

        if noise_scale is None:
            noise_scale = sigmas[0] if sigmas is not None else 1.0

        device = self._execution_device
        effective_batch = batch_size * num_videos_per_prompt

        # 3. Prepare text embeddings
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        tokenizer_padding_side = "left"
        if getattr(self, "tokenizer", None) is not None:
            tokenizer_padding_side = getattr(self.tokenizer, "padding_side", "left")
        connector_prompt_embeds, connector_audio_prompt_embeds, connector_attention_mask = self.connectors(
            prompt_embeds, prompt_attention_mask, padding_side=tokenizer_padding_side
        )

        # 4. Prepare video latents (same as parent)
        latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio
        if latents is not None:
            logger.info(
                "Got latents of shape [B, C, F, H, W]; `latent_num_frames`, `latent_height`, `latent_width` will be inferred."
            )
            _, _, latent_num_frames, latent_height, latent_width = latents.shape

        # Route image_conditions: frame-0 → replace (merge into video_conditions); others → keyframes
        replace_img_conds, keyframe_img_conds = self._split_image_conditions(
            image_conditions, num_frames, latent_num_frames
        )
        merged_replace_conditions = (video_conditions or []) + replace_img_conds

        num_channels_latents = self.transformer.config.in_channels
        latents, conditioning_mask, clean_latents, _ = self.prepare_latents(
            merged_replace_conditions,
            effective_batch,
            num_channels_latents,
            height,
            width,
            num_frames,
            frame_rate,
            noise_scale,
            torch.float32,
            device,
            generator,
            latents,
        )

        target_seq_len = latents.shape[1]  # tokens before any keyframe / IC-LoRA concat

        # 4a. Audio latents with conditioning
        duration_s = num_frames / frame_rate
        audio_num_frames = round(duration_s * self.audio_latents_per_second)
        if audio_latents is not None:
            logger.info("Got audio_latents of shape [B, C, L, M]; `audio_num_frames` will be inferred.")
            _, _, audio_num_frames, _ = audio_latents.shape

        num_mel_bins = self.audio_vae.config.mel_bins if getattr(self, "audio_vae", None) is not None else 64
        latent_mel_bins = num_mel_bins // self.audio_vae_mel_compression_ratio
        num_channels_latents_audio = (
            self.audio_vae.config.latent_channels if getattr(self, "audio_vae", None) is not None else 8
        )

        if audio_latents is not None:
            # User-provided pre-encoded latents: fall back to parent semantics (no audio conditioning mask)
            audio_latents = self.prepare_audio_latents(
                effective_batch,
                num_channels_latents=num_channels_latents_audio,
                audio_latent_length=audio_num_frames,
                num_mel_bins=num_mel_bins,
                noise_scale=noise_scale,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=audio_latents,
            )
            # Empty mask/clean — no audio conditioning applied.
            audio_conditioning_mask = torch.zeros(audio_latents.shape[:2] + (1,), device=device, dtype=torch.float32)
            audio_clean_latents = torch.zeros_like(audio_latents)
        else:
            (
                audio_latents,
                audio_conditioning_mask,
                audio_clean_latents,
            ) = self.prepare_audio_latents_with_conditioning(
                audio_conditions,
                batch_size=effective_batch,
                num_channels_latents=num_channels_latents_audio,
                audio_latent_length=audio_num_frames,
                num_mel_bins=num_mel_bins,
                audio_noise_scale=audio_noise_scale,
                dtype=torch.float32,
                device=device,
                generator=generator,
            )

        # 5. Prepare timesteps (two schedulers — video + a copy for audio)
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        mu = calculate_shift(
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_image_seq_len", 1024),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.95),
            self.scheduler.config.get("max_shift", 2.05),
        )

        audio_scheduler = copy.deepcopy(self.scheduler)
        _, _ = retrieve_timesteps(
            audio_scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas=sigmas,
            mu=mu,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 6. Positional ids
        video_coords = self.transformer.rope.prepare_video_coords(
            latents.shape[0], latent_num_frames, latent_height, latent_width, latents.device, fps=frame_rate
        )
        audio_coords = self.transformer.audio_rope.prepare_audio_coords(
            audio_latents.shape[0], audio_num_frames, audio_latents.device
        )

        # 6a. Keyframe image tokens (sequence-dim concat — for image_conditions with index != 0)
        keyframe_tokens, keyframe_coords, keyframe_strengths = self.prepare_keyframe_latents(
            keyframe_conditions=keyframe_img_conds,
            height=height,
            width=width,
            batch_size=effective_batch,
            device=device,
            dtype=torch.float32,
            generator=generator,
            frame_rate=frame_rate,
        )
        if keyframe_tokens is not None:
            latents = torch.cat([latents, keyframe_tokens * keyframe_strengths], dim=1)
            clean_latents = torch.cat([clean_latents, keyframe_tokens], dim=1)
            conditioning_mask = torch.cat([conditioning_mask, keyframe_strengths], dim=1)
            video_coords = torch.cat([video_coords, keyframe_coords], dim=2)

        # 6b. IC-LoRA reference-token concat (sequence-dim)
        control_tokens, control_coords = self.prepare_control_latents(
            control_video=control_video,
            control_video_latents=control_video_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            control_downscale_factor=control_downscale_factor,
            batch_size=effective_batch,
            device=device,
            dtype=torch.float32,
            generator=generator,
            frame_rate=frame_rate,
        )
        if control_tokens is not None:
            n_ref = control_tokens.shape[1]
            ref_mask = torch.full(
                (effective_batch, n_ref, 1),
                fill_value=control_strength,
                device=device,
                dtype=conditioning_mask.dtype,
            )
            # Concat raw reference tokens — strength is applied via the denoise mask, not by scaling
            # the latent values (matches LTX-2 reference in reference_video_cond.py:85-90).
            latents = torch.cat([latents, control_tokens], dim=1)
            clean_latents = torch.cat([clean_latents, control_tokens], dim=1)
            conditioning_mask = torch.cat([conditioning_mask, ref_mask], dim=1)
            # video_coords shape: [B, 3, N_tokens, 2] — concat on num_patches (dim=2)
            video_coords = torch.cat([video_coords, control_coords], dim=2)

        # 6c. Audio IC-LoRA reference-token PREPEND (sequence-dim, with negative-time RoPE)
        control_audio_tokens, control_audio_coords = self.prepare_control_audio_latents(
            control_audio=control_audio,
            control_audio_latents=control_audio_latents,
            batch_size=effective_batch,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        audio_ref_num_tokens = 0
        if control_audio_tokens is not None:
            audio_ref_num_tokens = control_audio_tokens.shape[1]
            ref_audio_mask = torch.full(
                (effective_batch, audio_ref_num_tokens, 1),
                fill_value=control_audio_strength,
                device=device,
                dtype=audio_conditioning_mask.dtype,
            )
            # Prepend (NOT append): ref tokens precede the target on the audio time axis. The
            # negative-time coords on `control_audio_coords` keep the target's `[0, T)` grid intact.
            audio_latents = torch.cat([control_audio_tokens, audio_latents], dim=1)
            audio_clean_latents = torch.cat([control_audio_tokens, audio_clean_latents], dim=1)
            audio_conditioning_mask = torch.cat([ref_audio_mask, audio_conditioning_mask], dim=1)
            # audio_coords shape: [B, 1, N_tokens, 2] — concat on num_patches (dim=2)
            audio_coords = torch.cat([control_audio_coords, audio_coords], dim=2)

        # Identity guidance is opt-in (default 0.0 = off) and only active when audio ref tokens
        # are present. ID-LoRA-style audio LoRAs benefit; vanilla audio LoRAs leave it at 0.
        identity_guidance_scale = float(identity_guidance_scale)
        do_identity_guidance = identity_guidance_scale > 0.0 and audio_ref_num_tokens > 0

        # 6d. CFG duplication (matches parent's line 1288 and lines 1358-1360)
        if self.do_classifier_free_guidance:
            conditioning_mask = torch.cat([conditioning_mask, conditioning_mask])
            audio_conditioning_mask = torch.cat([audio_conditioning_mask, audio_conditioning_mask])
            video_coords = video_coords.repeat((2,) + (1,) * (video_coords.ndim - 1))
            audio_coords = audio_coords.repeat((2,) + (1,) * (audio_coords.ndim - 1))
            # Note: clean_latents and audio_clean_latents are NOT duplicated — the blending
            # step slices the mask to [:bsz] to match them.

        # 7. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = latent_model_input.to(prompt_embeds.dtype)
                audio_latent_model_input = (
                    torch.cat([audio_latents] * 2) if self.do_classifier_free_guidance else audio_latents
                )
                audio_latent_model_input = audio_latent_model_input.to(prompt_embeds.dtype)

                timestep = t.expand(latent_model_input.shape[0])
                video_timestep = timestep.unsqueeze(-1) * (1 - conditioning_mask.squeeze(-1))
                audio_timestep = timestep.unsqueeze(-1) * (1 - audio_conditioning_mask.squeeze(-1))

                with self.transformer.cache_context("cond_uncond"):
                    noise_pred_video, noise_pred_audio = self.transformer(
                        hidden_states=latent_model_input,
                        audio_hidden_states=audio_latent_model_input,
                        encoder_hidden_states=connector_prompt_embeds,
                        audio_encoder_hidden_states=connector_audio_prompt_embeds,
                        timestep=video_timestep,
                        audio_timestep=audio_timestep,
                        sigma=timestep,
                        encoder_attention_mask=connector_attention_mask,
                        audio_encoder_attention_mask=connector_attention_mask,
                        num_frames=latent_num_frames,
                        height=latent_height,
                        width=latent_width,
                        fps=frame_rate,
                        audio_num_frames=audio_num_frames,
                        video_coords=video_coords,
                        audio_coords=audio_coords,
                        isolate_modalities=False,
                        spatio_temporal_guidance_blocks=None,
                        perturbation_mask=None,
                        use_cross_timestep=use_cross_timestep,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )
                noise_pred_video = noise_pred_video.float()
                noise_pred_audio = noise_pred_audio.float()

                if self.do_classifier_free_guidance:
                    noise_pred_video_uncond_text, noise_pred_video = noise_pred_video.chunk(2)
                    noise_pred_video = self.convert_velocity_to_x0(latents, noise_pred_video, i, self.scheduler)
                    noise_pred_video_uncond_text = self.convert_velocity_to_x0(
                        latents, noise_pred_video_uncond_text, i, self.scheduler
                    )
                    video_cfg_delta = (self.guidance_scale - 1) * (noise_pred_video - noise_pred_video_uncond_text)

                    noise_pred_audio_uncond_text, noise_pred_audio = noise_pred_audio.chunk(2)
                    noise_pred_audio = self.convert_velocity_to_x0(audio_latents, noise_pred_audio, i, audio_scheduler)
                    noise_pred_audio_uncond_text = self.convert_velocity_to_x0(
                        audio_latents, noise_pred_audio_uncond_text, i, audio_scheduler
                    )
                    audio_cfg_delta = (self.audio_guidance_scale - 1) * (
                        noise_pred_audio - noise_pred_audio_uncond_text
                    )

                    if self.do_spatio_temporal_guidance or self.do_modality_isolation_guidance or do_identity_guidance:
                        if i == 0:
                            video_prompt_embeds = connector_prompt_embeds.chunk(2, dim=0)[1]
                            audio_prompt_embeds = connector_audio_prompt_embeds.chunk(2, dim=0)[1]
                            prompt_attn_mask = connector_attention_mask.chunk(2, dim=0)[1]

                            video_pos_ids = video_coords.chunk(2, dim=0)[0]
                            audio_pos_ids = audio_coords.chunk(2, dim=0)[0]

                        timestep = timestep.chunk(2, dim=0)[0]
                        video_timestep = video_timestep.chunk(2, dim=0)[0]
                        audio_timestep = audio_timestep.chunk(2, dim=0)[0]
                else:
                    video_cfg_delta = audio_cfg_delta = 0

                    video_prompt_embeds = connector_prompt_embeds
                    audio_prompt_embeds = connector_audio_prompt_embeds
                    prompt_attn_mask = connector_attention_mask

                    video_pos_ids = video_coords
                    audio_pos_ids = audio_coords

                    noise_pred_video = self.convert_velocity_to_x0(latents, noise_pred_video, i, self.scheduler)
                    noise_pred_audio = self.convert_velocity_to_x0(audio_latents, noise_pred_audio, i, audio_scheduler)

                if self.do_spatio_temporal_guidance:
                    with self.transformer.cache_context("uncond_stg"):
                        noise_pred_video_uncond_stg, noise_pred_audio_uncond_stg = self.transformer(
                            hidden_states=latents.to(dtype=prompt_embeds.dtype),
                            audio_hidden_states=audio_latents.to(dtype=prompt_embeds.dtype),
                            encoder_hidden_states=video_prompt_embeds,
                            audio_encoder_hidden_states=audio_prompt_embeds,
                            timestep=video_timestep,
                            audio_timestep=audio_timestep,
                            sigma=timestep,
                            encoder_attention_mask=prompt_attn_mask,
                            audio_encoder_attention_mask=prompt_attn_mask,
                            num_frames=latent_num_frames,
                            height=latent_height,
                            width=latent_width,
                            fps=frame_rate,
                            audio_num_frames=audio_num_frames,
                            video_coords=video_pos_ids,
                            audio_coords=audio_pos_ids,
                            isolate_modalities=False,
                            spatio_temporal_guidance_blocks=spatio_temporal_guidance_blocks,
                            perturbation_mask=None,
                            use_cross_timestep=use_cross_timestep,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )
                    noise_pred_video_uncond_stg = noise_pred_video_uncond_stg.float()
                    noise_pred_audio_uncond_stg = noise_pred_audio_uncond_stg.float()
                    noise_pred_video_uncond_stg = self.convert_velocity_to_x0(
                        latents, noise_pred_video_uncond_stg, i, self.scheduler
                    )
                    noise_pred_audio_uncond_stg = self.convert_velocity_to_x0(
                        audio_latents, noise_pred_audio_uncond_stg, i, audio_scheduler
                    )

                    video_stg_delta = self.stg_scale * (noise_pred_video - noise_pred_video_uncond_stg)
                    audio_stg_delta = self.audio_stg_scale * (noise_pred_audio - noise_pred_audio_uncond_stg)
                else:
                    video_stg_delta = audio_stg_delta = 0

                if self.do_modality_isolation_guidance:
                    with self.transformer.cache_context("uncond_modality"):
                        (
                            noise_pred_video_uncond_modality,
                            noise_pred_audio_uncond_modality,
                        ) = self.transformer(
                            hidden_states=latents.to(dtype=prompt_embeds.dtype),
                            audio_hidden_states=audio_latents.to(dtype=prompt_embeds.dtype),
                            encoder_hidden_states=video_prompt_embeds,
                            audio_encoder_hidden_states=audio_prompt_embeds,
                            timestep=video_timestep,
                            audio_timestep=audio_timestep,
                            sigma=timestep,
                            encoder_attention_mask=prompt_attn_mask,
                            audio_encoder_attention_mask=prompt_attn_mask,
                            num_frames=latent_num_frames,
                            height=latent_height,
                            width=latent_width,
                            fps=frame_rate,
                            audio_num_frames=audio_num_frames,
                            video_coords=video_pos_ids,
                            audio_coords=audio_pos_ids,
                            isolate_modalities=True,
                            spatio_temporal_guidance_blocks=None,
                            perturbation_mask=None,
                            use_cross_timestep=use_cross_timestep,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )
                    noise_pred_video_uncond_modality = noise_pred_video_uncond_modality.float()
                    noise_pred_audio_uncond_modality = noise_pred_audio_uncond_modality.float()
                    noise_pred_video_uncond_modality = self.convert_velocity_to_x0(
                        latents, noise_pred_video_uncond_modality, i, self.scheduler
                    )
                    noise_pred_audio_uncond_modality = self.convert_velocity_to_x0(
                        audio_latents, noise_pred_audio_uncond_modality, i, audio_scheduler
                    )

                    video_modality_delta = (self.modality_scale - 1) * (
                        noise_pred_video - noise_pred_video_uncond_modality
                    )
                    audio_modality_delta = (self.audio_modality_scale - 1) * (
                        noise_pred_audio - noise_pred_audio_uncond_modality
                    )
                else:
                    video_modality_delta = audio_modality_delta = 0

                noise_pred_video_g = noise_pred_video + video_cfg_delta + video_stg_delta + video_modality_delta
                noise_pred_audio_g = noise_pred_audio + audio_cfg_delta + audio_stg_delta + audio_modality_delta

                # Identity guidance (audio IC-LoRA, opt-in): one extra forward pass on the
                # TARGET-only audio (ref tokens stripped) using the positive prompt. The delta
                # `scale * (x0_with_ref[:, ref:] - x0_noref)` is added only to the target slice
                # of `noise_pred_audio_g`, leaving ref-token predictions untouched (they're
                # locked by `audio_conditioning_mask=1.0` anyway). Mirrors ID-LoRA-2.3's
                # `inference_one_stage.py` lines 294-308, adapted to the diffusers x0 round-trip.
                if do_identity_guidance:
                    audio_target_input = audio_latents[:, audio_ref_num_tokens:, :].to(dtype=prompt_embeds.dtype)
                    audio_target_coords = audio_pos_ids[:, :, audio_ref_num_tokens:, :]
                    audio_target_timestep = audio_timestep[:, audio_ref_num_tokens:]

                    with self.transformer.cache_context("identity_guidance"):
                        _, noise_pred_audio_noref = self.transformer(
                            hidden_states=latents.to(dtype=prompt_embeds.dtype),
                            audio_hidden_states=audio_target_input,
                            encoder_hidden_states=video_prompt_embeds,
                            audio_encoder_hidden_states=audio_prompt_embeds,
                            timestep=video_timestep,
                            audio_timestep=audio_target_timestep,
                            sigma=timestep,
                            encoder_attention_mask=prompt_attn_mask,
                            audio_encoder_attention_mask=prompt_attn_mask,
                            num_frames=latent_num_frames,
                            height=latent_height,
                            width=latent_width,
                            fps=frame_rate,
                            audio_num_frames=audio_num_frames,
                            video_coords=video_pos_ids,
                            audio_coords=audio_target_coords,
                            isolate_modalities=False,
                            spatio_temporal_guidance_blocks=None,
                            perturbation_mask=None,
                            use_cross_timestep=use_cross_timestep,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )
                    noise_pred_audio_noref = noise_pred_audio_noref.float()
                    # Convert velocity → x0 against the target-only noisy latent.
                    audio_target_noisy = audio_latents[:, audio_ref_num_tokens:, :].float()
                    audio_x0_noref = self.convert_velocity_to_x0(
                        audio_target_noisy, noise_pred_audio_noref, i, audio_scheduler
                    )
                    # `noise_pred_audio` is the positive-prompt x0 WITH ref. Compare on target slice.
                    id_delta = identity_guidance_scale * (noise_pred_audio[:, audio_ref_num_tokens:] - audio_x0_noref)
                    noise_pred_audio_g = noise_pred_audio_g.clone()
                    noise_pred_audio_g[:, audio_ref_num_tokens:] = (
                        noise_pred_audio_g[:, audio_ref_num_tokens:] + id_delta
                    )

                if self.guidance_rescale > 0:
                    noise_pred_video = rescale_noise_cfg(
                        noise_pred_video_g, noise_pred_video, guidance_rescale=self.guidance_rescale
                    )
                else:
                    noise_pred_video = noise_pred_video_g

                if self.audio_guidance_rescale > 0:
                    noise_pred_audio = rescale_noise_cfg(
                        noise_pred_audio_g, noise_pred_audio, guidance_rescale=self.audio_guidance_rescale
                    )
                else:
                    noise_pred_audio = noise_pred_audio_g

                bsz = noise_pred_video.size(0)
                # Video x0-space blending (parent logic — unchanged; covers normal conditions + IC-LoRA ref tokens)
                denoised_sample_cond = (
                    noise_pred_video * (1 - conditioning_mask[:bsz]) + clean_latents.float() * conditioning_mask[:bsz]
                ).to(noise_pred_video.dtype)
                noise_pred_video = self.convert_x0_to_velocity(latents, denoised_sample_cond, i, self.scheduler)

                # Audio x0-space blending (NEW — mirrors the video blending)
                denoised_audio_sample_cond = (
                    noise_pred_audio * (1 - audio_conditioning_mask[:bsz])
                    + audio_clean_latents.float() * audio_conditioning_mask[:bsz]
                ).to(noise_pred_audio.dtype)
                noise_pred_audio = self.convert_x0_to_velocity(
                    audio_latents, denoised_audio_sample_cond, i, audio_scheduler
                )

                latents = self.scheduler.step(noise_pred_video, t, latents, return_dict=False)[0]
                audio_latents = audio_scheduler.step(noise_pred_audio, t, audio_latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        # 8. Strip appended tokens (keyframes + IC-LoRA reference) before decode.
        #    target_seq_len was captured before any sequence-dim concat.
        if keyframe_tokens is not None or control_tokens is not None:
            latents = latents[:, :target_seq_len, :]

        # Strip PREPENDED audio reference tokens (audio IC-LoRA) before audio decode.
        if audio_ref_num_tokens:
            audio_latents = audio_latents[:, audio_ref_num_tokens:, :]

        latents = self._unpack_latents(
            latents,
            latent_num_frames,
            latent_height,
            latent_width,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        )

        audio_latents = self._denormalize_audio_latents(
            audio_latents, self.audio_vae.latents_mean, self.audio_vae.latents_std
        )
        audio_latents = self._unpack_audio_latents(audio_latents, audio_num_frames, num_mel_bins=latent_mel_bins)

        if output_type == "latent":
            latents = self._denormalize_latents(
                latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
            )
            video = latents
            audio = audio_latents
        else:
            latents = latents.to(prompt_embeds.dtype)

            if not self.vae.config.timestep_conditioning:
                timestep = None
            else:
                noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=latents.dtype)
                if not isinstance(decode_timestep, list):
                    decode_timestep = [decode_timestep] * batch_size
                if decode_noise_scale is None:
                    decode_noise_scale = decode_timestep
                elif not isinstance(decode_noise_scale, list):
                    decode_noise_scale = [decode_noise_scale] * batch_size

                timestep = torch.tensor(decode_timestep, device=device, dtype=latents.dtype)
                decode_noise_scale = torch.tensor(decode_noise_scale, device=device, dtype=latents.dtype)[
                    :, None, None, None, None
                ]
                latents = (1 - decode_noise_scale) * latents + decode_noise_scale * noise

            latents = self._denormalize_latents(
                latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
            )
            latents = latents.to(self.vae.dtype)
            video = self.vae.decode(latents, timestep, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)

            audio_latents = audio_latents.to(self.audio_vae.dtype)
            generated_mel_spectrograms = self.audio_vae.decode(audio_latents, return_dict=False)[0]
            # Run vocoder in fp32 — bfloat16 accumulation errors compound through
            # BigVGAN v2's 108 sequential convs and degrade spectral metrics by 40–90%.
            # autocast upcasts per-op at kernel level (+70 MB vs +324 MB for model.float()).
            with torch.autocast(device_type=generated_mel_spectrograms.device.type, dtype=torch.float32):
                audio = self.vocoder(generated_mel_spectrograms)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video, audio)

        return LTX2PipelineOutput(frames=video, audio=audio)
