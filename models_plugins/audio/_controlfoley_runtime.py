"""Runtime for the ControlFoley plugin (xiaomi-research/controlfoley).

Kept free of ``bpy`` / Pallaidium imports so it can be exercised standalone.

ControlFoley is a research repo, not a pip package: ``controlfoley/`` (the
network and the feature extractors) and ``lib/`` (BigVGAN, the VAE, Synchformer,
CAV-MAE-ST and a vendored audiocraft, which supplies MusicGen-Style as the
timbre encoder) are meant to run from a checkout, and ``lib`` is a top-level
package name. Nothing is added to Blender's site-packages: on first use the
two source folders are downloaded as a pinned GitHub archive into
``%LOCALAPPDATA%/Pallaidium/controlfoley`` and put on ``sys.path`` from there.

Weights (all from the HF hub cache, fetched on first use):
  * YJX-Xiaomi/ControlFoley   weights/controlfoley.pth (11.3 GB, CC BY-NC 4.0)
                              ext_weights/{v1-44 VAE+vocoder, synchformer, cav_mae_st, CLAP}
  * facebook/musicgen-style   timbre encoder for reference-audio control
  * apple/DFN5B-CLIP-ViT-H-14-384   CLIP image/text encoder
  * nvidia/bigvgan_v2_44khz_128band_512x   44.1 kHz vocoder (config + generator only)
  * roberta-base, t5-base     CLAP / MusicGen text towers (small)
"""

import io
import os
import shutil
import sys
import types
import urllib.request
import zipfile

REPO_ID = "YJX-Xiaomi/ControlFoley"
SOURCE_SHA = "b6c902888d45d75f022e253e3db29836360a3f82"
SOURCE_URL = f"https://codeload.github.com/xiaomi-research/controlfoley/zip/{SOURCE_SHA}"

HOME = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "Pallaidium", "controlfoley")
SOURCE_DIR = os.path.join(HOME, "src-" + SOURCE_SHA[:8])

SAMPLE_RATE = 44100
CLAP_SR = 16000        # demo.py feeds CLAP the reference clip resampled to 16 kHz
TIMBRE_SR = 32000      # MusicGen-Style expects 32 kHz
TIMBRE_MIN_S, TIMBRE_MAX_S = 2.0, 4.0

_WEIGHTS = {
    "net":        "weights/controlfoley.pth",
    "vae":        "ext_weights/v1-44.pth",
    "synchformer": "ext_weights/synchformer_state_dict.pth",
    "cav_mae":    "ext_weights/cav_mae_st.pth",
    "clap":       "ext_weights/music_speech_audioset_epoch_15_esc_89.98.pt",
}
_EXTRA_REPOS = ("facebook/musicgen-style", "apple/DFN5B-CLIP-ViT-H-14-384")
_VOCODER_REPO = "nvidia/bigvgan_v2_44khz_128band_512x"      # only config.json + generator are used
_VOCODER_FILES = ["config.json", "bigvgan_generator.pt"]


# ── source checkout ────────────────────────────────────────────────────────

def ensure_source() -> str:
    """Download the pinned ControlFoley source (``controlfoley/`` + ``lib/``) once."""
    if os.path.isfile(os.path.join(SOURCE_DIR, "controlfoley", "audio_model.py")):
        return SOURCE_DIR
    print(f"ControlFoley: downloading source ({SOURCE_SHA[:8]}) ...")
    with urllib.request.urlopen(SOURCE_URL, timeout=120) as resp:
        data = resp.read()
    tmp = SOURCE_DIR + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    prefix = f"controlfoley-{SOURCE_SHA}/"
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.namelist():
            rel = member[len(prefix):] if member.startswith(prefix) else ""
            if not rel or rel.endswith("/") or rel.split("/", 1)[0] not in ("controlfoley", "lib"):
                continue
            dest = os.path.join(tmp, *rel.split("/"))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with zf.open(member) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
    shutil.rmtree(SOURCE_DIR, ignore_errors=True)
    os.replace(tmp, SOURCE_DIR)
    return SOURCE_DIR


def is_source_present() -> bool:
    return os.path.isfile(os.path.join(SOURCE_DIR, "controlfoley", "audio_model.py"))


def install_import_shims():
    """Make the repo importable next to Blender's stack.

    * flashy: audiocraft imports it only for multi-GPU training broadcasts
      (``flashy.distrib``); inference never reaches them.
    * wandb: pulled in by timm, crashes on NumPy 2 -> clean ImportError instead.
    """
    if "wandb" not in sys.modules:
        sys.modules["wandb"] = None
    if "flashy" not in sys.modules:
        try:
            import flashy  # noqa: F401
        except Exception:
            flashy = types.ModuleType("flashy")
            distrib = types.ModuleType("flashy.distrib")
            distrib.broadcast_tensors = lambda *a, **k: None
            distrib.average_tensors = lambda *a, **k: None
            distrib.world_size = lambda: 1
            distrib.rank = lambda: 0
            distrib.is_rank_zero = lambda: True
            flashy.distrib = distrib
            sys.modules["flashy"] = flashy
            sys.modules["flashy.distrib"] = distrib
    src = ensure_source()
    if src not in sys.path:
        sys.path.insert(0, src)


# ── weights ────────────────────────────────────────────────────────────────

def download_weights(local_files_only: bool = False) -> dict:
    """Resolve every checkpoint through the HF cache; returns name -> local path."""
    from huggingface_hub import hf_hub_download, snapshot_download

    paths = {k: hf_hub_download(REPO_ID, f, local_files_only=local_files_only)
             for k, f in _WEIGHTS.items()}
    for repo in _EXTRA_REPOS:
        snapshot_download(repo, local_files_only=local_files_only)
    paths["vocoder_dir"] = snapshot_download(
        _VOCODER_REPO, allow_patterns=_VOCODER_FILES, local_files_only=local_files_only)
    return paths


def is_cached() -> bool:
    try:
        download_weights(local_files_only=True)
        return True
    except Exception:
        return False


# ── model ──────────────────────────────────────────────────────────────────

def _patch_bigvgan_loader(vocoder_dir: str):
    """BigVGAN-v2's ``_from_pretrained`` predates huggingface_hub 1.x (its mixin no longer passes
    ``proxies`` / ``resume_download``, and hf_hub_download rejects them). Resolve the repo to its
    local snapshot instead, which also takes the loader's offline ``os.path.isdir`` branch."""
    from lib.bigvgan_v2.bigvgan import BigVGAN
    if getattr(BigVGAN, "_pallaidium_patched", False):
        return
    orig = BigVGAN._from_pretrained.__func__

    def _from_pretrained(cls, *, proxies=None, resume_download=False, model_id=None, **kw):
        if model_id == _VOCODER_REPO:
            model_id = vocoder_dir
        return orig(cls, proxies=proxies, resume_download=resume_download, model_id=model_id, **kw)

    BigVGAN._from_pretrained = classmethod(_from_pretrained)
    BigVGAN._pallaidium_patched = True


def load_models(device: str, net_dtype, local_files_only: bool = False) -> dict:
    """Build the generator network and every feature extractor."""
    import torch

    if not str(device).startswith("cuda"):
        raise RuntimeError("ControlFoley needs a CUDA GPU (its CAV-MAE-ST visual encoder is CUDA-only).")

    install_import_shims()
    paths = download_weights(local_files_only)

    _patch_bigvgan_loader(paths["vocoder_dir"])
    from controlfoley.audio_model import create_audio_generation_model
    from controlfoley.feature_extractor import FeaturesUtils
    from controlfoley.inference_utils import all_model_cfg

    cfg = all_model_cfg["large_44k"]

    # Build on the GPU in the target dtype, then stream the (mmapped) checkpoint in
    # tensor by tensor, so host RAM never holds the 11 GB fp32 state dict.
    net = create_audio_generation_model(cfg.model_name).to(device, net_dtype).eval()
    state = torch.load(paths["net"], map_location="cpu", weights_only=True, mmap=True)
    net.load_weights(state)
    del state

    # laion_clap's load_ckpt uses a plain torch.load (weights_only is True by default in
    # torch >= 2.6) and the CLAP checkpoint carries non-tensor objects.
    _orig_load = torch.load

    def _load_full(*a, **k):
        k.setdefault("weights_only", False)
        return _orig_load(*a, **k)

    # The same checkpoint also holds ``text_branch.embeddings.position_ids``, a buffer that
    # transformers 5 no longer registers -> strict load fails; drop the key.
    import laion_clap.hook as _clap_hook
    _orig_sd = _clap_hook.load_state_dict

    def _load_sd(*a, **k):
        sd = _orig_sd(*a, **k)
        sd.pop("text_branch.embeddings.position_ids", None)
        return sd

    torch.load = _load_full
    _clap_hook.load_state_dict = _load_sd
    try:
        feats = FeaturesUtils(
            tod_vae_ckpt=paths["vae"],
            synchformer_ckpt=paths["synchformer"],
            cav_mae_ckpt=paths["cav_mae"],
            clap_ckpt=paths["clap"],
            mode=cfg.mode,
            enable_conditions=True,
            need_vae_encoder=False,
        )
    finally:
        torch.load = _orig_load
        _clap_hook.load_state_dict = _orig_sd
    feats = feats.to(device, net_dtype).eval()
    if feats.clap_model is not None and net_dtype != torch.float32:
        # CLAP's BatchNorm stack needs fp32 weights and inputs (generate() casts to the
        # feature dtype, so the entry point converts back).
        feats.clap_model.float()
        _clap_encode = feats.encode_audio_with_clap
        feats.encode_audio_with_clap = lambda x, batch_size=-1: _clap_encode(x.float(), batch_size)
    return {"net": net, "feats": feats, "cfg": cfg, "device": device, "dtype": net_dtype}


# ── inputs ─────────────────────────────────────────────────────────────────

def _load_reference_audio(path: str, device, dtype):
    """Reference clip -> (clap_frames [1,1,N] @16 kHz, timbre_frames [1,1,N] @32 kHz, timbre_s).

    Mirrors demo.py: CLAP sees the whole clip, the timbre encoder 2-4 s of it.
    """
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        # mp3/aac/video containers: let torchaudio / av decode
        wav, sr = torchaudio.load(path)
        data = wav.numpy().T
    wav = torch.from_numpy(np.ascontiguousarray(data.T))          # [C, N]
    if wav.shape[-1] == 0:
        raise ValueError("The reference audio is empty.")

    clap = torchaudio.functional.resample(wav, sr, CLAP_SR) if sr != CLAP_SR else wav
    clap = clap.mean(dim=0, keepdim=True).reshape(1, -1).unsqueeze(0).to(device, dtype)

    timbre = torchaudio.functional.resample(wav, sr, TIMBRE_SR) if sr != TIMBRE_SR else wav
    lo, hi = int(TIMBRE_MIN_S * TIMBRE_SR), int(TIMBRE_MAX_S * TIMBRE_SR)
    if timbre.shape[-1] < lo:
        timbre = torch.nn.functional.pad(timbre, (0, lo - timbre.shape[-1]))
    elif timbre.shape[-1] > hi:
        timbre = timbre[..., :hi]
    timbre_s = timbre.shape[-1] / TIMBRE_SR
    timbre = timbre.mean(dim=0, keepdim=True).reshape(1, -1).unsqueeze(0).to(device, dtype)
    return clap, timbre, timbre_s


def probe_video_duration(path: str) -> float:
    import av
    with av.open(path) as c:
        s = c.streams.video[0]
        if s.duration is not None and s.time_base is not None:
            return float(s.duration * s.time_base)
        if c.duration:
            return c.duration / 1_000_000
    return 0.0


# ── generation ─────────────────────────────────────────────────────────────

def generate_audio(
    models: dict,
    *,
    video_path=None,
    ref_audio_path=None,
    prompt: str = "",
    neg_prompt: str = "",
    duration: float = 8.0,
    steps: int = 25,
    cfg_strength: float = 4.5,
    seed: int = 42,
    progress_fn=None,
):
    """Run one generation. Returns (waveform [1, N] float32 cpu, sample_rate, used_duration_s).

    Tasks follow the combination of inputs: video + prompt = TV2A / TC-V2A,
    video + reference audio = AC-V2A, video alone = V2A, prompt alone = T2A.
    """
    import torch
    from lib.flow_matching import FlowMatching
    from controlfoley.inference_utils import generate, load_video
    from pathlib import Path

    net, feats, cfg = models["net"], models["feats"], models["cfg"]
    device, dtype = models["device"], models["dtype"]
    fdtype = dtype                  # feature extractors share the network's dtype

    prompt = (prompt or "").strip()
    neg_prompt = (neg_prompt or "").strip()
    if not video_path and not prompt:
        raise ValueError("Give ControlFoley a video, a text prompt, or both.")
    if ref_audio_path and not video_path:
        raise ValueError("Reference-audio control (AC-V2A) needs a video to follow - select a MOVIE strip.")

    clip_f = vis_f = sync_f = None
    if video_path:
        info = load_video(Path(video_path), float(duration))
        if info.total_duration < duration:
            duration = info.total_duration
        clip_f = info.clip_embeddings.unsqueeze(0)
        sync_f = info.sync_embeddings.unsqueeze(0)
        vis_f = info.visual_features.unsqueeze(0)
        del info
    if duration < 1.0:
        raise ValueError(f"Duration too short ({duration:.2f}s).")

    audio_f = timbre_f = None
    timbre_s = 0.0
    if ref_audio_path:
        audio_f, timbre_f, timbre_s = _load_reference_audio(ref_audio_path, device, fdtype)

    seq_cfg = cfg.seq_cfg
    seq_cfg.total_time_seconds = float(duration)
    net.update_seq_lengths(seq_cfg.latent_sequence_length, seq_cfg.clip_sequence_length,
                           seq_cfg.visual_sequence_length, seq_cfg.sync_sequence_length)

    rng = torch.Generator(device=device)
    rng.manual_seed(int(seed) & 0xFFFFFFFFFFFF)
    fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=int(steps))

    # One ode_wrapper call per Euler step -> drive the progress bar from it.
    done = [0]
    orig_ode = net.ode_wrapper

    def _counted(*a, **k):
        out = orig_ode(*a, **k)
        done[0] += 1
        if progress_fn is not None:
            progress_fn(min(done[0], int(steps)), int(steps))
        return out

    net.ode_wrapper = _counted
    try:
        # autocast covers the few fp32 spots inside generate() (e.g. the timbre feature .float()).
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype, enabled=(dtype != torch.float32)):
            audio = generate(
                clip_f, vis_f, sync_f, audio_f, timbre_f, timbre_s,
                [prompt], negative_text=[neg_prompt],
                feature_utils=feats, net=net, fm=fm, rng=rng,
                cfg_strength=float(cfg_strength),
            )
    finally:
        net.ode_wrapper = orig_ode
    return audio.float().cpu()[0], seq_cfg.audio_sample_rate, float(duration)
