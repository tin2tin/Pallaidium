"""Multi-character dramatic TTS via Fish Audio Drama 3 (fishaudio/drama-3).

Drama 3 is a 35B MoE (3B active) TTS model: 80+ languages, voice cloning,
up to 16 speakers per text (<|speaker:0|> … <|speaker:15|>) and inline event
tags such as [sighs] / [laughs].

Voices: the panel holds an ordered voice list (scene.drama3_voices); entry N
is the reference for <|speaker:N|>, each with a reference clip and its
transcript (auto-transcribed with Whisper when left empty). The queue carries
the list as drama3_voices_json. Untagged prompts are spoken by the first voice.

The custom ``fish_omni`` architecture has no transformers / diffusers
implementation, so it can't run in Blender's Python.  This plugin runs it
locally through the native audio.cpp port instead:

  * runtime: audio.cpp ``drama-3-native`` branch (C++/GGML, CUDA)
    https://github.com/JarodMica/audio.cpp/blob/drama-3-native/docs/models/fish_drama3.md
  * weights: GGUF packages from https://huggingface.co/Jmica/Drama-3-GGUF
    (downloaded into the HF cache on first use, like any other model)

The plugin starts ``audiocpp_server.exe`` as a background process with a config
it writes itself, keeps it alive between jobs (the model stays loaded) and lets
the server unload the weights after DRAMA3_IDLE_UNLOAD_MS of inactivity so the
VRAM is free for other Pallaidium models. The server is stopped when Blender
quits. Nothing is added to requirements: the client is stdlib HTTP.

audio.cpp has no prebuilt Drama 3 binaries — build it once (VS 2022 C++ tools,
CUDA toolkit, CMake, git):

  git clone --branch drama-3-native --recurse-submodules ^
      https://github.com/JarodMica/audio.cpp.git %LOCALAPPDATA%\\Pallaidium\\audio.cpp
  cd %LOCALAPPDATA%\\Pallaidium\\audio.cpp
  .\\scripts\\build_windows.ps1 -Preset windows-cuda-release -ModelSet custom ^
      -Models fish_drama3 -Target audiocpp_server

The GGUF package and CPU expert offload are picked from the GPU's VRAM using
the port's documented profiles. Optional environment overrides:

  DRAMA3_SERVER_EXE         path to audiocpp_server.exe (default: the build above)
  DRAMA3_GGUF               package filename in Jmica/Drama-3-GGUF, or a local .gguf path
  DRAMA3_CPU_EXPERT_LAYERS  MoE layers whose experts run on the CPU (less VRAM, slower)
  DRAMA3_CONTEXT_LENGTH     token context (default 4096; longer refs/text need more)
  DRAMA3_MAX_TOKENS         hard cap on the per-request generation budget (default 2000;
                            each request is also bounded by its text length)
  DRAMA3_IDLE_UNLOAD_MS     unload weights after this idle time (default 300000)
  DRAMA3_PORT               local port (default 8035)
  DRAMA3_SERVER_URL         use an already-running server instead (audio.cpp or
                            SGLang-Omni) — the plugin then launches nothing
  DRAMA3_TIMEOUT            request timeout in seconds (default 1800)
"""

import atexit
import base64
import json
import os
import re
import time
import urllib.error
import urllib.request

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import solve_path, clean_filename

_GGUF_REPO = "Jmica/Drama-3-GGUF"
_MODEL_NAME = "drama-3"
_HOME = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "Pallaidium")
_DEFAULT_EXE = os.path.join(_HOME, "audio.cpp", "build", "windows-cuda-release", "bin",
                            "audiocpp_server.exe")
_WORK_DIR = os.path.join(_HOME, "drama3")
_PID_FILE = os.path.join(_WORK_DIR, "server.pid")

# (min VRAM GiB, package, cpu_expert_layers) — the audio.cpp port's documented
# starting profiles; the first row whose VRAM floor fits is used.
_PROFILES = [
    (40, "drama-3-backbone-8bit-experts-Q4_K.gguf", 0),
    (22, "drama-3-backbone-8bit-experts-Q4_K.gguf", 4),
    (15, "drama-3-backbone-8bit-experts-Q2_K.gguf", 8),
    (0,  "drama-3-backbone-8bit-experts-Q2_K.gguf", 20),
]

# Server process we launched (reused across jobs, stopped at exit) and the
# config it was started with, so a changed profile triggers a restart.
_SERVER_PROC = None
_SERVER_CFG = None
_SERVER_LOG = None
# package filename -> resolved local path (skips the hub round-trip per job)
_GGUF_PATHS = {}
# Largest context the server has been configured with this session. Only grows,
# so alternating short/long-reference jobs don't restart the server each time.
_CTX = 0
# (path, mtime) -> Whisper transcript, so a voice is transcribed once.
_TRANSCRIPTS = {}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _timeout() -> float:
    try:
        return float(os.environ.get("DRAMA3_TIMEOUT", "1800"))
    except ValueError:
        return 1800.0


def _get_json(url: str, timeout: float = 3.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8") or "null")


def _is_up(base: str) -> bool:
    # audio.cpp and SGLang both answer /health; /v1/models is the fallback.
    for path in ("/health", "/v1/models"):
        try:
            with urllib.request.urlopen(base + path, timeout=2.0) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
    return False


def _vram_gib() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    except Exception:
        pass
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
        ).stdout
        return float(out.strip().splitlines()[0]) / 1024
    except Exception:
        return 0.0


def _stop_server():
    global _SERVER_PROC, _SERVER_CFG, _SERVER_LOG
    if _SERVER_PROC is not None and _SERVER_PROC.poll() is None:
        print("Drama 3: stopping audiocpp_server")
        _SERVER_PROC.terminate()
        try:
            _SERVER_PROC.wait(timeout=10)
        except Exception:
            _SERVER_PROC.kill()
    if _SERVER_LOG is not None:
        try:
            _SERVER_LOG.close()
        except Exception:
            pass
    _SERVER_PROC = _SERVER_CFG = _SERVER_LOG = None


atexit.register(_stop_server)


def _resolve_gguf(prefs, package: str) -> str:
    """Local path of the GGUF package, downloading it into the HF cache if needed."""
    if os.path.isfile(package):
        return package
    cached = _GGUF_PATHS.get(package)
    if cached and os.path.isfile(cached):
        return cached
    from huggingface_hub import hf_hub_download
    local_only = bool(getattr(prefs, "local_files_only", False))
    try:
        snap_path = hf_hub_download(_GGUF_REPO, package, local_files_only=local_only)
    except Exception as e:
        if local_only:
            raise OSError(
                f"Drama 3: {package} is not downloaded. Uncheck 'Use Local Files Only' "
                "in Preferences to download it (~16-23 GB)."
            ) from e
        raise
    _GGUF_PATHS[package] = _gguf_named_link(snap_path, package)
    return _GGUF_PATHS[package]


def _gguf_named_link(snap_path: str, package: str) -> str:
    """Return a path to the GGUF that ends in '.gguf'.

    The HF snapshot entry is a symlink to an extension-less blob; audio.cpp
    resolves it and picks the loader from the extension ("unsupported tensor
    source format"). A hard link to the blob, kept inside the same repo cache
    folder (same volume), gives it a .gguf name without using extra disk.
    """
    blob = os.path.realpath(snap_path)
    if blob.lower().endswith(".gguf"):
        return blob
    repo_dir = os.path.dirname(os.path.dirname(blob))   # models--Jmica--Drama-3-GGUF
    link = os.path.join(repo_dir, "pallaidium", package)
    try:
        if os.path.isfile(link) and os.path.samefile(link, blob):
            return link
        os.makedirs(os.path.dirname(link), exist_ok=True)
        if os.path.lexists(link):
            os.remove(link)   # stale link to an older revision
        os.link(blob, link)
        return link
    except OSError as e:
        raise RuntimeError(
            f"Drama 3: could not create a .gguf hard link for {blob} ({e}). "
            "Set DRAMA3_GGUF to a local .gguf file instead."
        ) from e


def _build_config(prefs, port: int, context: int) -> dict:
    vram = _vram_gib()
    package, cpu_layers = next((p, c) for floor, p, c in _PROFILES if vram >= floor)
    package = os.environ.get("DRAMA3_GGUF", "").strip().strip('"') or package
    cpu_layers = _env_int("DRAMA3_CPU_EXPERT_LAYERS", cpu_layers)
    print(f"Drama 3: VRAM {vram:.1f} GiB -> {os.path.basename(package)}, "
          f"cpu_expert_layers={cpu_layers}")
    return {
        "host": "127.0.0.1",
        "port": port,
        "backend": "cuda" if vram > 0 else "cpu",
        "threads": max(1, min(8, (os.cpu_count() or 8) // 2)),
        "lazy_load": True,
        "max_loaded_models": 1,
        "idle_unload_ms": _env_int("DRAMA3_IDLE_UNLOAD_MS", 300000),
        # A long take can be minutes on a 24 GB card with offload.
        "busy_timeout_ms": 0,
        "ui": False,
        "models": [{
            "id": _MODEL_NAME,
            "family": "fish_drama3",
            "path": _resolve_gguf(prefs, package).replace("\\", "/"),
            "task": "tts",
            "mode": "offline",
            "session_options": {
                "fish_drama3.cpu_expert_layers": cpu_layers,
                "fish_drama3.context_length": context,
            },
            "default_request_options": {"max_tokens": _env_int("DRAMA3_MAX_TOKENS", 2000)},
        }],
    }


def _kill_orphan() -> bool:
    """Stop an audiocpp_server we started earlier but lost track of.

    Happens after a plugin reload (module globals reset) or a Blender crash.
    Only the PID recorded in our pid file is touched, and only if that PID is
    still an audiocpp_server process. Returns True if one was stopped.
    """
    import subprocess
    try:
        with open(_PID_FILE) as f:
            pid = int(f.read().strip())
    except (OSError, ValueError):
        return False
    try:
        if os.name == "nt":
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ).stdout
            if "audiocpp_server" not in out.lower():
                return False
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, timeout=10,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            with open(f"/proc/{pid}/comm") as f:
                if "audiocpp_server" not in f.read():
                    return False
            os.kill(pid, 15)
    except Exception:
        return False
    print(f"Drama 3: stopped orphaned audiocpp_server (pid {pid})")
    try:
        os.remove(_PID_FILE)
    except OSError:
        pass
    return True


def _ensure_local_server(prefs, phase, context: int) -> str:
    """Start (or reuse) our audiocpp_server and return its base URL."""
    global _SERVER_PROC, _SERVER_CFG, _SERVER_LOG
    import subprocess

    exe = os.environ.get("DRAMA3_SERVER_EXE", "").strip().strip('"') or _DEFAULT_EXE
    if not os.path.isfile(exe):
        raise RuntimeError(
            f"Drama 3: audiocpp_server.exe not found at {exe}. Build the audio.cpp "
            "drama-3-native branch (see the fish_drama3.py docstring) or set DRAMA3_SERVER_EXE."
        )
    port = _env_int("DRAMA3_PORT", 8035)
    base = f"http://127.0.0.1:{port}"

    phase("Preparing weights")
    cfg = _build_config(prefs, port, context)
    running = _SERVER_PROC is not None and _SERVER_PROC.poll() is None
    if running and cfg == _SERVER_CFG and _is_up(base):
        return base
    if running:
        _stop_server()   # profile changed — restart with the new config
    if _is_up(base) and _kill_orphan():
        time.sleep(1.0)
    if _is_up(base):
        raise RuntimeError(
            f"Drama 3: port {port} is already in use by another process. "
            "Set DRAMA3_SERVER_URL to use it, or DRAMA3_PORT to pick another port."
        )

    os.makedirs(_WORK_DIR, exist_ok=True)
    cfg_path = os.path.join(_WORK_DIR, "server.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    log_path = os.path.join(_WORK_DIR, "server.log")

    # The server needs the CUDA runtime DLLs; make sure the toolkit's bin dir is
    # on PATH even if Blender's environment was trimmed.
    env = dict(os.environ)
    cuda_bin = os.path.join(env.get("CUDA_PATH", ""), "bin")
    if os.path.isdir(cuda_bin):
        env["PATH"] = cuda_bin + os.pathsep + env.get("PATH", "")

    phase("Starting audio.cpp server")
    print(f"Drama 3: launching {exe} --config {cfg_path} (log: {log_path})")
    _SERVER_LOG = open(log_path, "ab")
    _SERVER_PROC = subprocess.Popen(
        [exe, "--config", cfg_path],
        cwd=os.path.dirname(exe), env=env,
        stdout=_SERVER_LOG, stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
    )
    _SERVER_CFG = cfg
    try:
        with open(_PID_FILE, "w") as f:
            f.write(str(_SERVER_PROC.pid))
    except OSError:
        pass
    deadline = time.time() + 60.0
    while time.time() < deadline:
        if _SERVER_PROC.poll() is not None:
            code = _SERVER_PROC.returncode
            _stop_server()
            raise RuntimeError(f"Drama 3: audiocpp_server exited with code {code} - see {log_path}")
        if _is_up(base):
            return base
        time.sleep(0.5)
    _stop_server()
    raise RuntimeError(f"Drama 3: audiocpp_server did not come up on {base} - see {log_path}")


def _trim_trailing_silence(wav_bytes: bytes, threshold: int = 64, keep_s: float = 0.3) -> bytes:
    """Cut a run-on silent tail (see max_tokens note in generate()). 16-bit PCM only."""
    import array
    import io
    import wave
    try:
        with wave.open(io.BytesIO(wav_bytes)) as w:
            params = w.getparams()
            frames = w.readframes(params.nframes)
        if params.sampwidth != 2:
            return wav_bytes
        a = array.array("h", frames)
        ch, sr = params.nchannels, params.framerate
        win = max(1, sr // 100) * ch            # 10 ms windows
        end = len(a)
        while end > 0 and max(abs(x) for x in a[max(0, end - win):end]) < threshold:
            end -= win
        end = min(len(a), max(0, end) + int(keep_s * sr) * ch)
        end -= end % ch
        if end >= len(a) or end <= 0:
            return wav_bytes
        print(f"Drama 3: trimmed {(len(a) - end) / ch / sr:.1f} s of trailing silence")
        out = io.BytesIO()
        with wave.open(out, "wb") as w:
            w.setparams(params)
            w.writeframes(a[:end].tobytes())
        return out.getvalue()
    except Exception as e:
        print(f"Drama 3: silence trim skipped ({e})")
        return wav_bytes


def _context_for(ref_seconds: float, text: str, max_tokens: int) -> int:
    """Context rows needed: ~21.5 codec rows per second of reference audio,
    the reference/input text, the generation budget and template overhead.
    Rounded up to 1024 and never below DRAMA3_CONTEXT_LENGTH (default 4096)."""
    global _CTX
    need = int(ref_seconds * 22) + len(text) + max_tokens + 1024
    ctx = max(_env_int("DRAMA3_CONTEXT_LENGTH", 4096), -(-need // 1024) * 1024, _CTX)
    _CTX = ctx
    return ctx


def _as_wav(path: str, speaker: int) -> tuple:
    """(wav path, seconds) for a reference clip; converts non-WAV / non-PCM
    files and clips to 60 s (Drama 3's reference limit)."""
    import wave
    try:
        with wave.open(path) as w:
            secs = w.getnframes() / float(w.getframerate())
            if w.getsampwidth() == 2 and secs <= 60:
                return path, secs
    except Exception:
        pass
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    data = data.mean(axis=1)[: int(60 * sr)]
    os.makedirs(_WORK_DIR, exist_ok=True)
    out = os.path.join(_WORK_DIR, f"ref_speaker{speaker}.wav")
    sf.write(out, data, sr, subtype="PCM_16")
    return out, len(data) / float(sr)


def _transcribe(path: str) -> str:
    """Whisper transcript of a reference clip (cached per file+mtime)."""
    key = (path, os.path.getmtime(path))
    if key in _TRANSCRIPTS:
        return _TRANSCRIPTS[key]
    from faster_whisper import WhisperModel
    print(f"Drama 3: transcribing reference {os.path.basename(path)} with Whisper (small, CPU)")
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, _info = model.transcribe(path)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    del model
    print(f"Drama 3: reference text = {text!r}")
    _TRANSCRIPTS[key] = text
    return text


def _format_turns(text: str) -> str:
    """One speaker turn per line: '<|speaker:0|>Hi\n<|speaker:1|>Hello'.
    Turns run together on one line made the model drop later speakers (it
    stalled into silence after the first turn); a line break before each tag
    delivered every turn in testing."""
    return re.sub(r"\s*(<\|speaker:\d+\|>)\s*", lambda m: "\n" + m.group(1), text).strip()


def _pick_model_id(base: str) -> str:
    """Return the served model id (audio.cpp routes by it); '' if not listed."""
    try:
        data = _get_json(base + "/v1/models") or {}
        ids = [m.get("id", "") for m in data.get("data", []) if isinstance(m, dict)]
    except Exception:
        return ""
    for mid in ids:
        if "drama" in mid.lower():
            return mid
    return ids[0] if ids else ""


class FishDrama3Plugin(ModelPlugin):
    MODEL_ID     = "fishaudio/drama-3"
    DISPLAY_NAME = "TTS: Fish Audio Drama 3"
    MODEL_TYPE   = "audio"
    DESCRIPTION  = (
        "Multi-speaker dramatic TTS with voice cloning, <|speaker:N|> and [sighs]/[laughs] "
        "tags, 80+ languages. Runs locally via audio.cpp + GGUF (Jmica/Drama-3-GGUF)"
    )

    # Voices are read from drama3_voices_json (scene.drama3_voices serialized
    # by the queue), not the shared single Speaker Ref. / Ref. Text fields.
    INPUTS       = InputSpec.PROMPT
    UI_SECTIONS  = [
        UISection.PROMPT,
        UISection.SEED,
    ]
    PARAMS = ParamSpec()
    REQUIRED_PACKAGES = ["huggingface_hub"]
    show_enhance = False

    # ------------------------------------------------------------------ load

    def load(self, prefs, scene, **kw):
        # Weights live in the audio.cpp server process; it is started lazily
        # in generate() so selecting the model costs nothing.
        return {"pipe": None, "model_id": None, "base": None}

    def _ensure_server(self, pipe_obj, inputs, prefs, context: int) -> str:
        external = os.environ.get("DRAMA3_SERVER_URL", "").strip().rstrip("/")
        if external:
            if not _is_up(external):
                raise RuntimeError(f"Drama 3: no server reachable at DRAMA3_SERVER_URL={external}.")
            base = external
        else:
            base = _ensure_local_server(prefs, lambda label: self.set_phase(inputs, label), context)
        if pipe_obj.get("base") != base:
            pipe_obj["base"] = base
            pipe_obj["model_id"] = _pick_model_id(base)
            print(f"Drama 3: server {base} model={pipe_obj['model_id'] or '(default)'}")
        return base

    # ------------------------------------------------------------------ request

    @staticmethod
    def _post(url: str, payload: dict) -> tuple:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "audio/wav"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=_timeout()) as r:
                return r.status, r.headers.get("Content-Type", ""), r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers.get("Content-Type", ""), e.read()

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs) -> str:
        text = (inputs.prompt or "").strip()
        if not text:
            raise ValueError("Drama 3: prompt is empty.")

        # ---- voices: [{speaker, audio, text}] from the UI voice list --------
        try:
            voices = json.loads(getattr(scene, "drama3_voices_json", "") or "[]")
        except ValueError:
            voices = []
        refs, ref_seconds = [], 0.0
        for v in voices:
            n, path = int(v.get("speaker", 0)), (v.get("audio") or "").strip()
            if not path:
                continue
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Drama 3: Speaker {n} audio not found: {path}")
            wav, secs = _as_wav(path, n)
            ref_text = (v.get("text") or "").strip()
            if not ref_text:
                self.set_phase(inputs, f"Transcribing speaker {n}")
                ref_text = _transcribe(wav)
                if not ref_text:
                    raise ValueError(f"Drama 3: Speaker {n} reference has no speech to transcribe; "
                                     "type its transcript in the Text field.")
            # The prompt builder reads the speaker tag from the reference text.
            ref_text = re.sub(r"^\s*<\|speaker:\d+\|>\s*", "", ref_text)
            refs.append({"speaker": n,
                         "ref": {"text": f"<|speaker:{n}|>{ref_text}", "audio_path": os.path.abspath(wav)}})
            ref_seconds += secs
            v["text"] = ref_text
        # Hand the resolved voices (with any Whisper transcripts) back to the
        # queue, which stores them on the job and in the strip metadata.
        if voices:
            try:
                scene.drama3_voices_json = json.dumps(voices)
            except Exception:
                pass

        # An untagged prompt is spoken by the first voice.
        if refs and not re.search(r"<\|speaker:\d+\|>", text):
            text = f"<|speaker:{refs[0]['speaker']}|>{text}"
        text = _format_turns(text)
        used = sorted({int(n) for n in re.findall(r"<\|speaker:(\d+)\|>", text)})
        missing = [n for n in used if n not in {r["speaker"] for r in refs}]
        if refs and missing:
            print(f"Drama 3: speakers {missing} have no voice reference; "
                  "the model invents a voice for them.")
        refs = [r["ref"] for r in refs]

        payload = {"input": text, "response_format": "wav"}
        if inputs.seed:
            # Pallaidium seeds can be negative; audio.cpp requires unsigned.
            payload["seed"] = int(inputs.seed) & 0xFFFFFFFF
        # Bound generation by text length: tagged takes ([laughs] etc.) can fail
        # to emit end-of-speech and run on as silence until the budget is spent
        # (a 77-char line produced 93 s at the 2000-token cap). ~21.5 codec
        # frames/s and ~14 chars/s of speech -> ~1.5 frames/char; allow ~3x plus
        # a floor. The server applies it per ~200-char chunk.
        spoken = len(re.sub(r"<\|speaker:\d+\|>", "", text).strip())
        payload["max_tokens"] = min(_env_int("DRAMA3_MAX_TOKENS", 2000),
                                    250 + 4 * min(spoken, 250))

        context = _context_for(ref_seconds, text + "".join(r["text"] for r in refs),
                               payload["max_tokens"])
        base = self._ensure_server(pipe_obj, inputs, prefs, context)
        if pipe_obj.get("model_id"):
            payload["model"] = pipe_obj["model_id"]

        # The runtimes take references differently. audio.cpp only forwards a
        # fixed set of top-level fields to the model; anything else (references
        # included) must sit in "options", or it is silently ignored. SGLang-Omni
        # takes a top-level references list. Try audio.cpp first; fall back to
        # the SGLang shape when the server rejects the request.
        attempts = [payload]
        if refs:
            attempts = [dict(payload, options={"references": json.dumps(refs)}),
                        dict(payload, references=refs)]

        print(f"Drama 3 | voices={len(refs)} ({ref_seconds:.1f} s ref) | speakers={used} | "
              f"seed={inputs.seed} | {len(text)} chars | context={context}")
        # The first request after start/idle-unload also loads the weights.
        self.set_phase(inputs, "Generating (audio.cpp)")

        status, ctype, body = 0, "", b""
        for i, p in enumerate(attempts):
            status, ctype, body = self._post(base + "/v1/audio/speech", p)
            if status == 200:
                break
            print(f"Drama 3: request shape {i + 1}/{len(attempts)} rejected (HTTP {status}): "
                  f"{body[:400].decode('utf-8', 'replace')}")
            if status not in (400, 422):
                break   # not a schema problem — don't retry with another shape

        if status != 200:
            raise RuntimeError(
                f"Drama 3: server error HTTP {status}: {body[:600].decode('utf-8', 'replace')}"
            )

        # response_format=json (or a server that ignores it) returns base64 WAV.
        if "json" in ctype.lower():
            data = json.loads(body.decode("utf-8"))
            b64 = data.get("audio") or data.get("data") or data.get("b64_json") or ""
            if not b64:
                raise RuntimeError(f"Drama 3: JSON response without audio: {str(data)[:300]}")
            body = base64.b64decode(b64.split(",", 1)[-1])
        if not body.startswith(b"RIFF"):
            raise RuntimeError(f"Drama 3: server returned non-WAV data ({ctype or 'unknown type'}).")

        self.set_phase(inputs, "Saving")
        body = _trim_trailing_silence(body)
        name = re.sub(r"<\|speaker:\d+\|>", " ", inputs.prompt or "").strip()
        output_path = solve_path(clean_filename(str(inputs.seed) + "_" + name) + ".wav")
        with open(output_path, "wb") as f:
            f.write(body)
        print(f"Drama 3 saved: {output_path}")
        return output_path

    # ------------------------------------------------------------------ UI

    def draw_custom_ui(self, col, context) -> bool:
        scene = context.scene
        voices = getattr(scene, "drama3_voices", None)
        if voices is None:
            col.label(text="Restart Blender to enable the Drama 3 voice list", icon="ERROR")
        else:
            col.label(text="Voices (speaker references)", icon="OUTLINER_OB_SPEAKER")
            for i, v in enumerate(voices):
                box = col.box()
                row = box.row(align=True)
                row.label(text=f"Speaker {i}:  <|speaker:{i}|>")
                row.operator("sequencer.drama3_voice_remove", text="", icon="X").index = i
                box.prop(v, "audio_path", text="Audio")
                box.prop(v, "ref_text", text="Text")
                if v.audio_path and not v.ref_text:
                    box.label(text="Text empty: will be transcribed with Whisper", icon="INFO")
            row = col.row()
            row.enabled = len(voices) < 16
            row.operator("sequencer.drama3_voice_add", text="Add Voice", icon="ADD")
            if not voices:
                col.label(text="No voices: the model invents one per speaker", icon="INFO")
        col.separator()
        tips = col.column(align=True)
        tips.scale_y = 0.8
        tips.label(text="Start each line with <|speaker:N|> to pick a voice")
        tips.label(text="Untagged text uses the first voice")
        tips.label(text="Events: [sighs] [laughs] ... inline in the text")
        box = col.box()
        external = os.environ.get("DRAMA3_SERVER_URL", "").strip()
        exe = os.environ.get("DRAMA3_SERVER_EXE", "").strip().strip('"') or _DEFAULT_EXE
        if external:
            box.label(text=f"Server: {external}", icon="URL")
        elif os.path.isfile(exe):
            box.label(text="Runtime: audio.cpp (local)", icon="CHECKMARK")
        else:
            box.label(text="audio.cpp not built - see plugin docstring", icon="ERROR")
        return False
