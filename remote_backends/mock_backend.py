"""Local mock backend implementing Pallaidium's Backend Contract v0.1.

Runs on the same computer as Blender — point the add-on's "Remote Backend URL"
at it (e.g. http://localhost:8000) and click "Refresh Remote Models".

Purpose: exercise the *whole* contract end-to-end (discovery, async video/image/
audio jobs, /v1/files upload, progress, cancellation, transcription) without any
cloud account or GPU. It returns small canned media:
  * image -> a 1x1 PNG (imports fine in the VSE)
  * audio -> a short silent WAV (imports fine)
  * video -> a tiny MP4 via ffmpeg if available, else a placeholder (use a real
             backend / the fal adapter for genuine video)
  * text  -> a fixed transcription string

Stdlib only.  Run:  python mock_backend.py --port 8000
"""

import argparse
import base64
import io
import json
import os
import struct
import subprocess
import tempfile
import threading
import time
import uuid
import wave
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


CONTRACT_VERSIONS = ["v0.1"]

# Models this mock advertises (with capability hints — see the extensions doc).
MODELS = [
    {"id": "mock-image", "type": "image", "modes": ["t2i", "i2i"],
     "max_ref_images": 3, "default_steps": 20},
    {"id": "mock-video", "type": "video", "modes": ["t2v", "i2v", "control"],
     "control_types": ["canny", "depth", "pose"]},
    {"id": "mock-tts", "type": "audio", "modes": ["tts"],
     "needs_speaker_ref": True, "needs_ref_text": True},
    {"id": "mock-asr", "type": "text", "modes": ["transcription"]},
]

_FILES = {}     # file_id -> bytes
_JOBS = {}      # job_id  -> {"created", "kind"}
_CANCELLED = set()
_LOCK = threading.Lock()

# How long a mock job "runs" before completing (seconds).
JOB_DURATION = 4.0


# --------------------------------------------------------------------------
# Canned media
# --------------------------------------------------------------------------
def _png_1x1() -> bytes:
    def chunk(typ, data):
        c = typ + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    raw = b"\x00\xff\x00\x00"  # one red pixel
    idat = zlib.compress(raw)
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def _wav_silence(seconds=1.0, rate=16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))
    return buf.getvalue()


def _mp4_tiny() -> bytes:
    """A ~1s solid-colour MP4 via ffmpeg if present, else a placeholder."""
    if _has_ffmpeg():
        path = os.path.join(tempfile.gettempdir(), f"mock_{uuid.uuid4().hex}.mp4")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=teal:s=320x240:d=1",
                 "-pix_fmt", "yuv420p", path],
                check=True, capture_output=True,
            )
            with open(path, "rb") as f:
                return f.read()
        except Exception:
            pass
        finally:
            if os.path.exists(path):
                os.remove(path)
    print("[mock] ffmpeg not found — returning a placeholder 'mp4' (won't play). "
          "Install ffmpeg or use a real backend for genuine video.")
    return b"\x00\x00\x00\x18ftypmp42" + b"MOCKPLACEHOLDER"


def _has_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


_KIND_MEDIA = {
    "image": (_png_1x1, "image/png"),
    "video": (_mp4_tiny, "video/mp4"),
    "audio": (_wav_silence, "audio/wav"),
}


# --------------------------------------------------------------------------
# HTTP handler
# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("[mock]", self.command, self.path)

    # -- helpers --
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _bin(self, data, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read(self):
        n = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(n) if n else b""

    # -- routes --
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/v1/health":
            return self._json({"status": "ok", "contract_versions": CONTRACT_VERSIONS})
        if path == "/v1/models":
            return self._json({"data": MODELS})
        if path.startswith("/v1/jobs/"):
            return self._job_status(path.rsplit("/", 1)[1])
        if path.startswith("/v1/files/"):
            fid = path.rsplit("/", 1)[1]
            with _LOCK:
                data = _FILES.get(fid)
            if data is None:
                return self._json({"error": "no such file"}, 404)
            return self._bin(data, "application/octet-stream")
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        path = self.path.split("?")[0]
        body = self._read()
        if path == "/v1/files":
            fid = "file-" + uuid.uuid4().hex
            with _LOCK:
                _FILES[fid] = body
            print(f"[mock]   stored {len(body)} bytes as {fid}")
            return self._json({"file_id": fid})
        if path == "/v1/audio/transcriptions":
            return self._json({"text": "This is a mock transcription."})
        if path in ("/v1/videos", "/v1/images/generations", "/v1/audio/speech"):
            payload = json.loads(body) if body else {}
            kind = {"/v1/videos": "video", "/v1/images/generations": "image",
                    "/v1/audio/speech": "audio"}[path]
            print(f"[mock]   {kind} request, payload keys: {sorted(payload.keys())}")
            jid = f"{kind}-{uuid.uuid4().hex}"
            with _LOCK:
                _JOBS[jid] = {"created": time.time(), "kind": kind}
            return self._json({"id": jid, "status": "queued"})
        return self._json({"error": "not found"}, 404)

    def _job_status(self, jid):
        with _LOCK:
            job = _JOBS.get(jid)
            cancelled = jid in _CANCELLED
        if job is None:
            return self._json({"error": "no such job"}, 404)
        if cancelled:
            return self._json({"id": jid, "status": "failed", "error": "cancelled"})
        elapsed = time.time() - job["created"]
        if elapsed < JOB_DURATION:
            return self._json({
                "id": jid, "status": "running", "phase": "generating",
                "progress": round(min(0.95, elapsed / JOB_DURATION), 2),
                "file_id": None, "error": None,
            })
        # complete: produce canned media
        maker, _ctype = _KIND_MEDIA[job["kind"]]
        fid = "out-" + jid
        with _LOCK:
            if fid not in _FILES:
                _FILES[fid] = maker()
        return self._json({
            "id": jid, "status": "succeeded", "phase": "done",
            "progress": 1.0, "file_id": fid, "error": None,
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Mock backend on http://{args.host}:{args.port}  (Ctrl-C to stop)")
    print(f"ffmpeg for real video: {'yes' if _has_ffmpeg() else 'NO (video is placeholder)'}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
