"""Standalone tests for the remote-backend integration (no Blender needed).

Run:  python test_remote_backend.py

Spins up an in-process mock server implementing the Backend Contract for all
four media types and exercises:
  * client: health/version, /v1/models, async job poll+download, sync url,
    b64_json, /v1/files upload, transcription, cooperative cancellation
  * factory: make_remote_plugin() per media type (endpoint/ext/INPUTS/UI)
  * mapping: build_payload() + upload_inputs() incl. multi-ref + voice clone

This shims the add-on package so the relative imports in remote_base.py resolve
without importing Blender (`bpy`).
"""

import io
import json
import os
import sys
import types
import threading
import importlib.util
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
PKG = "_palla_test"


# ---------------------------------------------------------------------------
# Package shim so `from .base import` / `from ..utils.remote_backend import` work
# ---------------------------------------------------------------------------
def _fake_pkg(name, path):
    m = types.ModuleType(name)
    m.__path__ = [path]
    m.__package__ = name
    sys.modules[name] = m


def _load(modname, relpath, pkg):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = pkg
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_fake_pkg(PKG, ROOT)
_fake_pkg(PKG + ".utils", os.path.join(ROOT, "utils"))
_fake_pkg(PKG + ".models", os.path.join(ROOT, "models"))

rb = _load(PKG + ".utils.remote_backend", "utils/remote_backend.py", PKG + ".utils")
_load(PKG + ".models.base", "models/base.py", PKG + ".models")
remote_base = _load(PKG + ".models.remote_base", "models/remote_base.py", PKG + ".models")
base = sys.modules[PKG + ".models.base"]


# ---------------------------------------------------------------------------
# Fakes (avoid PIL dependency)
# ---------------------------------------------------------------------------
class FakeImage:
    def __init__(self, tag=b"IMG"):
        self.tag = tag

    def save(self, fp, format=None):
        fp.write(self.tag)


class FakePrefs:
    def __init__(self, url):
        self.remote_backend_url = url
        self.remote_backend_key = ""


PNG = b"\x89PNG\r\n\x1a\nFAKE"
WAV = b"RIFFFAKEWAVE"
MP4 = b"\x00\x00\x00\x18ftypmp42FAKE"

MODELS = [
    {"id": "flux", "type": "image", "modes": ["t2i", "i2i"], "max_ref_images": 9,
     "default_steps": 28},
    {"id": "seedance", "type": "video", "modes": ["t2v", "i2v", "control"],
     "control_types": ["canny", "depth"]},
    {"id": "tts-clone", "type": "audio", "modes": ["tts"],
     "needs_speaker_ref": True, "needs_ref_text": True},
    {"id": "whisper", "type": "text", "modes": ["transcription"]},
]


class MockBackend(BaseHTTPRequestHandler):
    files = {}          # file_id -> bytes
    last_payload = {}   # endpoint -> dict
    job_polls = {}      # job_id -> count
    cancel_forever = False  # if True, jobs never finish (for cancel test)

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _bin(self, data, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/v1/health":
            return self._json({"status": "ok", "contract_versions": ["v0.1"]})
        if p == "/v1/models":
            return self._json({"data": MODELS})
        if p.startswith("/v1/jobs/"):
            jid = p.rsplit("/", 1)[1]
            n = MockBackend.job_polls.get(jid, 0) + 1
            MockBackend.job_polls[jid] = n
            if MockBackend.cancel_forever:
                return self._json({"id": jid, "status": "running",
                                   "phase": "generating", "progress": 0.3})
            if n < 2:
                return self._json({"id": jid, "status": "running",
                                   "phase": "generating", "progress": 0.5})
            # produce a file appropriate to the job kind encoded in the id
            kind = jid.split(":")[0]
            data = {"image": PNG, "video": MP4, "audio": WAV}.get(kind, PNG)
            fid = f"out-{jid}"
            MockBackend.files[fid] = data
            return self._json({"id": jid, "status": "succeeded",
                               "phase": "done", "progress": 1.0, "file_id": fid})
        if p.startswith("/v1/files/"):
            fid = p.rsplit("/", 1)[1]
            return self._bin(MockBackend.files.get(fid, b""), "application/octet-stream")
        return self._json({"error": "not found"}, 404)

    def _read_body(self):
        n = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(n) if n else b""

    def do_POST(self):
        p = self.path.split("?")[0]
        body = self._read_body()
        if p == "/v1/files":
            # crude multipart: store the raw payload, return an id
            fid = f"file-{len(MockBackend.files)}"
            MockBackend.files[fid] = body
            return self._json({"file_id": fid})
        if p == "/v1/audio/transcriptions":
            return self._json({"text": "hello world"})
        if p in ("/v1/videos", "/v1/images/generations", "/v1/audio/speech"):
            payload = json.loads(body) if body else {}
            MockBackend.last_payload[p] = payload
            # sync b64 path when asked
            if payload.get("_sync_b64"):
                import base64
                return self._json({"data": [{"b64_json": base64.b64encode(PNG).decode()}]})
            if payload.get("_sync_url"):
                fid = "sync-1"
                MockBackend.files[fid] = PNG
                return self._json({"data": [{"url": f"{BASE}/v1/files/{fid}"}]})
            kind = {"/v1/videos": "video", "/v1/images/generations": "image",
                    "/v1/audio/speech": "audio"}[p]
            return self._json({"id": f"{kind}:{len(MockBackend.last_payload)}",
                               "status": "queued"})
        return self._json({"error": "not found"}, 404)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
srv = HTTPServer(("127.0.0.1", 0), MockBackend)
PORT = srv.server_address[1]
BASE = f"http://127.0.0.1:{PORT}"
threading.Thread(target=srv.serve_forever, daemon=True).start()

SCRATCH = os.environ.get("TEMP", "/tmp")
PASSED = []


def ok(name):
    PASSED.append(name)
    print(f"  PASS {name}")


def main():
    client = rb.RemoteBackendClient(BASE)

    # -- client: health + models -------------------------------------------
    assert client.check_compatible()["status"] == "ok"
    models = client.models()
    assert len(models) == 4
    ok("health + /v1/models discovery")

    # -- client: async image job poll + download ---------------------------
    dst = os.path.join(SCRATCH, "rb_img.png")
    out = client.run("/v1/images/generations", {"model": "flux", "prompt": "x"}, dst)
    assert open(out, "rb").read() == PNG
    ok("image async submit->poll->download")

    # -- client: video + audio jobs ----------------------------------------
    assert open(client.run("/v1/videos", {"model": "seedance"},
                           os.path.join(SCRATCH, "rb_vid.mp4")), "rb").read() == MP4
    assert open(client.run("/v1/audio/speech", {"model": "tts", "input": "hi"},
                           os.path.join(SCRATCH, "rb_aud.wav")), "rb").read() == WAV
    ok("video + audio async jobs")

    # -- client: sync url + b64_json ---------------------------------------
    out = client.run("/v1/images/generations", {"_sync_url": True},
                     os.path.join(SCRATCH, "rb_url.png"))
    assert open(out, "rb").read() == PNG
    out = client.run("/v1/images/generations", {"_sync_b64": True},
                     os.path.join(SCRATCH, "rb_b64.png"))
    assert open(out, "rb").read() == PNG
    ok("sync url + b64_json responses")

    # -- client: file upload + transcription -------------------------------
    fid = client.upload_bytes("ref.png", PNG, "reference")
    assert fid in MockBackend.files
    assert client.transcribe(_tmpfile("a.wav", WAV), "whisper")["text"] == "hello world"
    ok("upload_bytes + transcription")

    # -- client: cancellation -> KeyboardInterrupt --------------------------
    MockBackend.cancel_forever = True
    flag = {"cancel": False}
    # cancel after the first poll
    import threading as _t
    def flip():
        import time; time.sleep(0.1); flag["cancel"] = True
    _t.Thread(target=flip, daemon=True).start()
    try:
        client.run("/v1/videos", {"model": "seedance"},
                   os.path.join(SCRATCH, "rb_cancel.mp4"),
                   should_cancel=lambda: flag["cancel"], )
        raise AssertionError("expected KeyboardInterrupt")
    except KeyboardInterrupt:
        ok("should_cancel -> KeyboardInterrupt")
    finally:
        MockBackend.cancel_forever = False

    # -- factory: one plugin per media type --------------------------------
    plugins = {m["id"]: remote_base.make_remote_plugin(m) for m in models}
    IS = base.InputSpec
    UISec = base.UISection

    img = plugins["flux"]
    assert img.MODEL_ID == "remote:flux" and img.REMOTE_MODEL_NAME == "flux"
    assert img.MODEL_TYPE == "image" and img.ENDPOINT == "/v1/images/generations"
    assert img.OUTPUT_EXT == ".png"
    assert IS.IMAGE in img.INPUTS and IS.MULTI_IMAGE in img.INPUTS  # i2i + 9 refs
    assert img.PARAMS.steps == 28

    vid = plugins["seedance"]
    assert vid.MODEL_TYPE == "video" and vid.ENDPOINT == "/v1/videos"
    assert vid.OUTPUT_EXT == ".mp4"
    assert IS.IMAGE in vid.INPUTS and IS.VIDEO in vid.INPUTS  # i2v + control

    aud = plugins["tts-clone"]
    assert aud.MODEL_TYPE == "audio" and aud.ENDPOINT == "/v1/audio/speech"
    assert IS.AUDIO_REF in aud.INPUTS and IS.TEXT_REF in aud.INPUTS  # voice clone

    txt = plugins["whisper"]
    assert txt.MODEL_TYPE == "text" and txt.requires_input_strip is True
    ok("factory builds correct plugin per media type")

    # -- mapping: image build_payload + multi-ref upload -------------------
    mi = base.ModelInputs(prompt="a cat", neg_prompt="blur", width=1024,
                          height=1024, steps=20, guidance=4.0, seed=7)
    mi.image = FakeImage(b"INIT")
    mi.images = [FakeImage(b"R1"), FakeImage(b"R2")]
    payload = img.build_payload(mi, None, None)
    assert payload["num_inference_steps"] == 20 and payload["guidance_scale"] == 4.0
    img.upload_inputs(client, mi, payload)
    assert "image_file_id" in payload
    assert len(payload["reference_file_ids"]) == 2
    ok("image payload + multi-ref upload")

    # -- mapping: video i2v + control upload -------------------------------
    mv = base.ModelInputs(prompt="pan", frames=49, fps=24.0, seed=1, strength=0.5)
    mv.image = FakeImage(b"FRAME")
    mv.video_path = _tmpfile("ctrl.mp4", MP4)
    pv = vid.build_payload(mv, None, None)
    vid.upload_inputs(client, mv, pv)
    assert pv["num_frames"] == 49 and pv["fps"] == 24.0
    assert "image_file_id" in pv and pv.get("image_b64")
    assert "control_file_id" in pv and pv["control_type"] == "canny"
    ok("video payload + i2v image_b64 + control upload")

    # -- mapping: voice clone (speaker ref + ref text) ---------------------
    ma = base.ModelInputs(prompt="speak this", speed=1.0)
    ma.audio_ref = _tmpfile("spk.wav", WAV)
    ma.text_ref = "reference transcript"
    pa = aud.build_payload(ma, None, None)
    aud.upload_inputs(client, ma, pa)
    assert pa["input"] == "speak this"
    assert "speaker_reference_id" in pa
    assert pa["speaker_reference_text"] == "reference transcript"
    ok("audio payload + voice-clone ref audio + ref text")

    # -- transcription generate() (no Blender) -----------------------------
    mt = base.ModelInputs()
    mt.video_path = _tmpfile("speech.wav", WAV)
    text = txt.generate(None, mt, None, FakePrefs(BASE))
    assert text == "hello world"
    ok("transcription generate() returns text")

    print(f"\nALL {len(PASSED)} CHECKS PASSED")


def _tmpfile(name, data):
    path = os.path.join(SCRATCH, "rbtest_" + name)
    with open(path, "wb") as f:
        f.write(data)
    return path


if __name__ == "__main__":
    try:
        main()
    finally:
        srv.shutdown()
