"""Standalone contract tests for the direct SiftQ MiniMax-H3 V2 plugin.

Run with a regular Python interpreter; Blender and a real SiftQ key are not
required.  The test imports the same public plugin class Pallaidium discovers
and exercises its generate() path against a deterministic local HTTP server.
"""

import base64
import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest
import urllib.error
import urllib.parse
import urllib.request
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock


ROOT = os.path.dirname(os.path.abspath(__file__))
PKG = "_palla_siftq_test"
MP4 = b"\x00\x00\x00\x18ftypmp42SIFTQ-TEST"


def _fake_pkg(name, path):
    module = types.ModuleType(name)
    module.__path__ = [path]
    module.__package__ = name
    sys.modules[name] = module


def _load(modname, relpath, package):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(ROOT, relpath))
    module = importlib.util.module_from_spec(spec)
    module.__package__ = package
    sys.modules[modname] = module
    spec.loader.exec_module(module)
    return module


_fake_pkg(PKG, ROOT)
_fake_pkg(PKG + ".utils", os.path.join(ROOT, "utils"))
_fake_pkg(PKG + ".models", os.path.join(ROOT, "models"))
_fake_pkg(PKG + ".models_plugins", os.path.join(ROOT, "models_plugins"))
_fake_pkg(PKG + ".models_plugins.video", os.path.join(ROOT, "models_plugins", "video"))

base = _load(PKG + ".models.base", "models/base.py", PKG + ".models")
h3 = _load(PKG + ".utils.siftq_h3_v2", "utils/siftq_h3_v2.py", PKG + ".utils")
plugin_mod = _load(
    PKG + ".models_plugins.video.siftq_minimax_h3",
    "models_plugins/video/siftq_minimax_h3.py",
    PKG + ".models_plugins.video",
)


class FakeImage:
    size = (512, 512)

    def convert(self, mode):
        return self

    def save(self, handle, format=None):
        handle.write(b"PNG-SIFTQ-TEST")


class MockState:
    statuses = ["succeeded"]
    task_mode = "video"
    response_mode = "normal"
    error_status = None
    download_mode = "valid"
    delete_mode = "cancelled"
    last_payload = None
    last_headers = None
    last_list_query = ""
    delete_count = 0
    download_count = 0

    @classmethod
    def reset(cls):
        cls.statuses = ["succeeded"]
        cls.task_mode = "video"
        cls.response_mode = "normal"
        cls.error_status = None
        cls.download_mode = "valid"
        cls.delete_mode = "cancelled"
        cls.last_payload = None
        cls.last_headers = None
        cls.last_list_query = ""
        cls.delete_count = 0
        cls.download_count = 0


ERROR_TYPES = {
    400: "bad_request_error",
    401: "authorized_error",
    402: "insufficient_balance_error",
    422: "unprocessable_entity_error",
    429: "rate_limit_error",
    500: "server_error",
    529: "overloaded_error",
}


class MockSiftQ(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _json(self, value, code=200):
        body = json.dumps(value).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _raw(self, body, content_type="application/json", code=200):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        return self.rfile.read(length) if length else b""

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        raw = self._body()
        MockState.last_headers = dict(self.headers.items())
        MockState.last_payload = json.loads(raw.decode("utf-8")) if raw else {}
        if MockState.error_status:
            code = MockState.error_status
            return self._json({
                "type": "error",
                "error": {
                    "type": ERROR_TYPES[code],
                    "message": "safe detail https://signed.example/file?api_key=secret",
                    "http_code": str(code),
                },
                "request_id": "req-test",
            }, code)
        if path.endswith("/v2/video_generation"):
            if MockState.response_mode == "malformed_json":
                return self._raw(b"{not-json")
            if MockState.response_mode == "non_object_json":
                return self._raw(b"[]")
            if MockState.response_mode == "missing_task_id":
                return self._json({})
            return self._json({"task_id": "task/one"})
        if path.endswith("/v2/h3_context_ir"):
            return self._json({"task_id": "context-one"})
        return self._json({"type": "error"}, 404)

    def do_GET(self):
        parsed = self.path.split("?", 1)
        path = parsed[0]
        if path == "/media/output.mp4":
            MockState.download_count += 1
            if MockState.download_mode == "expired_once" and MockState.download_count == 1:
                return self._json({"error": "expired"}, 403)
            download_mode = "valid" if MockState.download_mode == "expired_once" else MockState.download_mode
            if MockState.download_mode == "expired":
                return self._json({"error": "expired"}, 403)
            if MockState.download_mode == "http_error":
                return self._json({"error": "download failed"}, 503)
            body = {
                "valid": MP4,
                "empty": b"",
                "bad_signature": b"not-an-mp4",
                "bad_type": MP4,
            }[download_mode]
            content_type = "text/plain" if download_mode == "bad_type" else "video/mp4"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path.endswith("/v2/query/video_generation"):
            MockState.last_list_query = parsed[1] if len(parsed) > 1 else ""
            list_status = "processing" if MockState.response_mode == "list_processing" else "succeeded"
            item = self._task("listed", list_status)
            return self._json({"items": [item], "total": 1})
        if "/v2/query/video_generation/" in path:
            if MockState.response_mode == "missing_task":
                return self._json({})
            status = MockState.statuses.pop(0) if len(MockState.statuses) > 1 else MockState.statuses[0]
            task = self._task("task/one", status)
            if MockState.response_mode == "missing_result_id":
                task.pop("id", None)
            if MockState.response_mode == "missing_content" and status == "succeeded":
                task["content"] = {}
            if MockState.response_mode == "wrong_modality":
                task["modality"] = "video" if MockState.task_mode == "context" else "text"
            if MockState.response_mode == "v1_status":
                task["status"] = "Success"
            return self._json({"task": task})
        return self._json({"error": "not found"}, 404)

    def do_DELETE(self):
        MockState.delete_count += 1
        task_id = self.path.rsplit("/", 1)[-1]
        task_id = urllib.parse.unquote(task_id)
        if MockState.delete_mode == "reject":
            return self._json({
                "type": "error",
                "error": {
                    "type": "unprocessable_entity_error",
                    "message": "task state does not permit deletion",
                    "http_code": "422",
                },
                "request_id": "req-delete",
            }, 422)
        action = "deleted" if MockState.delete_mode == "deleted" else "cancelled"
        return self._json({"task_id": task_id, "action": action, "status": action})

    @staticmethod
    def _task(task_id, status):
        task = {
            "id": task_id,
            "model": "MiniMax-H3",
            "status": status,
            "task_type": "generation",
            "modality": "video",
        }
        if status == "succeeded":
            if MockState.task_mode == "context":
                task.update({
                    "task_type": "h3_context_ir",
                    "modality": "text",
                    "content": {"prompt": "enhanced prompt"},
                    "usage": {"total_tokens": 12, "prompt_tokens": 5, "completion_tokens": 7},
                })
            else:
                task.update({
                    "content": {"url": BASE + "/media/output.mp4"},
                    "usage": {"total_seconds": 5, "output_seconds": 5, "input_image_count": 1},
                })
        elif status == "failed":
            task["error"] = {"code": "generation_failed", "message": "provider rejected media"}
        return task


SERVER = HTTPServer(("127.0.0.1", 0), MockSiftQ)
PORT = SERVER.server_address[1]
BASE = f"http://127.0.0.1:{PORT}"
API_BASE = BASE + "/api/minimax/"
threading.Thread(target=SERVER.serve_forever, daemon=True).start()


def image_uri():
    return "data:image/png;base64," + base64.b64encode(b"image").decode("ascii")


def video_uri():
    return "data:video/mp4;base64," + base64.b64encode(b"video").decode("ascii")


def audio_uri():
    return "data:audio/wav;base64," + base64.b64encode(b"audio").decode("ascii")


def png_bytes(width=512, height=512):
    return (
        b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR"
        + width.to_bytes(4, "big") + height.to_bytes(4, "big")
        + b"\x08\x02\x00\x00\x00"
    )


def jpeg_bytes(width=512, height=512):
    return (
        b"\xff\xd8\xff\xc0\x00\x0b\x08"
        + height.to_bytes(2, "big") + width.to_bytes(2, "big")
        + b"\x01\x01\x11\x00\xff\xd9"
    )


def webp_bytes(width=512, height=512):
    header = bytearray(30)
    header[:4] = b"RIFF"
    header[4:8] = (22).to_bytes(4, "little")
    header[8:12] = b"WEBP"
    header[12:16] = b"VP8X"
    header[16:20] = (10).to_bytes(4, "little")
    header[24:27] = (width - 1).to_bytes(3, "little")
    header[27:30] = (height - 1).to_bytes(3, "little")
    return bytes(header)


def heif_bytes(width=512, height=512):
    return (
        (20).to_bytes(4, "big") + b"ispe" + b"\x00\x00\x00\x00"
        + width.to_bytes(4, "big") + height.to_bytes(4, "big")
    )


def bmff_box(box_type, payload=b""):
    return (len(payload) + 8).to_bytes(4, "big") + box_type + payload


def bmff_fullbox(box_type, payload=b""):
    return bmff_box(box_type, b"\x00\x00\x00\x00" + payload)


def mp4_reference_bytes(codec=b"avc1", width=512, height=512, fps=24, seconds=5):
    timescale = 24000
    samples = fps * seconds
    delta = timescale // fps
    mvhd = bmff_fullbox(
        b"mvhd", b"\x00" * 8 + timescale.to_bytes(4, "big")
        + (timescale * seconds).to_bytes(4, "big"),
    )
    tkhd = bmff_fullbox(
        b"tkhd", b"\x00" * 64
        + (width << 16).to_bytes(4, "big")
        + (height << 16).to_bytes(4, "big"),
    )
    mdhd = bmff_fullbox(
        b"mdhd", b"\x00" * 8 + timescale.to_bytes(4, "big")
        + (timescale * seconds).to_bytes(4, "big"),
    )
    hdlr = bmff_fullbox(b"hdlr", b"\x00" * 4 + b"vide")
    stsd = bmff_fullbox(b"stsd", (1).to_bytes(4, "big") + bmff_box(codec))
    stts = bmff_fullbox(
        b"stts", (1).to_bytes(4, "big")
        + samples.to_bytes(4, "big") + delta.to_bytes(4, "big"),
    )
    stbl = bmff_box(b"stbl", stsd + stts)
    minf = bmff_box(b"minf", stbl)
    mdia = bmff_box(b"mdia", mdhd + hdlr + minf)
    trak = bmff_box(b"trak", tkhd + mdia)
    return bmff_box(b"ftyp", b"isom\x00\x00\x02\x00isom") + bmff_box(b"moov", mvhd + trak)


def mp3_reference_bytes(frame_count=80):
    header = bytes.fromhex("fffb9000")  # MPEG-1 Layer III, 128 kbps, 44.1 kHz
    return (header + b"\x00" * 413) * frame_count


class SiftQContractTests(unittest.TestCase):
    def setUp(self):
        MockState.reset()
        self.client = h3.SiftQClient("unit-key", API_BASE, timeout=2)

    def test_exact_default_and_override_routes(self):
        expected = {
            "create_video": "https://siftq.com/api/minimax/v2/video_generation",
            "query_task": "https://siftq.com/api/minimax/v2/query/video_generation/task%20id",
            "list_tasks": "https://siftq.com/api/minimax/v2/query/video_generation",
            "delete_task": "https://siftq.com/api/minimax/v2/video_generation/task%20id",
            "create_context_ir": "https://siftq.com/api/minimax/v2/h3_context_ir",
        }
        self.assertEqual(h3.route_urls(task_id="task id"), expected)
        self.assertEqual(h3.route_urls("https://example.test/root", "a/b")["query_task"],
                         "https://example.test/root/v2/query/video_generation/a%2Fb")
        self.assertEqual(h3.route_urls("https://example.test/root/", "x")["create_video"],
                         "https://example.test/root/v2/video_generation")

    def test_invalid_base_and_missing_key(self):
        for value in (
            "", "relative/path", "ftp://example.test", "https://u:p@example.test",
            "https://example.test:not-a-port",
        ):
            with self.assertRaises(h3.SiftQValidationError):
                h3.normalize_base_url(value)
        with self.assertRaises(h3.SiftQValidationError):
            h3.SiftQClient("", API_BASE)
        with self.assertRaisesRegex(h3.SiftQValidationError, "timeout"):
            h3.SiftQClient("key", API_BASE, timeout=0)
        with self.assertRaises(h3.SiftQValidationError):
            h3.build_video_payload(prompt="x", media=[
                {"type": "image_url", "image_url": {"url": "https://example.test:bad/x"},
                 "role": "reference_image"},
            ], ratio="adaptive")
        with self.assertRaises(h3.SiftQError):
            h3.SiftQClient._validate_output_url("https://example.test:bad/video.mp4")

    def test_authenticated_api_redirect_cannot_cross_origin(self):
        handler = h3._SameOriginRedirectHandler()
        request = urllib.request.Request(
            "https://siftq.example/api/task",
            headers={"Authorization": "Bearer unit-key"},
        )
        with self.assertRaisesRegex(urllib.error.URLError, "cross-origin"):
            handler.redirect_request(
                request, None, 307, "redirect", {}, "https://other.example/task"
            )

    def test_all_advertised_payload_modes(self):
        text = h3.build_video_payload(prompt="ocean", resolution="2K", duration=5, ratio="16:9")
        first = h3.build_video_payload(
            prompt="pan", media=[{"type": "image_url", "image_url": {"url": image_uri()},
                                  "role": "first_frame"}], ratio="adaptive",
        )
        first_last = h3.build_video_payload(
            prompt="transition", media=[
                {"type": "image_url", "image_url": {"url": image_uri()}, "role": "first_frame"},
                {"type": "image_url", "image_url": {"url": image_uri()}, "role": "last_frame"},
            ], ratio="adaptive",
        )
        refs = h3.build_video_payload(
            prompt="reference", media=[
                {"type": "image_url", "image_url": {"url": image_uri()}, "role": "reference_image"},
                {"type": "video_url", "video_url": {"url": video_uri()}, "role": "reference_video"},
                {"type": "audio_url", "audio_url": {"url": audio_uri()}, "role": "reference_audio"},
            ], ratio="adaptive",
        )
        self.assertEqual(text["model"], "MiniMax-H3")
        self.assertEqual(first["content"][1]["role"], "first_frame")
        self.assertEqual(len(first_last["content"]), 3)
        self.assertEqual({item.get("role") for item in refs["content"][1:]},
                         {"reference_image", "reference_video", "reference_audio"})

    def test_payload_boundaries_and_v1_shapes_rejected(self):
        invalid_calls = [
            lambda: h3.build_video_payload(prompt="", ratio="16:9"),
            lambda: h3.build_video_payload(prompt="x", duration=0, ratio="16:9"),
            lambda: h3.build_video_payload(prompt="x", duration=16, ratio="16:9"),
            lambda: h3.build_video_payload(prompt="x", duration=True, ratio="16:9"),
            lambda: h3.build_video_payload(prompt="x" * 7001, ratio="16:9"),
            lambda: h3.build_video_payload(prompt="x", resolution="1080P", ratio="16:9"),
            lambda: h3.build_video_payload(prompt="x", ratio="adaptive"),
            lambda: h3.build_video_payload(prompt="x", media=[
                {"type": "image_url", "image_url": {"url": image_uri()}, "role": "last_frame"},
            ], ratio="adaptive"),
            lambda: h3.build_video_payload(prompt="x", media=[
                {"type": "image_url", "image_url": {"url": image_uri()}, "role": "first_frame"},
                {"type": "audio_url", "audio_url": {"url": audio_uri()}, "role": "reference_audio"},
            ], ratio="adaptive"),
            lambda: h3.build_video_payload(prompt="x", media=[
                {"type": "image_url", "image_url": {"url": image_uri()}, "role": "reference_image"}
                for _ in range(10)
            ], ratio="adaptive"),
            lambda: h3.build_video_payload(prompt="x", media=[
                {"type": "video_url", "video_url": {"url": video_uri()},
                 "role": "reference_audio"},
            ], ratio="adaptive"),
        ]
        for call in invalid_calls:
            with self.assertRaises(h3.SiftQValidationError):
                call()
        with mock.patch.object(h3, "MAX_REQUEST_BYTES", 100):
            with self.assertRaises(h3.SiftQValidationError):
                h3.build_video_payload(prompt="x" * 80, ratio="16:9")
        with mock.patch.object(h3, "MAX_IMAGE_BYTES", 4):
            with self.assertRaisesRegex(h3.SiftQValidationError, "per-file size"):
                h3.build_video_payload(prompt="x", media=[
                    {"type": "image_url", "image_url": {"url": image_uri()},
                     "role": "reference_image"},
                ], ratio="adaptive")
        MockState.response_mode = "v1_status"
        with self.assertRaisesRegex(h3.SiftQError, "unexpected status"):
            self.client.query_task("task/one")

    def test_submit_auth_headers_and_success_response(self):
        payload = h3.build_video_payload(prompt="safe", ratio="16:9")
        self.assertEqual(self.client.create_video(payload), "task/one")
        self.assertEqual(MockState.last_headers["Authorization"], "Bearer unit-key")
        self.assertEqual(MockState.last_headers["Content-Type"], "application/json")
        self.assertEqual(MockState.last_headers["User-Agent"], "Pallaidium-SiftQ/1.0")
        self.assertNotIn("unit-key", json.dumps(MockState.last_payload))

    def test_structured_http_errors_are_redacted(self):
        payload = h3.build_video_payload(prompt="safe", ratio="16:9")
        for code, error_type in ERROR_TYPES.items():
            MockState.error_status = code
            with self.assertRaises(h3.SiftQError) as raised:
                self.client.create_video(payload)
            self.assertEqual(raised.exception.status_code, code)
            self.assertEqual(raised.exception.error_type, error_type)
            self.assertEqual(raised.exception.http_code, str(code))
            self.assertEqual(raised.exception.request_id, "req-test")
            self.assertNotIn("signed.example", str(raised.exception))
            self.assertNotIn("secret", str(raised.exception))
        MockState.error_status = None

    def test_malformed_and_incomplete_envelopes(self):
        payload = h3.build_video_payload(prompt="safe", ratio="16:9")
        MockState.response_mode = "malformed_json"
        with self.assertRaisesRegex(h3.SiftQError, "malformed JSON"):
            self.client.create_video(payload)
        MockState.response_mode = "non_object_json"
        with self.assertRaisesRegex(h3.SiftQError, "non-object"):
            self.client.create_video(payload)
        MockState.response_mode = "normal"
        with mock.patch.object(h3, "MAX_JSON_RESPONSE_BYTES", 4):
            with self.assertRaisesRegex(h3.SiftQError, "oversized JSON"):
                self.client.create_video(payload)
        MockState.response_mode = "missing_task_id"
        with self.assertRaisesRegex(h3.SiftQError, "missing task_id"):
            self.client.create_video(payload)
        MockState.response_mode = "missing_task"
        with self.assertRaisesRegex(h3.SiftQError, "missing task"):
            self.client.query_task("task/one")
        MockState.response_mode = "missing_content"
        with self.assertRaisesRegex(h3.SiftQError, "without content.url"):
            self.client.query_task("task/one")
        MockState.response_mode = "missing_result_id"
        with self.assertRaisesRegex(h3.SiftQError, "missing id"):
            self.client.query_task("task/one")

        wrong_model = dict(payload, model="legacy-model")
        with self.assertRaisesRegex(h3.SiftQValidationError, "exactly"):
            self.client.create_video(wrong_model)

    def test_transport_timeout_is_normalized(self):
        payload = h3.build_video_payload(prompt="safe", ratio="16:9")
        with mock.patch.object(self.client._api_opener, "open", side_effect=TimeoutError()):
            with self.assertRaisesRegex(h3.SiftQError, "timed out") as raised:
                self.client.create_video(payload)
        self.assertTrue(raised.exception.delivery_uncertain)

    def test_poll_queued_running_success_and_usage(self):
        MockState.statuses = ["queued", "running", "succeeded"]
        phases, progress = [], []
        task = self.client.poll_task(
            "task/one", interval=0, max_wait=10,
            phase_fn=phases.append, progress_fn=lambda current, total: progress.append((current, total)),
        )
        self.assertEqual(task["status"], "succeeded")
        self.assertEqual(phases, ["Queued at SiftQ", "Generating at SiftQ"])
        self.assertEqual(progress[-1], (100, 100))
        self.assertIn("total 5s", h3.usage_note(task))

    def test_live_processing_status_alias_is_normalized(self):
        MockState.statuses = ["processing"]
        task = self.client.query_task("task/one")
        self.assertEqual(task["status"], "running")

        MockState.response_mode = "list_processing"
        listed = self.client.list_tasks(page_num=1, page_size=10, model=h3.MODEL_ID)
        self.assertEqual(listed["items"][0]["status"], "running")

        with self.assertRaises(h3.SiftQValidationError):
            self.client.list_tasks(status="processing")

    def test_failure_cancel_timeout_and_cancellation_rules(self):
        MockState.statuses = ["failed"]
        with self.assertRaisesRegex(h3.SiftQError, "generation_failed"):
            self.client.poll_task("task/one", interval=0, max_wait=1)
        MockState.statuses = ["cancelled"]
        with self.assertRaisesRegex(h3.SiftQError, "cancelled upstream"):
            self.client.poll_task("task/one", interval=0, max_wait=1)

        MockState.statuses = ["queued"]
        with self.assertRaises(KeyboardInterrupt):
            self.client.poll_task("task/one", interval=0, max_wait=1, should_cancel=lambda: True)
        self.assertEqual(MockState.delete_count, 1)

        MockState.statuses = ["running"]
        with self.assertRaises(KeyboardInterrupt):
            self.client.poll_task("task/one", interval=0, max_wait=1, should_cancel=lambda: True)
        self.assertEqual(MockState.delete_count, 1, "running tasks must not be sent to DELETE")

        MockState.statuses = ["queued"]
        ticks = iter([0.0, 2.0])
        with self.assertRaisesRegex(h3.SiftQError, "timed out"):
            self.client.poll_task(
                "task/one", interval=0, max_wait=1,
                clock_fn=lambda: next(ticks), sleep_fn=lambda _: None,
            )

    def test_list_context_ir_delete_and_callback_contracts(self):
        result = self.client.list_tasks(
            page_num=1, page_size=10, status="succeeded",
            task_ids=["a", "b"], model="MiniMax-H3", task_type="generation",
        )
        self.assertEqual(result["total"], 1)
        parsed = urllib.parse.parse_qs(MockState.last_list_query)
        self.assertEqual(parsed["filter.task_ids"], ["a", "b"])

        context_payload = h3.build_context_ir_payload(prompt="enhance", duration=5, ratio="16:9")
        self.assertNotIn("resolution", context_payload)
        self.assertEqual(self.client.create_context_ir(context_payload), "context-one")
        MockState.task_mode = "context"
        context_task = self.client.query_task("context-one")
        self.assertEqual(context_task["content"]["prompt"], "enhanced prompt")
        self.assertEqual(context_task["usage"]["total_tokens"], 12)
        MockState.response_mode = "wrong_modality"
        with self.assertRaisesRegex(h3.SiftQError, "unexpected modality"):
            self.client.query_task("context-one")

        MockState.task_mode = "video"
        MockState.response_mode = "normal"
        deleted = self.client.delete_task("task/one")
        self.assertEqual(deleted["action"], "cancelled")
        MockState.delete_mode = "deleted"
        deleted = self.client.delete_task("task/one")
        self.assertEqual(deleted["status"], "deleted")
        MockState.delete_mode = "reject"
        with self.assertRaises(h3.SiftQError) as rejected:
            self.client.delete_task("task/one")
        self.assertEqual(rejected.exception.status_code, 422)
        self.assertEqual(h3.callback_challenge_response({"challenge": "unchanged"}),
                         {"challenge": "unchanged"})
        with self.assertRaises(h3.SiftQValidationError):
            h3.callback_challenge_response({"challenge": ""})

    def test_download_validation_and_failure_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = os.path.join(directory, "video.mp4")
            self.assertEqual(self.client.download_video(BASE + "/media/output.mp4", destination),
                             destination)
            with open(destination, "rb") as handle:
                self.assertEqual(handle.read(), MP4)
            for mode in ("empty", "bad_signature", "bad_type", "expired", "http_error"):
                MockState.download_mode = mode
                with self.assertRaises(h3.SiftQError):
                    self.client.download_video(BASE + "/media/output.mp4", destination)
            with self.assertRaises(h3.SiftQError):
                self.client.download_video("file:///tmp/output.mp4", destination)

            MockState.download_mode = "expired_once"
            MockState.download_count = 0
            MockState.statuses = ["succeeded"]
            task = self.client.query_task("task/one")
            self.assertEqual(
                self.client.download_task_video("task/one", task, destination), destination,
            )
            self.assertEqual(MockState.download_count, 2)

    def test_file_type_and_image_dimension_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            bad = os.path.join(directory, "ref.txt")
            with open(bad, "wb") as handle:
                handle.write(b"not media")
            with self.assertRaises(h3.SiftQValidationError):
                self.client.data_uri_from_file(bad, "video")
        tiny = FakeImage()
        tiny.size = (128, 128)
        with self.assertRaises(h3.SiftQValidationError):
            self.client.pil_image_to_data_uri(tiny)

    def test_image_file_inspection_is_stdlib_only(self):
        samples = {
            ".png": (png_bytes(), "data:image/png;base64,"),
            ".jpg": (jpeg_bytes(), "data:image/jpeg;base64,"),
            ".webp": (webp_bytes(), "data:image/webp;base64,"),
            ".heic": (heif_bytes(), "data:image/heic;base64,"),
            ".heif": (heif_bytes(), "data:image/heif;base64,"),
        }
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(sys.modules, {"PIL": None, "PIL.Image": None}):
                for ext, (content, prefix) in samples.items():
                    path = os.path.join(directory, "frame" + ext)
                    with open(path, "wb") as handle:
                        handle.write(content)
                    self.assertTrue(self.client.data_uri_from_file(path, "image").startswith(prefix))

            invalid = os.path.join(directory, "invalid.png")
            with open(invalid, "wb") as handle:
                handle.write(b"not-a-png")
            with self.assertRaisesRegex(h3.SiftQValidationError, "Could not inspect"):
                self.client.data_uri_from_file(invalid, "image")

            tiny = os.path.join(directory, "tiny.png")
            with open(tiny, "wb") as handle:
                handle.write(png_bytes(128, 128))
            with self.assertRaisesRegex(h3.SiftQValidationError, "dimensions"):
                self.client.data_uri_from_file(tiny, "image")

    def test_timed_media_inspection_has_stdlib_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            video = os.path.join(directory, "reference.mp4")
            wav = os.path.join(directory, "reference.wav")
            mp3 = os.path.join(directory, "reference.mp3")
            with open(video, "wb") as handle:
                handle.write(mp4_reference_bytes())
            with wave.open(wav, "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(8000)
                output.writeframes(b"\x00\x00" * (8000 * 2))
            with open(mp3, "wb") as handle:
                handle.write(mp3_reference_bytes())

            with mock.patch.dict(sys.modules, {"av": None}):
                self.assertTrue(
                    self.client.data_uri_from_file(video, "video").startswith(
                        "data:video/mp4;base64,"
                    )
                )
                self.assertTrue(
                    self.client.data_uri_from_file(wav, "audio").startswith(
                        "data:audio/wav;base64,"
                    )
                )
                self.assertTrue(
                    self.client.data_uri_from_file(mp3, "audio").startswith(
                        "data:audio/mpeg;base64,"
                    )
                )


class SiftQPluginTests(unittest.TestCase):
    def setUp(self):
        MockState.reset()
        self.plugin = plugin_mod.SiftQMiniMaxH3Plugin()

    @staticmethod
    def scene(mode, **overrides):
        values = dict(
            siftq_mode=mode, siftq_resolution="768P", siftq_duration=5,
            siftq_ratio="16:9", siftq_reference_ratio="16:9", siftq_ref_count=1,
            siftq_ref_strip_1="", siftq_ref_strip_1_path="",
            siftq_ref_strip_2="", siftq_ref_strip_2_path="",
        )
        values.update(overrides)
        return types.SimpleNamespace(**values)

    def test_plugin_maps_first_last_and_reference_inputs(self):
        client = h3.SiftQClient("key", API_BASE)
        first_last_inputs = base.ModelInputs(prompt="transition", image=FakeImage(), last_image=FakeImage())
        first_last = self.plugin.build_request(first_last_inputs, self.scene("FIRST_LAST"), client)
        self.assertEqual([item.get("role") for item in first_last["content"][1:]],
                         ["first_frame", "last_frame"])
        self.assertEqual(first_last["ratio"], "adaptive")

        class MappingClient:
            def pil_image_to_data_uri(self, image):
                return image_uri()

            def data_uri_from_file(self, path, kind):
                return {"image": image_uri(), "video": video_uri(), "audio": audio_uri()}[kind]

        with tempfile.TemporaryDirectory() as directory:
            ref = os.path.join(directory, "ref.png")
            with open(ref, "wb") as handle:
                handle.write(b"ref")
            scene = self.scene("REFERENCE", siftq_ref_strip_1_path=ref)
            inputs = base.ModelInputs(prompt="reference", image=FakeImage())
            inputs.video_path = os.path.join(directory, "ref.mp4")
            inputs.audio_ref = os.path.join(directory, "ref.wav")
            mapped = self.plugin.build_request(inputs, scene, MappingClient())
        roles = [item.get("role") for item in mapped["content"][1:]]
        self.assertEqual(roles.count("reference_video"), 1)
        self.assertEqual(roles.count("reference_audio"), 1)
        self.assertEqual(roles.count("reference_image"), 1)
        self.assertNotIn("first_frame", roles)

    def test_reference_ratio_total_count_and_path_deduplication(self):
        class MappingClient:
            def data_uri_from_file(self, path, kind):
                return image_uri()

            def pil_image_to_data_uri(self, image):
                return image_uri()

        with tempfile.TemporaryDirectory() as directory:
            main = os.path.join(directory, "main.png")
            with open(main, "wb") as handle:
                handle.write(b"main")
            values = {
                "image_path": main,
                "siftq_reference_ratio": "adaptive",
                "siftq_ref_count": 9,
            }
            for index in range(1, 10):
                path = main if index == 1 else os.path.join(directory, f"ref-{index}.png")
                if path != main:
                    with open(path, "wb") as handle:
                        handle.write(b"ref")
                values[f"siftq_ref_strip_{index}"] = ""
                values[f"siftq_ref_strip_{index}_path"] = path
            scene = self.scene("REFERENCE", **values)
            mapped = self.plugin.build_request(
                base.ModelInputs(prompt="reference"), scene, MappingClient(),
            )
            self.assertEqual(mapped["ratio"], "adaptive")
            self.assertEqual(
                sum(item.get("role") == "reference_image" for item in mapped["content"]), 9,
            )

            extra = os.path.join(directory, "extra.png")
            with open(extra, "wb") as handle:
                handle.write(b"extra")
            scene.siftq_ref_strip_1_path = extra
            with self.assertRaisesRegex(h3.SiftQValidationError, "selected main IMAGE"):
                self.plugin.build_request(
                    base.ModelInputs(prompt="reference"), scene, MappingClient(),
                )

    def test_plugin_is_available_without_optional_pyav(self):
        with mock.patch.dict(sys.modules, {"av": None}):
            self.assertEqual(self.plugin.REQUIRED_PACKAGES, [])
            self.assertTrue(self.plugin.is_available())

    def test_frame_pickers_are_visible_and_map_without_selected_input(self):
        class FakeLayout:
            def __init__(self):
                self.search_labels = []
                self.operators = []

            def row(self, **kwargs):
                return self

            def prop_search(self, owner, attr, collection_owner, collection_attr, **kwargs):
                self.search_labels.append(kwargs.get("text"))

            def operator(self, *args, **kwargs):
                op = types.SimpleNamespace(action="")
                self.operators.append(op)
                return op

            def label(self, **kwargs):
                pass

        ui_scene = self.scene("FIRST_LAST")
        ui_scene.sequence_editor = types.SimpleNamespace(strips=[])
        layout = FakeLayout()
        context = types.SimpleNamespace(scene=ui_scene, sequencer_scene=ui_scene)
        self.plugin.draw_custom_ui(layout, context)
        self.assertEqual(layout.search_labels, ["First Frame", "Last Frame"])
        self.assertEqual([op.action for op in layout.operators], ["siftq_select1", "siftq_select2"])

        class MappingClient:
            def data_uri_from_file(self, path, kind):
                return image_uri()

        with tempfile.TemporaryDirectory() as directory:
            first = os.path.join(directory, "first.png")
            last = os.path.join(directory, "last.png")
            for path in (first, last):
                with open(path, "wb") as handle:
                    handle.write(b"rendered-strip")
            scene = self.scene(
                "FIRST_LAST", siftq_ref_strip_1_path=first,
                siftq_ref_strip_2_path=last,
            )
            mapped = self.plugin.build_request(
                base.ModelInputs(prompt="transition"), scene, MappingClient()
            )
        self.assertEqual(
            [item.get("role") for item in mapped["content"][1:]],
            ["first_frame", "last_frame"],
        )

    def test_interactive_image_picker_uses_source_file_without_vse_render(self):
        """A timeline IMAGE picker is a file reference, not a VSE composite."""
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "portrait.png")
            with open(source, "wb") as handle:
                handle.write(b"original-portrait-bytes")

            strip = types.SimpleNamespace(type="IMAGE", name="portrait")
            helpers = types.ModuleType(PKG + ".utils.helpers")
            helpers.find_strip_by_name = lambda scene, name: strip
            helpers.get_strip_path = lambda value: source

            def unexpected_render(value):
                raise AssertionError("IMAGE picker must not render through the VSE")

            helpers.load_strip_as_pil = unexpected_render

            class RecordingClient:
                def data_uri_from_file(self, path, kind):
                    self.path = path
                    self.kind = kind
                    return "data:image/png;base64,b3JpZ2luYWw="

            client = RecordingClient()
            scene = types.SimpleNamespace(
                siftq_ref_strip_1_path="",
                siftq_ref_strip_1="portrait",
            )
            with mock.patch.dict(sys.modules, {PKG + ".utils.helpers": helpers}):
                uri, key = self.plugin._picker_image(scene, "siftq_ref_strip_1", client)

            self.assertEqual(client.path, source)
            self.assertEqual(client.kind, "image")
            self.assertEqual(uri, "data:image/png;base64,b3JpZ2luYWw=")
            self.assertEqual(key, "path:" + os.path.normcase(os.path.abspath(source)))

    def test_public_generate_path(self):
        output_dir = tempfile.TemporaryDirectory()
        destination = os.path.join(output_dir.name, "siftq_public.mp4")
        helper_mod = types.ModuleType(PKG + ".utils.helpers")
        helper_mod.clean_filename = lambda value: "siftq_public"
        helper_mod.solve_path = lambda value: destination
        sys.modules[PKG + ".utils.helpers"] = helper_mod

        inputs = base.ModelInputs(prompt="public path", image=FakeImage())
        phases = []
        inputs.phase_fn = phases.append
        inputs.progress_fn = lambda current, total: None
        inputs.should_cancel = lambda: False
        MockState.statuses = ["queued", "succeeded"]
        env = {
            "SIFTQ_API_KEY": "unit-key",
            "SIFTQ_BASE_URL": API_BASE,
            "SIFTQ_POLL_INTERVAL_SECONDS": "0.1",
            "SIFTQ_POLL_TIMEOUT_SECONDS": "2",
        }
        try:
            with mock.patch.dict(os.environ, env, clear=False):
                captured = io.StringIO()
                with contextlib.redirect_stdout(captured):
                    result = self.plugin.generate(
                        None, inputs, self.scene("FIRST_FRAME"), types.SimpleNamespace()
                    )
            self.assertEqual(result, destination)
            with open(result, "rb") as handle:
                self.assertEqual(handle.read(), MP4)
            self.assertIn("SiftQ:", inputs.usage_note)
            self.assertIn("Submitting to SiftQ", phases)
            self.assertIn("mode=FIRST_FRAME", captured.getvalue())
            self.assertIn("media_roles=['first_frame']", captured.getvalue())
            self.assertIn("ratio=adaptive", captured.getvalue())
        finally:
            output_dir.cleanup()

    def test_create_timeout_recovers_unique_new_task_without_resubmitting(self):
        now = int(time.time())

        class RecoveringClient:
            create_calls = 0
            list_calls = 0

            def list_tasks(self, **kwargs):
                self.list_calls += 1
                old = {"id": "old", "created_at": now - 60, "model": "MiniMax-H3",
                       "task_type": "generation", "modality": "video"}
                if self.list_calls == 1:
                    return {"items": [old], "total": 1}
                new = {"id": "recovered", "created_at": now, "model": "MiniMax-H3",
                       "task_type": "generation", "modality": "video",
                       "resolution": "768P", "duration": 5}
                return {"items": [new, old], "total": 2}

            def create_video(self, payload):
                self.create_calls += 1
                raise h3.SiftQError("lost response", delivery_uncertain=True)

        client = RecoveringClient()
        inputs = base.ModelInputs(prompt="safe recovery")
        phases = []
        inputs.phase_fn = phases.append
        inputs.should_cancel = lambda: False
        payload = h3.build_video_payload(prompt="safe recovery", ratio="16:9")
        task_id = self.plugin._create_video_safely(client, payload, inputs)
        self.assertEqual(task_id, "recovered")
        self.assertEqual(client.create_calls, 1)
        self.assertIn("Recovering SiftQ submission", phases)

    def test_create_timeout_refuses_ambiguous_or_unsnapshotted_recovery(self):
        now = int(time.time())

        class AmbiguousClient:
            create_calls = 0
            list_calls = 0

            def list_tasks(self, **kwargs):
                self.list_calls += 1
                if self.list_calls == 1:
                    return {"items": [], "total": 0}
                items = [
                    {"id": task_id, "created_at": now, "model": "MiniMax-H3",
                     "task_type": "generation", "modality": "video",
                     "resolution": "768P", "duration": 5}
                    for task_id in ("new-a", "new-b")
                ]
                return {"items": items, "total": 2}

            def create_video(self, payload):
                self.create_calls += 1
                raise h3.SiftQError("lost response", delivery_uncertain=True)

        inputs = base.ModelInputs(prompt="ambiguous")
        inputs.should_cancel = lambda: False
        payload = h3.build_video_payload(prompt="ambiguous", ratio="16:9")
        ambiguous = AmbiguousClient()
        with self.assertRaisesRegex(h3.SiftQError, "multiple new tasks"):
            self.plugin._create_video_safely(ambiguous, payload, inputs)
        self.assertEqual(ambiguous.create_calls, 1)

        class NoSnapshotClient:
            create_calls = 0

            def list_tasks(self, **kwargs):
                raise h3.SiftQError("list unavailable")

            def create_video(self, payload):
                self.create_calls += 1
                raise h3.SiftQError("lost response", delivery_uncertain=True)

        no_snapshot = NoSnapshotClient()
        with self.assertRaisesRegex(h3.SiftQError, "pre-submit task snapshot"):
            self.plugin._create_video_safely(no_snapshot, payload, inputs)
        self.assertEqual(no_snapshot.create_calls, 1)

    def test_plugin_requires_runtime_key_and_mode_inputs(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(h3.SiftQValidationError):
                self.plugin._client_from_env()
        disabled_bpy = types.SimpleNamespace(app=types.SimpleNamespace(online_access=False))
        with mock.patch.dict(sys.modules, {"bpy": disabled_bpy}):
            with mock.patch.dict(os.environ, {"SIFTQ_API_KEY": "unit-key"}):
                with self.assertRaisesRegex(h3.SiftQValidationError, "Online Access"):
                    self.plugin._client_from_env()
        client = h3.SiftQClient("key", API_BASE)
        with self.assertRaisesRegex(h3.SiftQValidationError, "requires a First Frame"):
            self.plugin.build_request(base.ModelInputs(prompt="x"), self.scene("FIRST_FRAME"), client)
        with tempfile.TemporaryDirectory() as directory:
            first = os.path.join(directory, "first.png")
            with open(first, "wb") as handle:
                handle.write(b"rendered-strip")
            with self.assertRaisesRegex(h3.SiftQValidationError, "both picker fields"):
                self.plugin.build_request(
                    base.ModelInputs(prompt="x"),
                    self.scene("FIRST_LAST", siftq_ref_strip_1_path=first),
                    types.SimpleNamespace(data_uri_from_file=lambda path, kind: image_uri()),
                )
        with self.assertRaisesRegex(h3.SiftQValidationError, "requires at least one"):
            self.plugin.build_request(base.ModelInputs(prompt="x"), self.scene("REFERENCE"), client)

        enabled_bpy = types.SimpleNamespace(app=types.SimpleNamespace(online_access=True))
        with mock.patch.dict(sys.modules, {"bpy": enabled_bpy}):
            with mock.patch.dict(os.environ, {"SIFTQ_API_KEY": "unit-key"}, clear=True):
                self.assertEqual(self.plugin._client_from_env().timeout, 180.0)
            with mock.patch.dict(os.environ, {
                "SIFTQ_API_KEY": "unit-key",
                "SIFTQ_REQUEST_TIMEOUT_SECONDS": "90",
            }, clear=True):
                self.assertEqual(self.plugin._client_from_env().timeout, 90.0)

    def test_explicit_picker_failures_are_not_silently_dropped(self):
        class MappingClient:
            def data_uri_from_file(self, path, kind):
                return {"image": image_uri(), "video": video_uri()}[kind]

        inputs = base.ModelInputs(prompt="reference")
        inputs.video_path = "selected-reference.mp4"
        unresolved = self.scene(
            "REFERENCE", siftq_ref_strip_1="Missing timeline image",
        )
        with self.assertRaisesRegex(h3.SiftQValidationError, "Could not resolve"):
            self.plugin.build_request(inputs, unresolved, MappingClient())

        missing_file = self.scene(
            "REFERENCE", siftq_ref_strip_1_path=os.path.join(
                tempfile.gettempdir(), "siftq-missing-reference.png"
            ),
        )
        with self.assertRaisesRegex(h3.SiftQValidationError, "missing file"):
            self.plugin.build_request(inputs, missing_file, MappingClient())


if __name__ == "__main__":
    try:
        unittest.main(verbosity=2)
    finally:
        SERVER.shutdown()
        SERVER.server_close()
