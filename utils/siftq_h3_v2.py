"""Independent SiftQ MiniMax-H3 V2 transport and contract helpers.

This module deliberately does not import or reuse pre-existing provider
helpers. Image inspection is stdlib-only; reference video/audio inspection uses
PyAV when those modes are selected.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from io import BytesIO


DEFAULT_BASE_URL = "https://siftq.com/api/minimax/"
MODEL_ID = "MiniMax-H3"
USER_AGENT = "Pallaidium-SiftQ/1.0"

CREATE_VIDEO_PATH = "v2/video_generation"
QUERY_VIDEO_PATH = "v2/query/video_generation"
CONTEXT_IR_PATH = "v2/h3_context_ir"

VALID_STATUSES = {"queued", "running", "succeeded", "failed", "cancelled"}
# The published H3 V2 contract calls the active state ``running``.  The live
# SiftQ service has also emitted ``processing`` for that same state, including
# from the task-list route.  Normalize only this observed response alias while
# keeping request filters and the rest of the public client contract strict.
RESPONSE_STATUS_ALIASES = {"processing": "running"}
VALID_RESOLUTIONS = {"768P", "2K"}
VALID_RATIOS = {"adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"}
VALID_ROLES = {
    "first_frame", "last_frame", "reference_image", "reference_video", "reference_audio",
}

MAX_REQUEST_BYTES = 64 * 1024 * 1024
MAX_JSON_RESPONSE_BYTES = 1024 * 1024
MAX_DOWNLOAD_BYTES = 1024 * 1024 * 1024
MAX_IMAGE_BYTES = 30 * 1024 * 1024
MAX_VIDEO_BYTES = 50 * 1024 * 1024
MAX_AUDIO_BYTES = 15 * 1024 * 1024

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
_VIDEO_EXTS = {".mp4", ".mov"}
_AUDIO_EXTS = {".wav", ".mp3"}
_IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}
_IMAGE_MIME_BY_EXT = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".webp": "image/webp", ".heic": "image/heic", ".heif": "image/heif",
}
_VIDEO_MIMES = {"video/mp4", "video/quicktime"}
_AUDIO_MIMES = {"audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3"}
_TYPE_FIELD = {
    "text": "text",
    "image_url": "image_url",
    "video_url": "video_url",
    "audio_url": "audio_url",
}


def _jpeg_dimensions(data: bytes) -> tuple[int, int]:
    if not data.startswith(b"\xff\xd8"):
        raise ValueError("invalid JPEG signature")
    position = 2
    sof_markers = {
        0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
        0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
    }
    while position + 4 <= len(data):
        if data[position] != 0xFF:
            position += 1
            continue
        while position < len(data) and data[position] == 0xFF:
            position += 1
        if position >= len(data):
            break
        marker = data[position]
        position += 1
        if marker in {0x01, 0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            continue
        if position + 2 > len(data):
            break
        segment_length = int.from_bytes(data[position:position + 2], "big")
        if segment_length < 2 or position + segment_length > len(data):
            break
        if marker in sof_markers and segment_length >= 7:
            height = int.from_bytes(data[position + 3:position + 5], "big")
            width = int.from_bytes(data[position + 5:position + 7], "big")
            return width, height
        position += segment_length
    raise ValueError("JPEG dimensions not found")


def _webp_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < 30 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        raise ValueError("invalid WebP signature")
    chunk = data[12:16]
    if chunk == b"VP8X":
        width = 1 + int.from_bytes(data[24:27], "little")
        height = 1 + int.from_bytes(data[27:30], "little")
        return width, height
    if chunk == b"VP8 " and data[23:26] == b"\x9d\x01\x2a":
        width = int.from_bytes(data[26:28], "little") & 0x3FFF
        height = int.from_bytes(data[28:30], "little") & 0x3FFF
        return width, height
    if chunk == b"VP8L" and data[20] == 0x2F:
        packed = int.from_bytes(data[21:25], "little")
        return (packed & 0x3FFF) + 1, ((packed >> 14) & 0x3FFF) + 1
    raise ValueError("unsupported WebP bitstream")


def _heif_dimensions(data: bytes) -> tuple[int, int]:
    # HEIC/HEIF stores the primary image dimensions in an Image Spatial
    # Extents (ispe) full box: version/flags, width, height. The property can
    # occur inside nested ISO-BMFF boxes, so locate the typed box directly and
    # verify its declared size before reading it.
    position = 0
    while True:
        marker = data.find(b"ispe", position)
        if marker < 4:
            raise ValueError("HEIF ispe box not found")
        box_start = marker - 4
        box_size = int.from_bytes(data[box_start:marker], "big")
        if box_size >= 20 and box_start + box_size <= len(data) and marker + 12 <= len(data):
            width = int.from_bytes(data[marker + 8:marker + 12], "big")
            height = int.from_bytes(data[marker + 12:marker + 16], "big")
            return width, height
        position = marker + 4


def _image_dimensions(data: bytes, ext: str) -> tuple[int, int]:
    if ext == ".png":
        if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
            raise ValueError("invalid PNG header")
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if ext in {".jpg", ".jpeg"}:
        return _jpeg_dimensions(data)
    if ext == ".webp":
        return _webp_dimensions(data)
    if ext in {".heic", ".heif"}:
        return _heif_dimensions(data)
    raise ValueError("unsupported image type")


def _bmff_boxes(data: bytes, start: int = 0, end: int | None = None):
    """Yield bounded ISO-BMFF boxes as (type, payload_start, box_end)."""
    limit = len(data) if end is None else min(end, len(data))
    position = max(0, start)
    while position + 8 <= limit:
        size = int.from_bytes(data[position:position + 4], "big")
        box_type = data[position + 4:position + 8]
        header = 8
        if size == 1:
            if position + 16 > limit:
                raise ValueError("truncated extended BMFF box")
            size = int.from_bytes(data[position + 8:position + 16], "big")
            header = 16
        elif size == 0:
            size = limit - position
        if size < header or position + size > limit:
            raise ValueError("invalid BMFF box size")
        yield box_type, position + header, position + size
        position += size


def _bmff_children(data: bytes, parent, wanted: bytes | None = None):
    children = list(_bmff_boxes(data, parent[1], parent[2]))
    return [box for box in children if wanted is None or box[0] == wanted]


def _fullbox_timescale_duration(data: bytes, box) -> tuple[int, int]:
    payload, end = box[1], box[2]
    if payload + 4 > end:
        raise ValueError("truncated BMFF full box")
    version = data[payload]
    if version == 0:
        if payload + 20 > end:
            raise ValueError("truncated BMFF duration")
        return (
            int.from_bytes(data[payload + 12:payload + 16], "big"),
            int.from_bytes(data[payload + 16:payload + 20], "big"),
        )
    if version == 1:
        if payload + 32 > end:
            raise ValueError("truncated BMFF duration")
        return (
            int.from_bytes(data[payload + 20:payload + 24], "big"),
            int.from_bytes(data[payload + 24:payload + 32], "big"),
        )
    raise ValueError("unsupported BMFF full-box version")


def _mp4_metadata(data: bytes) -> dict:
    top = list(_bmff_boxes(data))
    if not top or top[0][0] != b"ftyp":
        raise ValueError("invalid MP4/MOV signature")
    moov = next((box for box in top if box[0] == b"moov"), None)
    if moov is None:
        raise ValueError("MP4/MOV is missing moov metadata")

    movie_seconds = 0.0
    mvhd = next(iter(_bmff_children(data, moov, b"mvhd")), None)
    if mvhd is not None:
        timescale, duration = _fullbox_timescale_duration(data, mvhd)
        movie_seconds = float(duration) / float(timescale) if timescale else 0.0

    video = None
    audio_codecs = []
    for trak in _bmff_children(data, moov, b"trak"):
        tkhd = next(iter(_bmff_children(data, trak, b"tkhd")), None)
        mdia = next(iter(_bmff_children(data, trak, b"mdia")), None)
        if mdia is None:
            continue
        hdlr = next(iter(_bmff_children(data, mdia, b"hdlr")), None)
        mdhd = next(iter(_bmff_children(data, mdia, b"mdhd")), None)
        if hdlr is None or hdlr[1] + 12 > hdlr[2]:
            continue
        handler = data[hdlr[1] + 8:hdlr[1] + 12]
        timescale = duration = 0
        if mdhd is not None:
            timescale, duration = _fullbox_timescale_duration(data, mdhd)

        minf = next(iter(_bmff_children(data, mdia, b"minf")), None)
        stbl = next(iter(_bmff_children(data, minf, b"stbl")), None) if minf else None
        stsd = next(iter(_bmff_children(data, stbl, b"stsd")), None) if stbl else None
        codec = b""
        if stsd is not None and stsd[1] + 16 <= stsd[2]:
            entry_start = stsd[1] + 8
            entry_size = int.from_bytes(data[entry_start:entry_start + 4], "big")
            if entry_size >= 8 and entry_start + entry_size <= stsd[2]:
                codec = data[entry_start + 4:entry_start + 8]

        if handler == b"soun":
            audio_codecs.append(codec)
            continue
        if handler != b"vide":
            continue

        width = height = 0
        if tkhd is not None and tkhd[2] - tkhd[1] >= 8:
            width = int.from_bytes(data[tkhd[2] - 8:tkhd[2] - 4], "big") >> 16
            height = int.from_bytes(data[tkhd[2] - 4:tkhd[2]], "big") >> 16

        fps = 0.0
        stts = next(iter(_bmff_children(data, stbl, b"stts")), None) if stbl else None
        if stts is not None and stts[1] + 8 <= stts[2] and timescale:
            entry_count = int.from_bytes(data[stts[1] + 4:stts[1] + 8], "big")
            position = stts[1] + 8
            sample_count = total_ticks = 0
            for _ in range(entry_count):
                if position + 8 > stts[2]:
                    raise ValueError("truncated MP4 stts table")
                count = int.from_bytes(data[position:position + 4], "big")
                delta = int.from_bytes(data[position + 4:position + 8], "big")
                sample_count += count
                total_ticks += count * delta
                position += 8
            if total_ticks:
                fps = float(sample_count * timescale) / float(total_ticks)
        track_seconds = float(duration) / float(timescale) if timescale else 0.0
        video = {
            "codec": codec,
            "width": width,
            "height": height,
            "fps": fps,
            "duration": track_seconds or movie_seconds,
        }

    if video is None:
        raise ValueError("MP4/MOV has no video track")
    video["duration"] = video["duration"] or movie_seconds
    video["audio_codecs"] = audio_codecs
    return video


def _mp3_duration(data: bytes) -> float:
    position = 0
    if data.startswith(b"ID3") and len(data) >= 10:
        size_bytes = data[6:10]
        if any(value & 0x80 for value in size_bytes):
            raise ValueError("invalid MP3 ID3 size")
        tag_size = sum(value << shift for value, shift in zip(size_bytes, (21, 14, 7, 0)))
        position = 10 + tag_size + (10 if data[5] & 0x10 else 0)

    bitrate_v1 = (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0)
    bitrate_v2 = (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0)
    sample_rates = (44100, 48000, 32000)
    seconds = 0.0
    frames = 0
    while position + 4 <= len(data):
        header = int.from_bytes(data[position:position + 4], "big")
        if header & 0xFFE00000 != 0xFFE00000:
            position += 1
            continue
        version_bits = (header >> 19) & 0x3
        layer_bits = (header >> 17) & 0x3
        bitrate_index = (header >> 12) & 0xF
        sample_index = (header >> 10) & 0x3
        padding = (header >> 9) & 0x1
        if version_bits == 1 or layer_bits != 1 or sample_index == 3:
            position += 1
            continue
        mpeg1 = version_bits == 3
        bitrate = (bitrate_v1 if mpeg1 else bitrate_v2)[bitrate_index] * 1000
        divisor = 1 if mpeg1 else (2 if version_bits == 2 else 4)
        sample_rate = sample_rates[sample_index] // divisor
        if not bitrate or not sample_rate:
            position += 1
            continue
        frame_size = ((144 if mpeg1 else 72) * bitrate // sample_rate) + padding
        if frame_size < 4 or position + frame_size > len(data):
            break
        seconds += float(1152 if mpeg1 else 576) / float(sample_rate)
        frames += 1
        position += frame_size
    if not frames:
        raise ValueError("MP3 frame header not found")
    return seconds


def _inspect_timed_media_stdlib(path: str, kind: str) -> None:
    ext = os.path.splitext(path)[1].lower()
    if kind == "video":
        with open(path, "rb") as handle:
            metadata = _mp4_metadata(handle.read(MAX_VIDEO_BYTES + 1))
        if metadata["codec"] not in {b"avc1", b"avc3", b"hvc1", b"hev1"}:
            raise SiftQValidationError("SiftQ reference video codec must be H.264 or H.265.")
        if any(codec not in {b"mp4a", b".mp3"} for codec in metadata["audio_codecs"]):
            raise SiftQValidationError("SiftQ reference video audio codec must be AAC or MP3.")
        _validate_dimensions(metadata["width"], metadata["height"], "SiftQ video")
        if not 23.976 <= metadata["fps"] <= 60.0:
            raise SiftQValidationError("SiftQ reference video FPS must be 23.976–60.")
        if not 2.0 <= metadata["duration"] <= 15.0:
            raise SiftQValidationError("SiftQ reference video duration must be 2–15 seconds.")
        return

    if ext == ".wav":
        import wave
        with wave.open(path, "rb") as source:
            rate = source.getframerate()
            duration = float(source.getnframes()) / float(rate) if rate else 0.0
    else:
        with open(path, "rb") as handle:
            duration = _mp3_duration(handle.read(MAX_AUDIO_BYTES + 1))
    if not 2.0 <= duration <= 15.0:
        raise SiftQValidationError("SiftQ reference audio duration must be 2–15 seconds.")


class SiftQError(RuntimeError):
    """Safe provider/transport error with structured metadata."""

    def __init__(self, message, *, status_code=None, error_type="", http_code="", request_id="",
                 delivery_uncertain=False):
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type or ""
        self.http_code = str(http_code or "")
        self.request_id = request_id or ""
        # True means the request body may have reached the provider even though
        # the client did not receive a usable create response. Callers must not
        # blindly retry a billable request in this state.
        self.delivery_uncertain = bool(delivery_uncertain)


class SiftQValidationError(ValueError):
    """Raised before a request when local inputs violate the H3 V2 contract."""


def normalize_base_url(base_url: str) -> str:
    value = (base_url or "").strip()
    if not value:
        raise SiftQValidationError("SiftQ base URL is empty.")
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise SiftQValidationError("SiftQ base URL must be an absolute HTTP(S) URL.")
    try:
        parsed.port
    except ValueError as exc:
        raise SiftQValidationError("SiftQ base URL contains an invalid port.") from exc
    if parsed.username or parsed.password:
        raise SiftQValidationError("SiftQ base URL must not contain credentials.")
    if parsed.query or parsed.fragment:
        raise SiftQValidationError("SiftQ base URL must not contain a query or fragment.")
    return value.rstrip("/") + "/"


def join_url(base_url: str, suffix: str) -> str:
    return normalize_base_url(base_url) + str(suffix).lstrip("/")


def route_urls(base_url: str = DEFAULT_BASE_URL, task_id: str = "task id") -> dict:
    encoded = urllib.parse.quote(str(task_id), safe="")
    return {
        "create_video": join_url(base_url, CREATE_VIDEO_PATH),
        "query_task": join_url(base_url, f"{QUERY_VIDEO_PATH}/{encoded}"),
        "list_tasks": join_url(base_url, QUERY_VIDEO_PATH),
        "delete_task": join_url(base_url, f"{CREATE_VIDEO_PATH}/{encoded}"),
        "create_context_ir": join_url(base_url, CONTEXT_IR_PATH),
    }


def _validate_dimensions(width: int, height: int, label: str) -> None:
    if not (256 <= int(width) <= 5760 and 256 <= int(height) <= 5760):
        raise SiftQValidationError(f"{label} dimensions must each be 256–5760 pixels.")
    ratio = float(width) / float(height)
    if not 0.4 <= ratio <= 2.5:
        raise SiftQValidationError(f"{label} aspect ratio must be between 0.4 and 2.5.")


def _validate_media_location(value: str, kind: str) -> None:
    if not isinstance(value, str) or not value:
        raise SiftQValidationError(f"{kind} media URL must be a non-empty string.")
    if value.startswith("mm_file://"):
        if not value[len("mm_file://"):]:
            raise SiftQValidationError("mm_file reference is missing a file ID.")
        return
    if value.startswith("data:"):
        match = re.match(r"^data:([a-z0-9.+-]+/[a-z0-9.+-]+);base64,(.+)$", value, re.DOTALL)
        if not match:
            raise SiftQValidationError(f"Malformed {kind} data URI.")
        mime = match.group(1)
        allowed = {"image": _IMAGE_MIMES, "video": _VIDEO_MIMES, "audio": _AUDIO_MIMES}[kind]
        if mime not in allowed:
            raise SiftQValidationError(f"Unsupported {kind} data-URI content type: {mime}.")
        try:
            decoded = base64.b64decode(match.group(2), validate=True)
        except Exception as exc:
            raise SiftQValidationError(f"Malformed base64 in {kind} data URI.") from exc
        max_bytes = {
            "image": MAX_IMAGE_BYTES, "video": MAX_VIDEO_BYTES, "audio": MAX_AUDIO_BYTES,
        }[kind]
        if not decoded or len(decoded) > max_bytes:
            raise SiftQValidationError(
                f"SiftQ {kind} data URI is empty or exceeds its per-file size limit."
            )
        return
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise SiftQValidationError(f"{kind} media location must be HTTP(S), mm_file, or data URI.")
    try:
        parsed.port
    except ValueError as exc:
        raise SiftQValidationError(f"{kind} media URL contains an invalid port.") from exc
    if parsed.username or parsed.password:
        raise SiftQValidationError(f"{kind} media URL must not contain credentials.")


def validate_content(content: list, ratio: str) -> None:
    if not isinstance(content, list) or not content:
        raise SiftQValidationError("SiftQ content must be a non-empty list.")
    if ratio not in VALID_RATIOS:
        raise SiftQValidationError(f"Unsupported SiftQ ratio: {ratio!r}.")

    counts = {role: 0 for role in VALID_ROLES}
    has_prompt = False
    for item in content:
        if not isinstance(item, dict):
            raise SiftQValidationError("Every SiftQ content item must be an object.")
        item_type = item.get("type")
        expected_field = _TYPE_FIELD.get(item_type)
        if expected_field is None:
            raise SiftQValidationError(f"Unsupported SiftQ content type: {item_type!r}.")
        if expected_field not in item:
            raise SiftQValidationError(f"Content type {item_type!r} requires {expected_field!r}.")
        if item_type == "text":
            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                raise SiftQValidationError("SiftQ requires a non-empty text prompt.")
            if len(text) > 7000:
                raise SiftQValidationError("SiftQ prompt items are limited to 7000 characters.")
            if item.get("role") is not None:
                raise SiftQValidationError("Text content must not declare a media role.")
            has_prompt = True
            continue

        value = item.get(expected_field)
        if not isinstance(value, dict) or not isinstance(value.get("url"), str):
            raise SiftQValidationError(f"{expected_field} must be an object containing url.")
        kind = item_type.split("_", 1)[0]
        _validate_media_location(value["url"], kind)
        role = item.get("role")
        if role is None and item_type == "image_url":
            role = "first_frame"
        if role not in VALID_ROLES:
            raise SiftQValidationError(f"Unsupported or missing SiftQ media role: {role!r}.")
        if role.startswith("reference_") and role != f"reference_{kind}":
            raise SiftQValidationError(f"Role {role!r} does not match {item_type!r}.")
        if role in {"first_frame", "last_frame"} and item_type != "image_url":
            raise SiftQValidationError(f"Role {role!r} requires image_url content.")
        counts[role] += 1

    if not has_prompt:
        raise SiftQValidationError("SiftQ content must include a non-empty text prompt.")
    if counts["first_frame"] > 1 or counts["last_frame"] > 1:
        raise SiftQValidationError("SiftQ accepts at most one first frame and one last frame.")
    if counts["last_frame"] and not counts["first_frame"]:
        raise SiftQValidationError("A SiftQ last frame requires a first frame.")
    if counts["reference_image"] > 9:
        raise SiftQValidationError("SiftQ accepts at most 9 reference images.")
    if counts["reference_video"] > 3 or counts["reference_audio"] > 3:
        raise SiftQValidationError("SiftQ accepts at most 3 reference videos and 3 reference audio files.")

    has_frames = bool(counts["first_frame"] or counts["last_frame"])
    has_references = any(counts[r] for r in ("reference_image", "reference_video", "reference_audio"))
    if has_frames and has_references:
        raise SiftQValidationError("First/last-frame roles cannot be combined with reference roles.")
    if has_frames and ratio != "adaptive":
        raise SiftQValidationError("First/last-frame requests must use ratio='adaptive'.")
    if not has_frames and not has_references and ratio == "adaptive":
        raise SiftQValidationError("Text-only requests require a concrete ratio.")


def build_video_payload(*, prompt: str, media: list | None = None, resolution: str = "768P",
                        duration: int = 5, ratio: str = "16:9") -> dict:
    if resolution not in VALID_RESOLUTIONS:
        raise SiftQValidationError(f"Unsupported SiftQ resolution: {resolution!r}.")
    if isinstance(duration, bool) or not isinstance(duration, int) or not 4 <= duration <= 15:
        raise SiftQValidationError("SiftQ duration must be an integer from 4 through 15 seconds.")
    content = [{"type": "text", "text": prompt}] + list(media or [])
    validate_content(content, ratio)
    payload = {
        "model": MODEL_ID,
        "content": content,
        "resolution": resolution,
        "duration": duration,
        "ratio": ratio,
    }
    if len(json.dumps(payload, separators=(",", ":")).encode("utf-8")) > MAX_REQUEST_BYTES:
        raise SiftQValidationError("SiftQ request body exceeds the 64 MB limit.")
    return payload


def build_context_ir_payload(*, prompt: str, media: list | None = None,
                             duration: int = 5, ratio: str = "16:9") -> dict:
    video = build_video_payload(
        prompt=prompt, media=media, resolution="768P", duration=duration, ratio=ratio,
    )
    video.pop("resolution")
    return video


def callback_challenge_response(payload: dict) -> dict:
    challenge = payload.get("challenge") if isinstance(payload, dict) else None
    if not isinstance(challenge, str) or not challenge:
        raise SiftQValidationError("SiftQ callback verification requires a non-empty challenge.")
    return {"challenge": challenge}


def _url_origin(value: str) -> tuple[str, str, int | None]:
    parsed = urllib.parse.urlsplit(value)
    try:
        port = parsed.port
    except ValueError as exc:
        raise urllib.error.URLError("invalid SiftQ redirect URL") from exc
    if port is None:
        port = 443 if parsed.scheme.lower() == "https" else 80
    return parsed.scheme.lower(), (parsed.hostname or "").lower(), port


class _SameOriginRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Prevent an authenticated API redirect from forwarding the bearer key."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if _url_origin(req.full_url) != _url_origin(newurl):
            raise urllib.error.URLError("cross-origin SiftQ API redirect blocked")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class SiftQClient:
    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL, timeout: float = 60.0):
        self.api_key = (api_key or "").strip()
        if not self.api_key:
            raise SiftQValidationError("SIFTQ_API_KEY is missing.")
        self.base_url = normalize_base_url(base_url)
        self.timeout = float(timeout)
        if self.timeout <= 0:
            raise SiftQValidationError("SiftQ request timeout must be positive.")
        self._api_opener = urllib.request.build_opener(_SameOriginRedirectHandler())

    def url(self, suffix: str) -> str:
        return join_url(self.base_url, suffix)

    def routes(self, task_id: str = "task id") -> dict:
        return route_urls(self.base_url, task_id)

    def _safe_message(self, value) -> str:
        text = str(value or "provider request failed").replace(self.api_key, "[redacted]")
        text = re.sub(r"(?i)bearer\s+[^\s,;]+", "Bearer [redacted]", text)
        text = re.sub(r"https?://[^\s]+", "[redacted-url]", text)
        text = re.sub(r"(?i)(api[_-]?key=)[^&\s]+", r"\1[redacted]", text)
        return text[:300]

    def _provider_error(self, status_code: int, raw: bytes) -> SiftQError:
        error_type = ""
        http_code = ""
        message = f"SiftQ request failed with HTTP {status_code}."
        request_id = ""
        try:
            envelope = json.loads(raw.decode("utf-8")) if raw else {}
            err = envelope.get("error") if isinstance(envelope, dict) else None
            if isinstance(err, dict):
                error_type = str(err.get("type") or "")
                message = self._safe_message(err.get("message") or message)
                http_code = str(err.get("http_code") or "")
            request_id = str(envelope.get("request_id") or "") if isinstance(envelope, dict) else ""
        except Exception:
            pass
        label = f"SiftQ {error_type}" if error_type else "SiftQ error"
        return SiftQError(
            f"{label} (HTTP {status_code}): {message}",
            status_code=status_code, error_type=error_type,
            http_code=http_code, request_id=request_id,
        )

    def _request_json(self, method: str, url: str, payload: dict | None = None) -> dict:
        data = None
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": USER_AGENT,
        }
        if payload is not None:
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            if len(data) > MAX_REQUEST_BYTES:
                raise SiftQValidationError("SiftQ request body exceeds the 64 MB limit.")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with self._api_opener.open(req, timeout=self.timeout) as response:
                raw = response.read(MAX_JSON_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as exc:
            raise self._provider_error(exc.code, exc.read(MAX_JSON_RESPONSE_BYTES)) from exc
        except urllib.error.URLError as exc:
            raise SiftQError(
                f"SiftQ connection failed: {self._safe_message(exc.reason)}",
                delivery_uncertain=True,
            ) from exc
        except TimeoutError as exc:
            raise SiftQError(
                "SiftQ request timed out while waiting for a response.",
                delivery_uncertain=True,
            ) from exc
        except OSError as exc:
            raise SiftQError(
                f"SiftQ connection failed: {self._safe_message(exc)}",
                delivery_uncertain=True,
            ) from exc
        if len(raw) > MAX_JSON_RESPONSE_BYTES:
            raise SiftQError("SiftQ returned an oversized JSON response.")
        try:
            result = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception as exc:
            raise SiftQError("SiftQ returned malformed JSON.") from exc
        if not isinstance(result, dict):
            raise SiftQError("SiftQ returned an unexpected non-object response.")
        return result

    @staticmethod
    def _validate_payload(payload: dict, *, context_ir: bool = False) -> None:
        if not isinstance(payload, dict) or payload.get("model") != MODEL_ID:
            raise SiftQValidationError(f"SiftQ model must be exactly {MODEL_ID!r}.")
        if context_ir:
            if "resolution" in payload:
                raise SiftQValidationError("H3 Context-IR requests must omit resolution.")
        elif payload.get("resolution") not in VALID_RESOLUTIONS:
            raise SiftQValidationError("SiftQ video requests require resolution 768P or 2K.")
        duration = payload.get("duration")
        if isinstance(duration, bool) or not isinstance(duration, int) or not 4 <= duration <= 15:
            raise SiftQValidationError("SiftQ duration must be an integer from 4 through 15 seconds.")
        validate_content(payload.get("content"), payload.get("ratio"))

    def create_video(self, payload: dict) -> str:
        self._validate_payload(payload)
        response = self._request_json("POST", self.url(CREATE_VIDEO_PATH), payload)
        task_id = response.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise SiftQError(
                "SiftQ create-video response is missing task_id.",
                delivery_uncertain=True,
            )
        return task_id

    def create_context_ir(self, payload: dict) -> str:
        self._validate_payload(payload, context_ir=True)
        response = self._request_json("POST", self.url(CONTEXT_IR_PATH), payload)
        task_id = response.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise SiftQError("SiftQ Context-IR response is missing task_id.")
        return task_id

    @staticmethod
    def _validate_output_url(value: str) -> None:
        parsed = urllib.parse.urlsplit(value or "")
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise SiftQError("SiftQ succeeded but returned an invalid output URL.")
        try:
            parsed.port
        except ValueError as exc:
            raise SiftQError("SiftQ succeeded but returned an invalid output URL.") from exc
        if parsed.username or parsed.password:
            raise SiftQError("SiftQ output URL must not contain user credentials.")

    def parse_task(self, response: dict) -> dict:
        task = response.get("task") if isinstance(response, dict) else None
        if not isinstance(task, dict):
            raise SiftQError("SiftQ query response is missing task.")
        task = dict(task)
        task_id = task.get("id")
        raw_status = task.get("status")
        status = RESPONSE_STATUS_ALIASES.get(raw_status, raw_status)
        if not isinstance(task_id, str) or not task_id:
            raise SiftQError("SiftQ task is missing id.")
        if status not in VALID_STATUSES:
            raise SiftQError(f"SiftQ task returned an unexpected status: {raw_status!r}.")
        task["status"] = status
        if status == "succeeded":
            content = task.get("content")
            if not isinstance(content, dict):
                raise SiftQError("SiftQ succeeded task is missing content.")
            if task.get("task_type") == "h3_context_ir":
                if task.get("modality") != "text":
                    raise SiftQError("SiftQ Context-IR task returned an unexpected modality.")
                if not isinstance(content.get("prompt"), str) or not content["prompt"].strip():
                    raise SiftQError("SiftQ Context-IR task succeeded without content.prompt.")
            else:
                if task.get("task_type") not in {"generation", "regeneration"}:
                    raise SiftQError("SiftQ video task returned an unexpected task_type.")
                if task.get("modality") != "video":
                    raise SiftQError("SiftQ video task returned an unexpected modality.")
                url = content.get("url")
                if not isinstance(url, str) or not url:
                    raise SiftQError("SiftQ video task succeeded without content.url.")
                self._validate_output_url(url)
        return task

    def query_task(self, task_id: str) -> dict:
        if not isinstance(task_id, str) or not task_id:
            raise SiftQValidationError("SiftQ task ID must be non-empty.")
        encoded = urllib.parse.quote(task_id, safe="")
        response = self._request_json("GET", self.url(f"{QUERY_VIDEO_PATH}/{encoded}"))
        return self.parse_task(response)

    def list_tasks(self, *, page_num=None, page_size=None, status=None, task_ids=None,
                   model=None, task_type=None) -> dict:
        pairs = []
        for key, value in (("page_num", page_num), ("page_size", page_size)):
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise SiftQValidationError(f"{key} must be a positive integer.")
                pairs.append((key, value))
        if status is not None:
            if status not in VALID_STATUSES:
                raise SiftQValidationError("Invalid SiftQ task status filter.")
            pairs.append(("filter.status", status))
        for task_id in task_ids or []:
            if not isinstance(task_id, str) or not task_id:
                raise SiftQValidationError("filter.task_ids entries must be non-empty strings.")
            pairs.append(("filter.task_ids", task_id))
        if model is not None:
            pairs.append(("filter.model", model))
        if task_type is not None:
            if task_type not in {"generation", "h3_context_ir", "regeneration"}:
                raise SiftQValidationError("Invalid SiftQ task_type filter.")
            pairs.append(("filter.task_type", task_type))
        url = self.url(QUERY_VIDEO_PATH)
        if pairs:
            url += "?" + urllib.parse.urlencode(pairs, doseq=True)
        response = self._request_json("GET", url)
        items = response.get("items")
        total = response.get("total")
        if not isinstance(items, list) or isinstance(total, bool) or not isinstance(total, int):
            raise SiftQError("SiftQ list response must contain items and integer total.")
        parsed = [self.parse_task({"task": item}) for item in items]
        return {"items": parsed, "total": total}

    def delete_task(self, task_id: str) -> dict:
        if not isinstance(task_id, str) or not task_id:
            raise SiftQValidationError("SiftQ task ID must be non-empty.")
        encoded = urllib.parse.quote(task_id, safe="")
        response = self._request_json("DELETE", self.url(f"{CREATE_VIDEO_PATH}/{encoded}"))
        if response.get("task_id") != task_id:
            raise SiftQError("SiftQ delete response returned a mismatched task_id.")
        if response.get("action") not in {"cancelled", "deleted"}:
            raise SiftQError("SiftQ delete response returned an unexpected action.")
        if response.get("status") not in {"cancelled", "deleted"}:
            raise SiftQError("SiftQ delete response returned an unexpected status.")
        return response

    def poll_task(self, task_id: str, *, max_wait: float = 3600.0, interval: float = 2.0,
                  should_cancel=None, phase_fn=None, progress_fn=None,
                  sleep_fn=time.sleep, clock_fn=time.monotonic) -> dict:
        if max_wait <= 0 or interval < 0:
            raise SiftQValidationError("SiftQ polling limits must be positive.")
        deadline = clock_fn() + max_wait
        while True:
            task = self.query_task(task_id)
            status = task["status"]
            if should_cancel is not None and should_cancel():
                if status == "queued":
                    try:
                        self.delete_task(task_id)
                    except SiftQError:
                        pass
                raise KeyboardInterrupt(
                    "SiftQ generation cancelled locally; a running upstream task may continue."
                )
            if status == "succeeded":
                if progress_fn:
                    progress_fn(100, 100)
                return task
            if status == "failed":
                err = task.get("error") if isinstance(task.get("error"), dict) else {}
                code = str(err.get("code") or "unknown")
                message = self._safe_message(err.get("message") or "generation failed")
                raise SiftQError(f"SiftQ task failed ({code}): {message}")
            if status == "cancelled":
                raise SiftQError("SiftQ task was cancelled upstream.")
            if phase_fn:
                phase_fn("Queued at SiftQ" if status == "queued" else "Generating at SiftQ")
            if progress_fn:
                progress_fn(10 if status == "queued" else 50, 100)
            if clock_fn() >= deadline:
                raise SiftQError(f"SiftQ task timed out after {max_wait:.0f} seconds.")
            sleep_fn(interval)

    def pil_image_to_data_uri(self, image) -> str:
        if image is None or not hasattr(image, "size"):
            raise SiftQValidationError("SiftQ image input is missing or invalid.")
        width, height = image.size
        _validate_dimensions(width, height, "SiftQ image")
        buf = BytesIO()
        converted = image.convert("RGB") if hasattr(image, "convert") else image
        converted.save(buf, format="PNG")
        data = buf.getvalue()
        if len(data) > MAX_IMAGE_BYTES:
            raise SiftQValidationError("SiftQ image exceeds the 30 MB limit.")
        return "data:image/png;base64," + base64.b64encode(data).decode("ascii")

    def data_uri_from_file(self, path: str, kind: str) -> str:
        if kind not in {"image", "video", "audio"}:
            raise SiftQValidationError(f"Unsupported media kind: {kind!r}.")
        if not path or not os.path.isfile(path):
            raise SiftQValidationError(f"SiftQ {kind} reference file does not exist.")
        ext = os.path.splitext(path)[1].lower()
        allowed_exts = {"image": _IMAGE_EXTS, "video": _VIDEO_EXTS, "audio": _AUDIO_EXTS}[kind]
        max_bytes = {"image": MAX_IMAGE_BYTES, "video": MAX_VIDEO_BYTES, "audio": MAX_AUDIO_BYTES}[kind]
        if ext not in allowed_exts:
            raise SiftQValidationError(f"Unsupported SiftQ {kind} file type: {ext or '(none)'}.")
        size = os.path.getsize(path)
        if size <= 0 or size > max_bytes:
            raise SiftQValidationError(f"SiftQ {kind} file is empty or exceeds its size limit.")
        if kind == "image":
            try:
                with open(path, "rb") as handle:
                    data = handle.read(max_bytes + 1)
                width, height = _image_dimensions(data, ext)
                _validate_dimensions(width, height, "SiftQ image")
            except SiftQValidationError:
                raise
            except Exception as exc:
                raise SiftQValidationError("Could not inspect SiftQ reference image.") from exc
        else:
            self._inspect_timed_media(path, kind)
        mime = (_IMAGE_MIME_BY_EXT.get(ext) if kind == "image" else None) or (mimetypes.guess_type(path)[0] or {
            "image": "image/png", "video": "video/mp4", "audio": "audio/wav",
        }[kind]).lower()
        if kind == "audio" and mime == "audio/x-wav":
            mime = "audio/wav"
        if kind != "image":
            with open(path, "rb") as handle:
                data = handle.read(max_bytes + 1)
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    @staticmethod
    def _inspect_timed_media(path: str, kind: str) -> None:
        try:
            import av
        except ImportError:
            try:
                _inspect_timed_media_stdlib(path, kind)
                return
            except SiftQValidationError:
                raise
            except Exception as exc:
                raise SiftQValidationError(
                    f"Could not inspect SiftQ reference {kind}."
                ) from exc
        try:
            with av.open(path) as container:
                duration = float(container.duration or 0) / float(av.time_base)
                if duration <= 0:
                    stream_durations = [
                        float(stream.duration * stream.time_base)
                        for stream in container.streams
                        if stream.duration is not None and stream.time_base is not None
                    ]
                    duration = max(stream_durations, default=0.0)
                if not 2.0 <= duration <= 15.0:
                    raise SiftQValidationError(
                        f"SiftQ reference {kind} duration must be 2–15 seconds."
                    )
                if kind == "video":
                    stream = next((s for s in container.streams if s.type == "video"), None)
                    if stream is None:
                        raise SiftQValidationError("SiftQ reference video has no video stream.")
                    _validate_dimensions(stream.width, stream.height, "SiftQ video")
                    fps = float(stream.average_rate) if stream.average_rate else 0.0
                    if not 23.976 <= fps <= 60.0:
                        raise SiftQValidationError("SiftQ reference video FPS must be 23.976–60.")
                    codec = (getattr(stream.codec_context, "name", "") or "").lower()
                    if codec and codec not in {"h264", "hevc", "h265"}:
                        raise SiftQValidationError("SiftQ reference video codec must be H.264 or H.265.")
                    for audio_stream in (s for s in container.streams if s.type == "audio"):
                        audio_codec = (getattr(audio_stream.codec_context, "name", "") or "").lower()
                        if audio_codec and audio_codec not in {"aac", "mp3", "mp3float"}:
                            raise SiftQValidationError(
                                "SiftQ reference video audio codec must be AAC or MP3."
                            )
                else:
                    stream = next((s for s in container.streams if s.type == "audio"), None)
                    if stream is None:
                        raise SiftQValidationError("SiftQ reference audio has no audio stream.")
                    codec = (getattr(stream.codec_context, "name", "") or "").lower()
                    ext = os.path.splitext(path)[1].lower()
                    valid_codec = codec.startswith("pcm_") if ext == ".wav" else codec.startswith("mp3")
                    if codec and not valid_codec:
                        raise SiftQValidationError("SiftQ reference audio must be WAV or MP3.")
        except SiftQValidationError:
            raise
        except Exception as exc:
            raise SiftQValidationError(f"Could not inspect SiftQ reference {kind}.") from exc

    def download_video(self, url: str, destination: str) -> str:
        self._validate_output_url(url)
        directory = os.path.dirname(destination) or "."
        os.makedirs(directory, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=".siftq-", suffix=".part", dir=directory)
        os.close(fd)
        try:
            request = urllib.request.Request(
                url,
                headers={"Accept": "video/mp4", "User-Agent": USER_AGENT},
                method="GET",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response, \
                        open(temp_path, "wb") as output:
                    content_type = (response.headers.get("Content-Type", "") or "").split(";", 1)[0].lower()
                    if content_type not in {"video/mp4", "application/octet-stream"}:
                        raise SiftQError("SiftQ output download returned an invalid content type.")
                    total = 0
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > MAX_DOWNLOAD_BYTES:
                            raise SiftQError("SiftQ output download exceeded the safety limit.")
                        output.write(chunk)
            except urllib.error.HTTPError as exc:
                raise SiftQError(
                    f"SiftQ output download failed with HTTP {exc.code}.",
                    status_code=exc.code,
                ) from exc
            except urllib.error.URLError as exc:
                raise SiftQError("SiftQ output download failed.") from exc
            except TimeoutError as exc:
                raise SiftQError("SiftQ output download timed out.") from exc
            except OSError as exc:
                raise SiftQError("SiftQ output download failed.") from exc
            with open(temp_path, "rb") as handle:
                header = handle.read(16)
            if len(header) < 12 or header[4:8] != b"ftyp":
                raise SiftQError("SiftQ output is empty or is not an MP4 file.")
            os.replace(temp_path, destination)
            return destination
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def download_task_video(self, task_id: str, task: dict, destination: str) -> str:
        """Download once, refreshing a time-limited task URL after expiry."""
        content = task.get("content") if isinstance(task, dict) else None
        url = content.get("url") if isinstance(content, dict) else ""
        try:
            return self.download_video(url, destination)
        except SiftQError as exc:
            if exc.status_code not in {401, 403, 404}:
                raise
        refreshed = self.query_task(task_id)
        if refreshed.get("status") != "succeeded":
            raise SiftQError("SiftQ output URL expired and the refreshed task is not succeeded.")
        return self.download_video(refreshed["content"]["url"], destination)


def usage_note(task: dict) -> str:
    usage = task.get("usage") if isinstance(task, dict) else None
    if not isinstance(usage, dict):
        return ""
    parts = []
    for key, label in (
        ("total_seconds", "total"), ("input_seconds", "input"),
        ("output_seconds", "output"), ("input_image_count", "images"),
    ):
        value = usage.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            suffix = "s" if "seconds" in key else ""
            parts.append(f"{label} {value:g}{suffix}")
    return "SiftQ: " + ", ".join(parts) if parts else ""
