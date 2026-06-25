"""Shared stdlib HTTP plumbing for the reference adapters.

The example adapters (``comfyui_adapter.py``, ``fal_adapter.py``) used to depend
on ``fastapi`` / ``uvicorn`` / ``httpx``. They are now **stdlib-only** so the
add-on can launch them with Blender's own Python — no ``pip install`` step. This
module holds the bits all of them share:

  * a tiny :class:`BaseAdapterHandler` (JSON / binary replies, body reading,
    multipart parsing) built on ``http.server``;
  * outbound HTTP helpers over ``urllib`` (``get_json`` / ``post_json`` /
    ``get_bytes`` / ``post_multipart``);
  * :func:`serve` to run the threaded server from a ``--port`` CLI.

Nothing here is provider-specific; the ComfyUI / fal mapping lives in each
adapter. The mock backend (``mock_backend.py``) is intentionally standalone and
does not use this module, so it stays a minimal, copy-pasteable reference.
"""

import json
import urllib.request
import urllib.parse
import urllib.error
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


# --------------------------------------------------------------------------
# Inbound: request handler base
# --------------------------------------------------------------------------
class BaseAdapterHandler(BaseHTTPRequestHandler):
    """Common reply/parse helpers for a contract adapter.

    Subclasses implement ``do_GET`` / ``do_POST`` and route on ``self.path``.
    Set ``log_tag`` on the subclass to prefix the request log lines.
    """

    log_tag = "adapter"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # noqa: A003 — match stdlib signature
        print(f"[{self.log_tag}]", self.command, self.path)

    # -- replies --
    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_bytes(self, data, ctype="application/octet-stream", code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # -- request body --
    def read_body(self) -> bytes:
        n = int(self.headers.get("Content-Length", 0) or 0)
        return self.rfile.read(n) if n else b""

    def read_json(self) -> dict:
        body = self.read_body()
        return json.loads(body) if body else {}

    def read_multipart(self):
        """Return (fields, files) from a multipart/form-data POST body.

        ``fields`` is ``{name: str}`` and ``files`` is
        ``{name: (filename, content_type, bytes)}``. Used for the contract's
        ``POST /v1/files`` upload so the adapter can forward the real bytes.
        """
        ctype = self.headers.get("Content-Type", "")
        return parse_multipart(self.read_body(), ctype)


def serve(handler_cls, host="127.0.0.1", port=8000, banner=None):
    """Run ``handler_cls`` on a threaded server until Ctrl-C."""
    srv = ThreadingHTTPServer((host, port), handler_cls)
    if banner:
        print(banner)
    print(f"[{getattr(handler_cls, 'log_tag', 'adapter')}] listening on "
          f"http://{host}:{port}  (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


# --------------------------------------------------------------------------
# Multipart parsing (inbound /v1/files)
# --------------------------------------------------------------------------
def parse_multipart(body: bytes, content_type: str):
    """Parse a multipart/form-data body into (fields, files).

    Minimal RFC 7578 parser — enough for the contract's single-file uploads
    (the add-on's stdlib client posts one ``file`` part plus a ``purpose``
    field). Returns ``({name: str}, {name: (filename, ctype, bytes)})``.
    """
    fields: dict = {}
    files: dict = {}
    if "multipart/form-data" not in content_type:
        return fields, files

    boundary = ""
    for part in content_type.split(";"):
        part = part.strip()
        if part.startswith("boundary="):
            boundary = part[len("boundary="):].strip().strip('"')
            break
    if not boundary:
        return fields, files

    delim = ("--" + boundary).encode()
    for chunk in body.split(delim):
        if not chunk or chunk in (b"--\r\n", b"--", b"\r\n"):
            continue
        chunk = chunk.strip(b"\r\n")
        if b"\r\n\r\n" not in chunk:
            continue
        raw_headers, data = chunk.split(b"\r\n\r\n", 1)
        headers = raw_headers.decode("utf-8", "replace")
        name, filename, part_ctype = "", None, "application/octet-stream"
        for line in headers.split("\r\n"):
            low = line.lower()
            if low.startswith("content-disposition"):
                for token in line.split(";"):
                    token = token.strip()
                    if token.startswith("name="):
                        name = token[len("name="):].strip().strip('"')
                    elif token.startswith("filename="):
                        filename = token[len("filename="):].strip().strip('"')
            elif low.startswith("content-type"):
                part_ctype = line.split(":", 1)[1].strip()
        if not name:
            continue
        if filename is not None:
            files[name] = (filename or "upload.bin", part_ctype, data)
        else:
            fields[name] = data.decode("utf-8", "replace")
    return fields, files


# --------------------------------------------------------------------------
# Outbound: urllib helpers (replaces httpx)
# --------------------------------------------------------------------------
def get_json(url, headers=None, timeout=30):
    """GET ``url`` and parse the JSON reply (raises on transport error)."""
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def get_bytes(url, params=None, headers=None, timeout=300):
    """GET raw bytes; returns (data, content_type)."""
    if params:
        url = url + ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), r.headers.get("Content-Type", "application/octet-stream")


def post_json(url, obj, headers=None, timeout=60):
    """POST ``obj`` as JSON. Returns (status_code, parsed_body_or_text).

    Never raises on HTTP error status — the caller inspects the code so it can
    surface a backend's structured error body (e.g. ComfyUI /prompt rejects).
    """
    data = json.dumps(obj).encode("utf-8")
    h = {"Content-Type": "application/json", "Accept": "application/json"}
    h.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, _parse_body(r.read(), r.headers.get("Content-Type", ""))
    except urllib.error.HTTPError as e:
        return e.code, _parse_body(e.read(), e.headers.get("Content-Type", ""))


def post_multipart(url, files=None, data=None, headers=None, timeout=120):
    """POST a multipart/form-data request. Returns parsed JSON reply.

    ``files`` = ``{field: (filename, bytes, content_type)}``;
    ``data``  = ``{field: str}``.
    """
    body, ctype = encode_multipart(data or {}, files or {})
    h = {"Content-Type": ctype}
    h.update(headers or {})
    req = urllib.request.Request(url, data=body, headers=h, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        return _parse_body(raw, r.headers.get("Content-Type", "application/json"))


def encode_multipart(fields: dict, files: dict):
    """Build a multipart/form-data body. Returns (bytes, content_type)."""
    boundary = "----pallaidium" + uuid.uuid4().hex
    out = bytearray()
    for name, value in fields.items():
        out += f"--{boundary}\r\n".encode()
        out += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        out += f"{value}\r\n".encode()
    for name, (filename, content, content_type) in files.items():
        out += f"--{boundary}\r\n".encode()
        out += (f'Content-Disposition: form-data; name="{name}"; '
                f'filename="{filename}"\r\n').encode()
        out += f"Content-Type: {content_type}\r\n\r\n".encode()
        out += content
        out += b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"


def _parse_body(raw: bytes, content_type: str):
    if "application/json" in (content_type or "") or not content_type:
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:  # noqa: BLE001
            pass
    return raw.decode("utf-8", "replace")
