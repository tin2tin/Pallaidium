"""Minimal client for an OpenAI-/v1-dialect generation backend.

Implements the "Backend Contract v0.1" surface: a remote service exposing
``/v1/health``, ``/v1/models``, ``/v1/{videos,images/generations,audio/...}``,
``/v1/files`` and ``/v1/jobs/{id}``.  Any service that satisfies that contract
(self-hosted or online) can drive Pallaidium through RemoteModelPlugin.

Deliberately stdlib-only (urllib) so it adds no new dependency.  It is not a
high-performance client — generations are slow and network-bound anyway, so
clarity wins over speed here.
"""

import json
import os
import time
import uuid
import mimetypes
import urllib.request
import urllib.error
import urllib.parse


# Contract version this client speaks. Surfaced to the backend so it can
# refuse / adapt if it only supports something else.
CONTRACT_VERSION = "v0.1"

# Terminal job states per the contract.
_TERMINAL_OK = "succeeded"
_TERMINAL_FAIL = "failed"


class RemoteBackendError(RuntimeError):
    """Raised for any backend / transport / contract failure."""


class RemoteBackendClient:
    """Thin HTTP client for a single backend base URL.

    Auth is optional; when a key is given it is sent three ways at once
    (Bearer header, X-API-Key header, and ?api_key query) because the contract
    says a backend may accept any one of them.
    """

    def __init__(self, base_url: str, api_key: str = "", timeout: float = 60.0,
                 submit_timeout: float = 300.0):
        if not base_url:
            raise RemoteBackendError(
                "No backend URL configured. Set the add-on preference "
                "'Remote Backend URL' or the PALLAIDIUM_BACKEND_URL env var."
            )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or ""
        # Short timeout for health/poll/download; longer one for the submit POST
        # because some backends generate synchronously and answer slowly.
        self.timeout = timeout
        self.submit_timeout = submit_timeout

    # -- low-level ----------------------------------------------------------

    def _url(self, path: str) -> str:
        url = self.base_url + "/" + path.lstrip("/")
        if self.api_key:
            sep = "&" if "?" in url else "?"
            url += sep + urllib.parse.urlencode({"api_key": self.api_key})
        return url

    def _headers(self, extra: dict | None = None) -> dict:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["X-API-Key"] = self.api_key
        if extra:
            headers.update(extra)
        return headers

    def _request(self, method: str, path: str, *, data: bytes | None = None,
                 headers: dict | None = None, expect_json: bool = True,
                 timeout: float | None = None):
        req = urllib.request.Request(
            self._url(path), data=data, method=method,
            headers=self._headers(headers),
        )
        try:
            with urllib.request.urlopen(
                req, timeout=self.timeout if timeout is None else timeout
            ) as resp:
                body = resp.read()
                if expect_json:
                    return json.loads(body.decode("utf-8")) if body else {}
                return body, resp.headers
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:500]
            raise RemoteBackendError(
                f"{method} {path} -> HTTP {e.code}: {detail}"
            ) from e
        except urllib.error.URLError as e:
            raise RemoteBackendError(
                f"{method} {path} -> connection failed: {e.reason}"
            ) from e

    def _get(self, path: str):
        return self._request("GET", path)

    def _post_json(self, path: str, payload: dict, *, timeout: float | None = None):
        data = json.dumps(payload).encode("utf-8")
        return self._request(
            "POST", path, data=data,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )

    # -- contract endpoints -------------------------------------------------

    def health(self) -> dict:
        """GET /v1/health -> {status, contract_versions}."""
        return self._get("/v1/health")

    def check_compatible(self) -> dict:
        """Verify the backend advertises our contract version. Returns health."""
        info = self.health()
        versions = info.get("contract_versions") or []
        if versions and CONTRACT_VERSION not in versions:
            raise RemoteBackendError(
                f"Backend speaks {versions}, add-on needs {CONTRACT_VERSION!r}."
            )
        return info

    def models(self) -> list:
        """GET /v1/models -> list of {id, type, modes}."""
        return self._get("/v1/models").get("data", [])

    def upload_bytes(self, filename: str, data: bytes,
                     purpose: str = "reference") -> str:
        """POST /v1/files (multipart) from in-memory bytes -> file_id.

        Used for ModelInputs that hold PIL images / data rather than paths, so
        no temp file is written.
        """
        mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        body, content_type = _encode_multipart(
            fields={"purpose": purpose},
            files={"file": (filename, data, mime)},
        )
        resp = self._request(
            "POST", "/v1/files", data=body,
            headers={"Content-Type": content_type},
            timeout=self.submit_timeout,
        )
        file_id = resp.get("file_id")
        if not file_id:
            raise RemoteBackendError(f"/v1/files returned no file_id: {resp}")
        return file_id

    def upload_file(self, file_path: str, purpose: str = "reference") -> str:
        """POST /v1/files (multipart) from a path -> file_id."""
        with open(file_path, "rb") as fh:
            content = fh.read()
        return self.upload_bytes(os.path.basename(file_path), content, purpose)

    def transcribe(self, file_path: str, model: str) -> dict:
        """POST /v1/audio/transcriptions (multipart) -> {"text": ...}."""
        with open(file_path, "rb") as fh:
            content = fh.read()
        filename = os.path.basename(file_path)
        mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        body, content_type = _encode_multipart(
            fields={"model": model},
            files={"file": (filename, content, mime)},
        )
        return self._request(
            "POST", "/v1/audio/transcriptions", data=body,
            headers={"Content-Type": content_type},
            timeout=self.submit_timeout,
        )

    def download_file(self, file_id: str, dst_path: str) -> str:
        """GET /v1/files/{id} -> save binary to dst_path."""
        body, _ = self._request(
            "GET", f"/v1/files/{file_id}", expect_json=False,
        )
        os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
        with open(dst_path, "wb") as fh:
            fh.write(body)
        return dst_path

    def download_url(self, url: str, dst_path: str) -> str:
        """Download an OpenAI-style direct result URL to dst_path."""
        if url.startswith(self.base_url):
            path = url[len(self.base_url):] or "/"
            body, _ = self._request("GET", path, expect_json=False)
        else:
            body, _ = _download_absolute(url, self.timeout)
        os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
        with open(dst_path, "wb") as fh:
            fh.write(body)
        return dst_path

    def poll_job(self, job_id: str, dst_path: str, *,
                 phase_fn=None, progress_fn=None, should_cancel=None,
                 interval: float = 2.0, max_wait: float = 3600.0) -> str:
        """Poll GET /v1/jobs/{id} to a terminal state, then download the result.

        ``phase_fn(label)`` and ``progress_fn(step, total)`` mirror the queue
        callbacks on ModelInputs, so the Pallaidium queue panel shows live
        progress for remote jobs exactly like local ones.

        ``should_cancel()`` (optional) is checked each iteration; when it returns
        True we raise ``KeyboardInterrupt`` — the queue worker maps that to a
        CANCELLED job (not FAILED).
        """
        deadline = time.time() + max_wait
        last_phase = None
        while True:
            if should_cancel is not None and should_cancel():
                raise KeyboardInterrupt(f"Remote job {job_id} cancelled by user.")
            job = self._get(f"/v1/jobs/{job_id}")
            status = job.get("status")

            phase = job.get("phase")
            if phase and phase != last_phase and phase_fn:
                phase_fn(phase)
                last_phase = phase
            prog = job.get("progress")
            if isinstance(prog, (int, float)) and progress_fn:
                # contract progress is 0..1; report as (current, 100).
                progress_fn(int(round(prog * 100)), 100)

            if status == _TERMINAL_OK:
                file_id = job.get("file_id")
                if not file_id:
                    raise RemoteBackendError(
                        f"Job {job_id} succeeded but returned no file_id."
                    )
                return self.download_file(file_id, dst_path)
            if status == _TERMINAL_FAIL:
                raise RemoteBackendError(
                    f"Job {job_id} failed: {job.get('error') or 'unknown error'}"
                )
            if time.time() > deadline:
                raise RemoteBackendError(
                    f"Job {job_id} timed out after {max_wait:.0f}s (status={status})."
                )
            time.sleep(interval)

    def run(self, path: str, payload: dict, dst_path: str, *,
            phase_fn=None, progress_fn=None, should_cancel=None) -> str:
        """Submit a generation and resolve it to a local file path.

        Handles both contract response shapes:
          * async  -> {"id", "status": "queued"}  (poll jobs, download file)
          * direct -> {"file_id"} | {"url"} | {"data": [{"url"|"b64_json"}]}
        """
        resp = self._post_json(path, payload, timeout=self.submit_timeout)

        # Async job.
        if resp.get("id") and resp.get("status") in ("queued", "running"):
            return self.poll_job(
                resp["id"], dst_path,
                phase_fn=phase_fn, progress_fn=progress_fn,
                should_cancel=should_cancel,
            )

        # Direct file_id.
        if resp.get("file_id"):
            return self.download_file(resp["file_id"], dst_path)

        # OpenAI-style direct result: url or base64.
        data = resp.get("data")
        if isinstance(data, list) and data:
            first = data[0]
            if first.get("url"):
                return self.download_url(first["url"], dst_path)
            if first.get("b64_json"):
                return self._write_b64(first["b64_json"], dst_path)
        if resp.get("url"):
            return self.download_url(resp["url"], dst_path)
        if resp.get("b64_json"):
            return self._write_b64(resp["b64_json"], dst_path)

        raise RemoteBackendError(
            f"Unrecognised response from {path}: {json.dumps(resp)[:300]}"
        )

    @staticmethod
    def _write_b64(b64: str, dst_path: str) -> str:
        """Decode a base64 payload and write it to dst_path."""
        import base64
        os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
        with open(dst_path, "wb") as fh:
            fh.write(base64.b64decode(b64))
        return dst_path


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def client_from_prefs(prefs, timeout: float = 60.0) -> RemoteBackendClient:
    """Build a client from add-on prefs, falling back to env vars.

    Reads ``prefs.remote_backend_url`` / ``prefs.remote_backend_key`` if those
    properties exist, otherwise PALLAIDIUM_BACKEND_URL / PALLAIDIUM_BACKEND_KEY.
    This keeps the PoC working without editing preferences.py.
    """
    url = getattr(prefs, "remote_backend_url", "") or os.environ.get(
        "PALLAIDIUM_BACKEND_URL", ""
    )
    key = getattr(prefs, "remote_backend_key", "") or os.environ.get(
        "PALLAIDIUM_BACKEND_KEY", ""
    )
    return RemoteBackendClient(url, key, timeout=timeout)


# ---------------------------------------------------------------------------
# Discovery cache — persist the last /v1/models list so remote models survive a
# Blender restart without re-querying the backend (cineloom-style).
# ---------------------------------------------------------------------------

def _datafiles_dir() -> str:
    try:
        import bpy
        d = os.path.join(bpy.utils.user_resource("DATAFILES", create=True), "Pallaidium")
    except Exception:  # noqa: BLE001 — outside Blender (tests)
        d = os.path.join(os.path.expanduser("~"), ".pallaidium")
    os.makedirs(d, exist_ok=True)
    return d


def _discovery_cache_path() -> str:
    return os.path.join(_datafiles_dir(), "discovery.json")


def save_discovery_cache(url: str, entries: list) -> None:
    """Persist the discovered model list for ``url`` to discovery.json."""
    try:
        with open(_discovery_cache_path(), "w", encoding="utf-8") as f:
            json.dump({"url": url, "at": time.time(), "entries": entries}, f)
    except Exception as e:  # noqa: BLE001 — cache is best-effort
        print(f"[pallaidium] could not write discovery cache: {e}")


def load_discovery_cache(url: str = "") -> list:
    """Return cached model entries, or [] if absent.

    If ``url`` is given, only return the cache when it matches the cached URL
    (so stale models from a different backend aren't shown).
    """
    try:
        with open(_discovery_cache_path(), encoding="utf-8") as f:
            data = json.load(f)
    except Exception:  # noqa: BLE001 — no/with bad cache -> nothing to load
        return []
    if url and data.get("url") and data["url"].rstrip("/") != url.rstrip("/"):
        return []
    entries = data.get("entries")
    return entries if isinstance(entries, list) else []


def _download_absolute(url: str, timeout: float):
    """GET an absolute URL that is not under our base_url (e.g. a CDN link)."""
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read(), resp.headers
    except urllib.error.URLError as e:
        raise RemoteBackendError(f"download {url} failed: {e}") from e


def _encode_multipart(fields: dict, files: dict):
    """Encode multipart/form-data. files: {name: (filename, bytes, mime)}."""
    boundary = "----PallaidiumBoundary" + uuid.uuid4().hex
    crlf = b"\r\n"
    buf = []
    for name, value in fields.items():
        buf.append(b"--" + boundary.encode())
        buf.append(
            f'Content-Disposition: form-data; name="{name}"'.encode()
        )
        buf.append(b"")
        buf.append(str(value).encode())
    for name, (filename, content, mime) in files.items():
        buf.append(b"--" + boundary.encode())
        buf.append(
            f'Content-Disposition: form-data; name="{name}"; '
            f'filename="{filename}"'.encode()
        )
        buf.append(f"Content-Type: {mime}".encode())
        buf.append(b"")
        buf.append(content)
    buf.append(b"--" + boundary.encode() + b"--")
    buf.append(b"")
    body = crlf.join(buf)
    return body, f"multipart/form-data; boundary={boundary}"
