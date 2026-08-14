# SiftQ MiniMax-H3 V2 provider

Pallaidium includes a direct cloud video plugin named **SiftQ MiniMax H3
(cloud)**. It uses the MiniMax-H3 V2 request/response contract through SiftQ,
with an independent provider identity and runtime configuration.

## Configuration

In Blender's **Video Editing → Generative AI** panel, select **Video** and
**SiftQ MiniMax H3 (cloud)**, then paste the key into **Session API Key**. The
field is password-masked and marked `SKIP_SAVE`: it applies only to the current
Blender process and is never stored in add-on preferences, render queue,
metadata, `.blend` files, the repository, command line or adapter logs. Clear
the field to remove the session override; if Blender was launched with
`SIFTQ_API_KEY`, that original value is restored. Disabling the add-on also
removes the session override, and closing Blender ends the session.

For scripted launches, setting the environment before starting Blender remains
supported:

Blender's **Online Access** setting must also be enabled. The extension declares
its network permission in `blender_manifest.toml` and refuses to submit while
Blender reports online access as disabled.

PowerShell:

```powershell
$env:SIFTQ_API_KEY="your-runtime-key"
```

Bash:

```bash
export SIFTQ_API_KEY="your-runtime-key"
```

The default API base URL is:

```text
https://siftq.com/api/minimax/
```

For a compatible test gateway or deployment override, set `SIFTQ_BASE_URL`
before starting Blender. The value must be an absolute HTTP(S) URL without
embedded credentials, query parameters or a fragment.

Optional controls are `SIFTQ_REQUEST_TIMEOUT_SECONDS` (per-request timeout,
default `60`), `SIFTQ_POLL_TIMEOUT_SECONDS` (whole polling deadline, default
`3600`) and `SIFTQ_POLL_INTERVAL_SECONDS` (default `2`).

## Usage

1. In the VSE panel, choose **Output → Video**.
2. Select **SiftQ MiniMax H3 (cloud)**.
3. Choose a SiftQ mode:
   - **Text to Video** — prompt only.
   - **First Frame** — choose an IMAGE, MOVIE, SCENE or META strip in the
     visible **First Frame** picker. For IMAGE strips, Pallaidium uploads the
     original source file so a landscape VSE canvas cannot add black borders or
     override the source aspect ratio; timeline transform/crop settings are not
     baked into that upload. Other supported strip types are snapshotted to an
     image at queue-add time without requiring Pillow/OpenCV. Selecting a
     timeline input strip remains a compatible fallback.
   - **First + Last Frame** — choose image strips in the visible **First Frame**
     and **Last Frame** pickers. A selected META strip containing two ordered
     image children remains a compatible fallback.
   - **Reference to Video** — the main IMAGE or MOVIE strip becomes one
     reference image or video; use the dedicated reference-image pickers for
     additional images and **Ref. Audio** for an optional WAV/MP3 reference.
     A main IMAGE counts toward the 9-image total, and duplicate paths are
     removed before submission.
     Dedicated IMAGE-strip references likewise use their original source files,
     without baking VSE canvas padding, transforms or crop settings.
4. Set `768P` or `2K`, a whole duration from 4 through 15 seconds, and an aspect
   ratio. Text mode exposes only concrete ratios; frame modes always send
   `adaptive`; reference mode can use `adaptive` or a concrete ratio.
5. Add the job to the queue and run it. Pallaidium submits once, polls with a
   deadline, downloads the time-limited result URL immediately, validates the
   content type and MP4 signature, and adds the video to the timeline.

## Advertised limits

- Model protocol ID: `MiniMax-H3`.
- Prompt: required, up to 7000 characters per text item.
- Reference images: up to 9.
- Reference video: one through the current Pallaidium main-input surface.
- Reference audio: one through the current Pallaidium audio-reference surface.
- First/last-frame roles cannot be mixed with reference roles.
- The upstream contract can accept up to three reference videos and three
  reference audio files, but this plugin does **not** advertise those counts
  because Pallaidium currently exposes one of each.
- H3 Context-IR, task-list UI and callback handling are contract-tested client
  primitives but are not advertised as Pallaidium model features.

## Cancellation and errors

Pallaidium cancellation is cooperative. If SiftQ still reports `queued`, the
plugin sends the V2 DELETE request. The upstream contract rejects cancellation
once a task is `running`; in that case Pallaidium stops waiting locally, but the
upstream task may continue. The UI must not be interpreted as a guarantee that
running cloud work was cancelled or refunded.

Structured HTTP errors are normalized without logging bearer values, raw error
bodies or signed output URLs. A live end-to-end check requires a runtime key;
the standalone `test_siftq_provider.py` suite uses only a local deterministic
mock.

Pallaidium waits up to 180 seconds for each SiftQ HTTP response by default
(`SIFTQ_REQUEST_TIMEOUT_SECONDS`). Before a billable create request it snapshots
recent task IDs. If the create response is lost, it does not resubmit: for up to
120 seconds (`SIFTQ_SUBMIT_RECOVERY_SECONDS`) it searches for one uniquely new,
time-matched generation task and continues polling only when that match is
unambiguous. Otherwise it stops and tells the user to inspect the task list
before retrying.

The documented active-task status is `running`. The live SiftQ task endpoints
have also returned `processing`; Pallaidium treats that observed response alias
as `running` while continuing to send only documented status values in request
filters.

If a time-limited output URL has expired with HTTP 401, 403 or 404, Pallaidium
queries the succeeded task once for a refreshed URL and retries the validated
download.
