"""
System operators: install/uninstall dependencies, sound notification,
LoRA file refresh, IP Adapter file browsers.
"""

import bpy
import os
import glob
import subprocess
import importlib
import importlib.metadata
import importlib.util
import aud

from bpy_extras.io_utils import ExportHelper
from bpy.types import Operator
from bpy.props import StringProperty, BoolProperty

import threading

from ..utils.helpers import (
    ADDON_ID,
    python_exec,
    site_packages_dir,
    DependencyManager,
    BlenderInternalManager,
    SmartSkipManager,
    run_pip_streaming,
    install_requirements_binary_only,
    install_requirements_allow_source,
    write_requirements_file,
)

# ---------------------------------------------------------------------------
# Async dependency operation state (written by worker thread, read by timer)
# ---------------------------------------------------------------------------

_dep_state: dict = {
    "running": False,
    "op_type": "",          # "install" | "uninstall"
    "progress": 0.0,        # 0.0–1.0
    "phase": "",            # current batch description
    "status_line": "",      # latest stripped pip stdout line
    "failure_report": "",   # formatted Markdown bug report
    "needs_restart": False, # True after a successful install; cleared on Blender restart via SKIP_SAVE
}
_dep_cancel_event = threading.Event()
_dep_worker_thread: threading.Thread | None = None
_dep_failed_batches: list = []   # {"phase", "packages", "output"}
_dep_was_running = False         # sentinel for running→done transition

_DEP_TIMER_INTERVAL = 0.2

# File written on disk so errors survive a Blender restart.
_ADDON_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
_INSTALL_ERRORS_FILE = os.path.join(_ADDON_DIR, "_install_failures.json")


def load_install_errors_from_disk() -> dict | None:
    """Return the saved failure dict, or None if there is no failure file."""
    import json
    try:
        with open(_INSTALL_ERRORS_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def clear_install_errors_file():
    """Delete the on-disk failure file (called after a clean install or user dismiss)."""
    try:
        if os.path.exists(_INSTALL_ERRORS_FILE):
            os.remove(_INSTALL_ERRORS_FILE)
    except Exception:
        pass


def _fix_site_packages_permissions(target_dir: str):
    """Recursively ensure all files in target_dir are readable/writable.

    pip on Windows sometimes installs files with read-only attributes inherited
    from zip metadata, causing PermissionError when importlib tries to read them.
    """
    import stat
    try:
        for root, _dirs, files in os.walk(target_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                try:
                    current = os.stat(fpath).st_mode
                    os.chmod(fpath, current | stat.S_IREAD | stat.S_IWRITE)
                except Exception:
                    pass
    except Exception:
        pass


def _get_dep_prefs():
    try:
        return bpy.context.preferences.addons[ADDON_ID].preferences
    except Exception:
        return None


def _build_failure_report() -> str:
    import platform, sys as _sys
    lines = [
        "## Pallaidium Dependency Install Failures",
        f"- Platform: {platform.system()} {platform.release()}",
        f"- Python: {_sys.version.split()[0]}",
        f"- Blender: {bpy.app.version_string}",
        "",
        "### Failed batches",
    ]
    for f in _dep_failed_batches:
        lines += [
            f"**Phase**: {f['phase']}",
            f"**Packages**: {f['packages']}",
            "```",
            f['output'][-3000:],
            "```",
            "",
        ]
    lines += [
        "---",
        "_Please open a GitHub issue at https://github.com/tin2tin/Pallaidium and paste this report._",
    ]
    return "\n".join(lines)


def _dep_tick() -> float | None:
    global _dep_was_running
    state = _dep_state
    currently_running = state["running"]

    # Detect finish transition — build failure report on the main thread
    if _dep_was_running and not currently_running and _dep_failed_batches:
        report = _build_failure_report()
        state["failure_report"] = report
        try:
            txt = bpy.data.texts.get("Pallaidium Install Errors") or bpy.data.texts.new("Pallaidium Install Errors")
            txt.clear()
            txt.write(report)
        except Exception:
            pass
        # Persist errors to disk so they survive a Blender restart
        import json
        try:
            payload = {
                "report": report,
                "summary": [f['packages'] for f in _dep_failed_batches],
                "batches": [
                    {"phase": f["phase"], "packages": f["packages"],
                     "output_tail": "\n".join(f["output"].splitlines()[-30:])}
                    for f in _dep_failed_batches
                ],
            }
            with open(_INSTALL_ERRORS_FILE, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        except Exception:
            pass
        # Print summary to the Blender system console so it's always visible
        import sys
        print("\n" + "=" * 60, file=sys.stderr, flush=True)
        print("PALLAIDIUM: Dependency installation errors:", file=sys.stderr, flush=True)
        for f in _dep_failed_batches:
            print(f"  Phase   : {f['phase']}", file=sys.stderr, flush=True)
            print(f"  Packages: {f['packages']}", file=sys.stderr, flush=True)
            tail = "\n".join(f['output'].splitlines()[-8:])
            for ln in tail.splitlines():
                print(f"    {ln}", file=sys.stderr, flush=True)
        print("  Full report: open the Text Editor and select 'Pallaidium Install Errors'", file=sys.stderr, flush=True)
        print("=" * 60 + "\n", file=sys.stderr, flush=True)
        # Show a popup so the user is notified even if Preferences is not open
        failed_names = [f['packages'] for f in _dep_failed_batches]
        def _draw_error_popup(self, context):
            layout = self.layout
            layout.label(text=f"{len(_dep_failed_batches)} package batch(es) failed to install.", icon='ERROR')
            for name in failed_names[:4]:
                layout.label(text=f"  • {name[:60]}")
            layout.separator()
            layout.label(text="Open Add-on Preferences → copy the error report,")
            layout.label(text="or check the Text Editor block 'Pallaidium Install Errors'.")
        try:
            bpy.context.window_manager.popup_menu(
                _draw_error_popup, title="Pallaidium: Install Failed", icon='ERROR'
            )
        except Exception:
            pass
    _dep_was_running = currently_running

    try:
        prefs = _get_dep_prefs()
        if prefs:
            prefs.dep_is_running     = currently_running
            prefs.dep_op_type        = state["op_type"]
            prefs.dep_progress       = float(state["progress"])
            prefs.dep_phase          = state["phase"]
            prefs.dep_status_line    = state["status_line"][:120]
            prefs.dep_has_errors     = bool(_dep_failed_batches) and not currently_running
            prefs.dep_failure_report = state.get("failure_report", "")
            prefs.dep_needs_restart  = state.get("needs_restart", False)
    except Exception:
        pass
    try:
        for win in bpy.context.window_manager.windows:
            for area in win.screen.areas:
                if area.type in ("PREFERENCES", "SEQUENCE_EDITOR"):
                    area.tag_redraw()
    except Exception:
        pass
    return _DEP_TIMER_INTERVAL if currently_running else None


def _ensure_python_headers_linux(on_line):
    """Copy Python C headers into Blender's embedded Python include dir on Linux.

    Portable Blender builds ship without Python.h, which causes any package that
    compiles a C/Cython/CUDA extension (sageattention, thinc from source, etc.) to
    fail immediately.  We fix this automatically:

      1. If Python.h already exists — nothing to do.
      2. Static headers (Include/*.h) — extracted directly from the python.org source
         tarball using only urllib + tarfile; no compiler needed.
      3. pyconfig.h — try the system Python include dir first (same version = same
         ABI); fall back to running ./configure in the extracted source tree if
         autotools are present.

    Prints progress through on_line().  Never raises — logs a warning and returns
    False if headers could not be installed so the caller can print manual steps.
    """
    import sys, sysconfig, tarfile, shutil, urllib.request, subprocess, re, tempfile

    include_dir = sysconfig.get_path("include")   # e.g. .../5.2/python/include/python3.13
    if not include_dir:
        on_line("PALLAIDIUM headers: could not determine Python include path — skipping")
        return False

    python_h = os.path.join(include_dir, "Python.h")
    if os.path.exists(python_h):
        return True  # already present, nothing to do

    ver = sys.version_info
    ver_str   = f"{ver.major}.{ver.minor}.{ver.micro}"
    ver_tag   = f"python{ver.major}.{ver.minor}"
    tarball   = f"Python-{ver_str}.tar.xz"
    url       = f"https://www.python.org/ftp/python/{ver_str}/{tarball}"

    on_line(f"PALLAIDIUM headers: Python.h missing in {include_dir}")
    on_line(f"PALLAIDIUM headers: downloading {url} ...")

    try:
        os.makedirs(include_dir, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="pallaidium_pyhdr_") as tmp:
            tarball_path = os.path.join(tmp, tarball)

            # --- download ---
            def _reporthook(count, block, total):
                if total > 0 and count % 200 == 0:
                    mb = count * block / 1_048_576
                    on_line(f"PALLAIDIUM headers: downloaded {mb:.1f} MB / {total/1_048_576:.1f} MB")
            urllib.request.urlretrieve(url, tarball_path, reporthook=_reporthook)
            on_line("PALLAIDIUM headers: download complete, extracting headers ...")

            # --- extract Include/ and the root-level cpython/ sub-dir ---
            src_root = os.path.join(tmp, f"Python-{ver_str}")
            with tarfile.open(tarball_path, "r:xz") as tf:
                members = [
                    m for m in tf.getmembers()
                    if re.match(
                        rf"Python-{re.escape(ver_str)}/(Include/|Modules/_io/|PC/pyconfig\.h)",
                        m.name,
                    )
                ]
                tf.extractall(tmp, members=members)

            src_include = os.path.join(src_root, "Include")
            if not os.path.isdir(src_include):
                on_line("PALLAIDIUM headers: Include/ not found in tarball — aborting")
                return False

            # copy all .h files from Include/ (flat, ignore cpython/ sub-dir for now)
            copied = 0
            for fname in os.listdir(src_include):
                if fname.endswith(".h"):
                    shutil.copy2(os.path.join(src_include, fname), include_dir)
                    copied += 1
            # copy cpython/ sub-directory
            cpython_src = os.path.join(src_include, "cpython")
            if os.path.isdir(cpython_src):
                cpython_dst = os.path.join(include_dir, "cpython")
                if os.path.isdir(cpython_dst):
                    shutil.rmtree(cpython_dst)
                shutil.copytree(cpython_src, cpython_dst)
            on_line(f"PALLAIDIUM headers: copied {copied} header files")

            # --- pyconfig.h ---
            # Prefer the system Python's pyconfig.h if the version matches exactly.
            pyconfig_dst = os.path.join(include_dir, "pyconfig.h")
            system_pyconfig = f"/usr/include/{ver_tag}/pyconfig.h"
            if os.path.exists(system_pyconfig):
                shutil.copy2(system_pyconfig, pyconfig_dst)
                on_line(f"PALLAIDIUM headers: copied pyconfig.h from {system_pyconfig}")
            else:
                # Fall back: run ./configure in the extracted source tree
                on_line("PALLAIDIUM headers: system pyconfig.h not found, running ./configure ...")
                configure_script = os.path.join(src_root, "configure")
                if os.path.exists(configure_script):
                    r = subprocess.run(
                        ["./configure", f"--prefix={os.path.join(tmp, 'build')}"],
                        cwd=src_root, capture_output=True, text=True, timeout=120,
                    )
                    generated = os.path.join(src_root, "pyconfig.h")
                    if r.returncode == 0 and os.path.exists(generated):
                        shutil.copy2(generated, pyconfig_dst)
                        on_line("PALLAIDIUM headers: pyconfig.h generated via ./configure")
                    else:
                        on_line("PALLAIDIUM headers: ./configure failed — pyconfig.h not installed")
                        on_line("  Manual fix: https://github.com/tin2tin/Pallaidium — see Linux guide step 7")
                        return False
                else:
                    on_line("PALLAIDIUM headers: configure script not found — pyconfig.h not installed")
                    return False

        on_line(f"PALLAIDIUM headers: Python.h successfully installed in {include_dir}")
        return True

    except Exception as exc:
        on_line(f"PALLAIDIUM headers: failed — {exc}")
        on_line("  Manual fix: see Linux installation guide step 7")
        return False


def _run_install(snapshot: dict, cancel_event: threading.Event):
    import sys, traceback, platform as _plat
    state     = _dep_state
    pybin     = snapshot["pybin"]
    addon_dir = snapshot["addon_dir"]
    batches   = snapshot["batches"]
    total_pkgs = snapshot["total_pkgs"]
    done_pkgs = 0
    batch_output_lines: list = []

    def on_line(line):
        print(line, flush=True)
        state["status_line"] = line
        batch_output_lines.append(line)

    try:
        # Linux pre-flight: ensure Python.h is present so compiled extensions work
        if _plat.system() == "Linux":
            state["phase"] = "Checking Python headers..."
            _ensure_python_headers_linux(on_line)
            if cancel_event.is_set():
                return

        # pip self-upgrade
        run_pip_streaming(
            [pybin, "-m", "pip", "install", "--upgrade", "pip", "--disable-pip-version-check"],
            on_line=on_line, cancel_event=cancel_event,
        )
        if cancel_event.is_set():
            return

        for phase_name, only_binary, lines in batches:
            BATCH_SIZE = 5
            for i in range(0, len(lines), BATCH_SIZE):
                if cancel_event.is_set():
                    return
                batch = lines[i: i + BATCH_SIZE]
                batch_num   = (i // BATCH_SIZE) + 1
                total_batch = (len(lines) + BATCH_SIZE - 1) // BATCH_SIZE
                names = [SmartSkipManager.extract_package_name(x) or x for x in batch]
                state["phase"] = f"[{phase_name}] Batch {batch_num}/{total_batch}: {', '.join(names)}"

                batch_output_lines.clear()
                temp_req = os.path.join(addon_dir, f"_temp_dep_{phase_name}_{batch_num}.txt")
                write_requirements_file(temp_req, batch)

                cmd = [
                    pybin, "-m", "pip", "install",
                    "--upgrade",
                    "--disable-pip-version-check",
                    "--no-warn-script-location",
                    "--no-deps",
                    "--target", site_packages_dir,
                    "-r", temp_req,
                ]
                if only_binary:
                    cmd.insert(-2, "--only-binary=:all:")

                success = run_pip_streaming(cmd, on_line=on_line, cancel_event=cancel_event)
                if os.path.exists(temp_req):
                    os.remove(temp_req)

                # Fix read-only attributes pip sometimes sets on Windows
                if success and site_packages_dir:
                    _fix_site_packages_permissions(site_packages_dir)

                if not success and not cancel_event.is_set():
                    print(f"\nPALLAIDIUM pip FAILED — {phase_name} batch {batch_num}/{total_batch}: {', '.join(names)}", file=sys.stderr, flush=True)
                    for ln in batch_output_lines[-15:]:
                        print(f"  {ln}", file=sys.stderr, flush=True)
                    _req_filename = "requirements_linux.txt" if _plat.system() == "Linux" else "requirements.txt"
                    print(f"  → To skip, add '#' before each name in {_req_filename} and click Install Dependencies again.", file=sys.stderr, flush=True)
                    _dep_failed_batches.append({
                        "phase":    f"{phase_name} batch {batch_num}/{total_batch}",
                        "packages": ", ".join(names),
                        "output":   "\n".join(batch_output_lines),
                    })

                done_pkgs += len(batch)
                state["progress"] = min(done_pkgs / max(1, total_pkgs), 1.0)

        print("\n" + "=" * 60, flush=True)
        if _dep_failed_batches:
            state["phase"] = f"Done with {len(_dep_failed_batches)} error(s) — see report"
            print(f"PALLAIDIUM: Installation finished with {len(_dep_failed_batches)} failed batch(es):", flush=True)
            for f in _dep_failed_batches:
                print(f"  FAILED  [{f['phase']}]  {f['packages']}", flush=True)
            print("  Open Add-on Preferences and click 'Copy Error Report'.", flush=True)
        else:
            state["phase"] = "Installation complete — please restart Blender"
            clear_install_errors_file()
            print(f"PALLAIDIUM: Installation complete ({total_pkgs} package(s)). Please restart Blender.", flush=True)
        print("=" * 60 + "\n", flush=True)
        if total_pkgs > 0:
            state["needs_restart"] = True
        state["progress"] = 1.0

    except Exception:
        tb = traceback.format_exc()
        print(f"\nPALLAIDIUM install thread crashed:\n{tb}", file=sys.stderr, flush=True)
        _dep_failed_batches.append({
            "phase":    state.get("phase", "unknown"),
            "packages": "(internal error — see traceback)",
            "output":   tb,
        })
        state["phase"]    = "Install crashed — see error report"
        state["progress"] = 1.0

    finally:
        state["running"] = False


def _run_uninstall(snapshot: dict, cancel_event: threading.Event):
    state = _dep_state
    state["phase"]    = "Uninstalling dependencies..."
    state["progress"] = 0.0

    try:
        if not os.path.isdir(site_packages_dir):
            state["status_line"] = "Nothing to remove."
        else:
            # Count total files first so we can show real progress.
            all_files = []
            for root, _dirs, files in os.walk(site_packages_dir):
                for f in files:
                    all_files.append(os.path.join(root, f))
            total = max(len(all_files), 1)

            state["status_line"] = f"Removing {total} files..."

            for i, fpath in enumerate(all_files):
                if cancel_event.is_set():
                    state["phase"]   = "Uninstall cancelled"
                    state["running"] = False
                    return
                try:
                    os.remove(fpath)
                except Exception:
                    pass
                state["progress"]    = (i + 1) / total
                state["status_line"] = os.path.basename(fpath)

            # Remove leftover empty directories.
            for root, dirs, _files in os.walk(site_packages_dir, topdown=False):
                for d in dirs:
                    try:
                        os.rmdir(os.path.join(root, d))
                    except Exception:
                        pass

            os.makedirs(site_packages_dir, exist_ok=True)
            state["status_line"] = f"Removed {total} files."

    except Exception as e:
        state["status_line"] = f"Error during uninstall: {e}"

    state["phase"]    = "Uninstall complete — please restart Blender"
    state["progress"] = 1.0
    state["running"]  = False


class GENERATOR_OT_export_requirements(Operator, ExportHelper):
    bl_idname = "sequencer.export_requirements"
    bl_label = "Export requirements.txt"
    bl_options = {'REGISTER'}
    filename_ext = ".txt"
    filter_glob: bpy.props.StringProperty(default="requirements.txt", options={'HIDDEN'}, maxlen=255)

    def execute(self, context):
        dists = importlib.metadata.distributions()
        lines = []
        for dist in dists:
            try: lines.append(f"{dist.metadata['Name']}=={dist.version}")
            except: pass
        lines.sort()
        try:
            with open(self.filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
        except Exception: return {'CANCELLED'}
        return {'FINISHED'}


class GENERATOR_OT_install(Operator):
    bl_idname = "sequencer.install_generator"
    bl_label = "Install Dependencies"
    bl_options = {"REGISTER"}
    force_reinstall: bpy.props.BoolProperty(name="Force Reinstall", default=False)

    @classmethod
    def poll(cls, context):
        return not _dep_state["running"]

    def execute(self, context):
        global _dep_worker_thread, _dep_failed_batches
        if _dep_state["running"]:
            self.report({"WARNING"}, "Dependency operation already running.")
            return {"CANCELLED"}

        import platform as _plat
        pybin     = python_exec()
        addon_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        _linux_req = os.path.join(addon_dir, "requirements_linux.txt")
        local_req = _linux_req if _plat.system() == "Linux" and os.path.exists(_linux_req) \
                    else os.path.join(addon_dir, "requirements.txt")
        mgr       = DependencyManager()

        # Build batch list on main thread (fast — file reads + importlib checks only)
        batches = []
        _use_binary_only = _plat.system() == "Windows"
        if os.path.exists(local_req):
            with open(local_req, 'r') as f:
                raw = f.read().splitlines()
            safe = BlenderInternalManager.filter_list(raw)
            lines = safe if self.force_reinstall else SmartSkipManager.filter_existing(safe)
            if lines:
                batches.append(("Base", _use_binary_only, lines))

        for phase_name, phase_lines in [
            ("SourceLibs", mgr.get_phase_1_5_source_libs()),
            ("Torch",      mgr.get_phase_2_torch()),
            ("Git",        mgr.get_phase_3_git_and_extensions()),
        ]:
            safe = BlenderInternalManager.filter_list(phase_lines)
            lines = safe if self.force_reinstall else SmartSkipManager.filter_existing(safe)
            if lines:
                batches.append((phase_name, False, lines))

        # Linux: thinc/spacy must be installed as binary wheels (Python 3.13 Cython compat)
        if _plat.system() == "Linux":
            linux_bin = mgr.get_phase_linux_binary_only()
            safe_bin = BlenderInternalManager.filter_list(linux_bin)
            bin_lines = safe_bin if self.force_reinstall else SmartSkipManager.filter_existing(safe_bin)
            if bin_lines:
                batches.append(("LinuxBinary", True, bin_lines))

        total_pkgs = sum(len(b[2]) for b in batches)

        _dep_cancel_event.clear()
        _dep_failed_batches.clear()
        _dep_state.update({
            "running": True, "op_type": "install",
            "progress": 0.0, "phase": "Starting...",
            "status_line": "", "failure_report": "",
        })

        snapshot = {"pybin": pybin, "addon_dir": addon_dir,
                    "batches": batches, "total_pkgs": total_pkgs}
        _dep_worker_thread = threading.Thread(
            target=_run_install, args=(snapshot, _dep_cancel_event), daemon=True
        )
        _dep_worker_thread.start()

        if not bpy.app.timers.is_registered(_dep_tick):
            bpy.app.timers.register(_dep_tick, first_interval=_DEP_TIMER_INTERVAL)

        self.report({"INFO"}, "Dependency installation started in the background.")
        return {"FINISHED"}


class GENERATOR_OT_uninstall(Operator):
    bl_idname = "sequencer.uninstall_generator"
    bl_label = "Uninstall Dependencies"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return not _dep_state["running"]

    def execute(self, context):
        global _dep_worker_thread, _dep_failed_batches
        if _dep_state["running"]:
            self.report({"WARNING"}, "Dependency operation already running.")
            return {"CANCELLED"}

        _dep_cancel_event.clear()
        _dep_failed_batches.clear()
        _dep_state.update({
            "running": True, "op_type": "uninstall",
            "progress": 0.0, "phase": "Starting uninstall...",
            "status_line": "", "failure_report": "",
        })

        _dep_worker_thread = threading.Thread(
            target=_run_uninstall, args=({}, _dep_cancel_event), daemon=True
        )
        _dep_worker_thread.start()

        if not bpy.app.timers.is_registered(_dep_tick):
            bpy.app.timers.register(_dep_tick, first_interval=_DEP_TIMER_INTERVAL)

        self.report({"INFO"}, "Dependency uninstall started in the background.")
        return {"FINISHED"}


class GENERATOR_OT_cancel_dep_op(Operator):
    bl_idname = "sequencer.cancel_dep_op"
    bl_label = "Cancel"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return _dep_state["running"]

    def execute(self, context):
        _dep_cancel_event.set()
        _dep_state["phase"] = "Cancelling..."
        return {"FINISHED"}


class GENERATOR_OT_copy_install_report(Operator):
    bl_idname = "sequencer.copy_install_report"
    bl_label = "Copy Error Report"
    bl_description = "Copy the installation error report to clipboard for a GitHub bug report"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return bool(_dep_state.get("failure_report", ""))

    def execute(self, context):
        report = _dep_state.get("failure_report", "")
        if report:
            bpy.context.window_manager.clipboard = report
            self.report({"INFO"}, "Error report copied to clipboard.")
        else:
            self.report({"WARNING"}, "No error report available.")
        return {"FINISHED"}


class GENERATOR_OT_dismiss_install_errors(Operator):
    bl_idname = "sequencer.dismiss_install_errors"
    bl_label = "Dismiss"
    bl_description = "Clear the install error banner and delete the saved error report"
    bl_options = {"REGISTER"}

    def execute(self, context):
        clear_install_errors_file()
        _dep_state["failure_report"] = ""
        _dep_failed_batches.clear()
        try:
            prefs = _get_dep_prefs()
            if prefs:
                prefs.dep_has_errors     = False
                prefs.dep_failure_report = ""
        except Exception:
            pass
        for area in context.screen.areas:
            area.tag_redraw()
        self.report({"INFO"}, "Install error report cleared.")
        return {"FINISHED"}


class GENERATOR_OT_sound_notification(Operator):
    """Test your notification settings"""

    bl_idname = "renderreminder.pallaidium_play_notification"
    bl_label = "Test Notification"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        preferences = context.preferences
        addon_prefs = preferences.addons[ADDON_ID].preferences
        if addon_prefs.playsound:
            device = aud.Device()

            def coinSound():
                sound = aud.Sound("")
                handle = device.play(
                    sound.triangle(1000)
                    .highpass(20)
                    .lowpass(2000)
                    .ADSR(0, 0.5, 1, 0)
                    .fadeout(0.1, 0.1)
                    .limit(0, 1)
                )
                handle = device.play(
                    sound.triangle(1500)
                    .highpass(20)
                    .lowpass(2000)
                    .ADSR(0, 0.5, 1, 0)
                    .fadeout(0.2, 0.2)
                    .delay(0.1)
                    .limit(0, 1)
                )

            def ding():
                sound = aud.Sound("")
                handle = device.play(
                    sound.triangle(3000)
                    .highpass(20)
                    .lowpass(1000)
                    .ADSR(0, 0.5, 1, 0)
                    .fadeout(0, 1)
                    .limit(0, 1)
                )

            if addon_prefs.soundselect == "ding":
                ding()
            if addon_prefs.soundselect == "coin":
                coinSound()
            if addon_prefs.soundselect == "user":
                file = str(addon_prefs.usersound)
                if os.path.isfile(file):
                    sound = aud.Sound(file)
                    handle = device.play(sound)
        return {"FINISHED"}


class LORA_OT_RefreshFiles(Operator):
    bl_idname = "lora.refresh_files"
    bl_label = "Refresh Files"

    def execute(self, context):
        scene = context.scene
        directory = bpy.path.abspath(scene.lora_folder)
        lora_files = scene.lora_files
        lora_files.clear()
        if not directory:
            self.report({"ERROR"}, "No folder selected")
            return {"CANCELLED"}
        for filename in os.listdir(directory):
            if filename.endswith(".safetensors"):
                file_item = lora_files.add()
                file_item.name = filename.replace(".safetensors", "")
                file_item.enabled = False
                file_item.weight_value = 1.0
        return {"FINISHED"}


class IPAdapterFaceFileBrowserOperator(Operator):
    bl_idname = "ip_adapter_face.file_browser"
    bl_label = "Open IP Adapter Face File Browser"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    import_as_folder: bpy.props.BoolProperty(name="Import as Folder", default=False)

    def execute(self, context):
        valid_image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif", ".hdr"}
        scene = context.scene

        if self.filepath:
            if self.import_as_folder:
                files_to_import = bpy.context.scene.ip_adapter_face_files_to_import
                files_to_import.clear()
                print("Importing folder:", self.filepath)
                for file_path in glob.glob(os.path.join(self.filepath, "*")):
                    if os.path.isfile(file_path):
                        if os.path.splitext(file_path)[1].lower() in valid_image_extensions:
                            print("Found image file in folder:", os.path.basename(file_path))
                            new_file = files_to_import.add()
                            new_file.path = os.path.abspath(self.filepath)
                scene.ip_adapter_face_folder = os.path.abspath(os.path.dirname(self.filepath))
                self.report({"INFO"}, f"{len(files_to_import)} image files found in folder.")
            else:
                print("Importing file:", self.filepath)
                if os.path.splitext(self.filepath)[1].lower() in valid_image_extensions:
                    print("Adding image file:", os.path.basename(self.filepath))
                    files_to_import = bpy.context.scene.ip_adapter_face_files_to_import
                    new_file = files_to_import.add()
                    new_file.name = os.path.abspath(self.filepath)
                    self.report({"INFO"}, "Image file added.")
                    scene.ip_adapter_face_folder = os.path.abspath(self.filepath)
                else:
                    self.report({"ERROR"}, "Selected file is not a valid image.")
        else:
            self.report({"ERROR"}, "No file selected.")
        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class IPAdapterStyleFileBrowserOperator(Operator):
    bl_idname = "ip_adapter_style.file_browser"
    bl_label = "Open IP Adapter Style File Browser"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    import_as_folder: bpy.props.BoolProperty(name="Import as Folder", default=False)

    def execute(self, context):
        valid_image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif", ".hdr"}
        scene = context.scene

        if self.filepath:
            if self.import_as_folder:
                files_to_import = bpy.context.scene.ip_adapter_style_files_to_import
                files_to_import.clear()
                self.filepath = os.path.dirname(self.filepath)
                print("Importing folder:", self.filepath)
                for file_path in glob.glob(os.path.join(self.filepath, "*")):
                    if os.path.isfile(file_path):
                        if os.path.splitext(file_path)[1].lower() in valid_image_extensions:
                            print("Found image file in folder:", os.path.basename(file_path))
                            new_file = files_to_import.add()
                            new_file.name = os.path.basename(file_path)
                            new_file.path = os.path.abspath(file_path)
                scene.ip_adapter_style_folder = os.path.abspath(self.filepath)
                self.report({"INFO"}, f"{len(files_to_import)} image files found in folder.")
            else:
                print("Importing file:", self.filepath)
                if os.path.splitext(self.filepath)[1].lower() in valid_image_extensions:
                    print("Adding image file:", os.path.basename(self.filepath))
                    files_to_import = bpy.context.scene.ip_adapter_style_files_to_import
                    new_file = files_to_import.add()
                    new_file.name = os.path.basename(self.filepath)
                    new_file.path = os.path.abspath(self.filepath)
                    self.report({"INFO"}, "Image file added.")
                    scene.ip_adapter_style_folder = os.path.abspath(self.filepath)
                else:
                    self.report({"ERROR"}, "Selected file is not a valid image.")
        else:
            self.report({"ERROR"}, "No file selected.")
        return {"FINISHED"}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}
