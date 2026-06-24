"""
Plugin discovery and registry for Pallaidium Generative AI.

Scans models_plugins/ on import, instantiates every ModelPlugin subclass found,
and builds PLUGIN_REGISTRY (MODEL_ID → plugin instance) plus per-type item lists
for Blender EnumProperty dropdowns.

Plugin files can use relative imports (e.g. from ...utils.helpers import …)
because the loader registers synthetic parent-package entries in sys.modules
so Python's import machinery resolves them correctly.

Usage:
    from ..models import PLUGIN_REGISTRY, get_enum_items, get_plugin
"""

import importlib.util
import os
import sys
import traceback
import types
from pathlib import Path

# Keep SDNQ's torch.compile / Triton-matmul fast paths OFF.
#
# `import triton` and raw Triton JIT now build on this Windows box (see
# __init__.py::_configure_triton_build), but torch's *Inductor* codegen is NOT
# compatible with the installed triton-windows build: SDNQ's compiled dequantizer
# (dequantize_packed_*_compiled → torch.compile → Inductor) crashes with
# "'KernelMetadata' object has no attribute 'cluster_dims'", taking down every
# SDNQ plugin (ernie/ernie_turbo/krea2). Until the triton↔torch version skew is
# resolved, force the eager dequant path. (use_quantized_matmul is gated on the
# same flag in plugins, so this also keeps that numerically-broken int8 path off.)
# Must be set before SDNQ's first import (i.e. before any from_pretrained).
os.environ.setdefault("SDNQ_USE_TORCH_COMPILE", "0")
os.environ.setdefault("SDNQ_USE_TRITON_MM", "0")

from .base import ModelPlugin, UISection, InputSpec, ParamSpec

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PLUGIN_REGISTRY: dict[str, ModelPlugin] = {}

# MODEL_ID -> "local" | "remote".  Lets get_enum_items() filter the dropdowns by
# the user's Local / Remote / Local&Remote preference.
_PLUGIN_ORIGIN: dict[str, str] = {}

_ENUM_ITEMS: dict[str, list] = {
    "video": [],
    "image": [],
    "audio": [],
    "text":  [],
}

_PLUGINS_DIR = Path(__file__).parent.parent / "models_plugins"

# Root package name, e.g. "bl_ext.user_default.pallaidium_generative_ai"
_ROOT_PKG = __package__.rsplit(".", 1)[0]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _ensure_fake_package(pkg_name: str, path: str) -> None:
    """Register a synthetic package in sys.modules if not already present.

    This lets plugin files use relative imports like:
        from ...utils.helpers import gfx_device
    where '...' resolves back to the add-on root package.
    """
    if pkg_name not in sys.modules:
        fake = types.ModuleType(pkg_name)
        fake.__path__ = [path]
        fake.__package__ = pkg_name
        sys.modules[pkg_name] = fake


def _load_plugin_file(py_file: Path) -> list[ModelPlugin]:
    """Import one plugin file and return all ModelPlugin instances found in it.

    Sets __package__ on the loaded module so that relative imports resolve
    correctly relative to the add-on root package.
    """
    type_dir    = py_file.parent.name                          # e.g. "text"
    plugins_pkg = f"{_ROOT_PKG}.models_plugins"               # e.g. "...pallaidium_generative_ai.models_plugins"
    type_pkg    = f"{plugins_pkg}.{type_dir}"                  # e.g. "...models_plugins.text"
    module_name = f"{type_pkg}.{py_file.stem}"                 # e.g. "...models_plugins.text.blip_caption"

    # Register synthetic parent packages so relative imports resolve
    _ensure_fake_package(plugins_pkg, str(_PLUGINS_DIR))
    _ensure_fake_package(type_pkg, str(py_file.parent))

    spec = importlib.util.spec_from_file_location(module_name, py_file)
    if spec is None or spec.loader is None:
        return []

    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = type_pkg  # critical: lets '...' in relative imports resolve correctly
    sys.modules[module_name] = mod

    try:
        spec.loader.exec_module(mod)
    except ModuleNotFoundError as e:
        print(f"[Pallaidium] Plugin {py_file.name} skipped: missing dependency ({e.name}). Install dependencies first.")
        sys.modules.pop(module_name, None)
        return []
    except Exception:
        print(f"[Pallaidium] Plugin {py_file.name} failed to load:")
        traceback.print_exc()
        # Remove broken module from sys.modules to allow retry on next discovery
        sys.modules.pop(module_name, None)
        return []

    instances = []
    for obj in vars(mod).values():
        if (
            isinstance(obj, type)
            and issubclass(obj, ModelPlugin)
            and obj is not ModelPlugin
            and obj.MODEL_ID
        ):
            try:
                inst = obj()
            except Exception:
                print(f"[Pallaidium] Could not instantiate {obj.__name__} in {py_file.name}:")
                traceback.print_exc()
                continue

            errors = _validate_plugin(inst, py_file.name)
            if errors:
                for err in errors:
                    print(f"[Pallaidium] {obj.__name__} ({py_file.name}): {err}")
                print(f"[Pallaidium] {obj.__name__} skipped due to validation errors above.")
                continue

            instances.append(inst)

    return instances


_VALID_MODEL_TYPES = {"video", "image", "audio", "text"}
_VALID_UI_SECTIONS = {s.value for s in UISection}


def _validate_plugin(inst: ModelPlugin, filename: str) -> list[str]:
    """Return a list of human-readable error strings, empty if plugin is valid."""
    errors = []

    if not inst.MODEL_ID:
        errors.append("MODEL_ID is empty")
    if not inst.DISPLAY_NAME:
        errors.append("DISPLAY_NAME is empty")
    if inst.MODEL_TYPE not in _VALID_MODEL_TYPES:
        errors.append(
            f"MODEL_TYPE {inst.MODEL_TYPE!r} is not one of {sorted(_VALID_MODEL_TYPES)}"
        )

    if not isinstance(inst.PARAMS, ParamSpec):
        errors.append(
            f"PARAMS must be a ParamSpec instance, got {type(inst.PARAMS).__name__!r}"
        )

    if not isinstance(inst.INPUTS, InputSpec):
        errors.append(
            f"INPUTS must be an InputSpec flag, got {type(inst.INPUTS).__name__!r}"
        )

    if not isinstance(inst.UI_SECTIONS, list):
        errors.append("UI_SECTIONS must be a list")
    else:
        for sec in inst.UI_SECTIONS:
            if not isinstance(sec, UISection):
                errors.append(
                    f"UI_SECTIONS contains invalid entry {sec!r} "
                    f"(must be a UISection member, e.g. UISection.PROMPT)"
                )

    return errors


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover(plugins_dir: Path = _PLUGINS_DIR) -> None:
    """Scan plugins_dir recursively, load every plugin, populate the registry.

    Files/directories whose names start with '_' are skipped (covers
    _template.py, _README.md, __pycache__, __init__.py, etc.).

    Safe to call again to reload during development.
    """
    PLUGIN_REGISTRY.clear()
    _PLUGIN_ORIGIN.clear()
    for v in _ENUM_ITEMS.values():
        v.clear()

    if not plugins_dir.exists():
        print(f"[Pallaidium] models_plugins directory not found: {plugins_dir}")
        return

    py_files = sorted(
        (
            f for f in plugins_dir.rglob("*.py")
            if not any(
                part.startswith("_")
                for part in f.parts[len(plugins_dir.parts):]
            )
        ),
        key=lambda f: (f.parent.name, f.name),
    )

    for py_file in py_files:
        for inst in _load_plugin_file(py_file):
            if inst.MODEL_ID in PLUGIN_REGISTRY:
                print(
                    f"[Pallaidium] Duplicate MODEL_ID {inst.MODEL_ID!r} "
                    f"in {py_file.name} — skipping."
                )
                continue

            PLUGIN_REGISTRY[inst.MODEL_ID] = inst
            _PLUGIN_ORIGIN[inst.MODEL_ID] = "local"

            media_type = inst.MODEL_TYPE
            if media_type in _ENUM_ITEMS:
                _ENUM_ITEMS[media_type].append(
                    (inst.MODEL_ID, inst.DISPLAY_NAME, inst.DESCRIPTION)
                )
                print(f"[Pallaidium] Registered {media_type} plugin: {inst.MODEL_ID}")
            else:
                print(
                    f"[Pallaidium] Plugin {inst.MODEL_ID!r} has unknown "
                    f"MODEL_TYPE {media_type!r} — skipping enum registration."
                )

    print(f"[Pallaidium] Plugin discovery complete: {len(PLUGIN_REGISTRY)} plugin(s) loaded.")


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_enum_items(media_type: str, source: str = "LOCAL") -> list:
    """Return Blender EnumProperty items for a given media type.

    ``source`` ∈ {"LOCAL", "REMOTE", "BOTH"} filters by plugin origin so the
    dropdowns reflect the user's model-source preference. Defaults to "LOCAL"
    so any caller that omits it keeps the original local-only behavior.

    Returns a single placeholder tuple if nothing matches (Blender disallows
    empty EnumProperty item lists).
    """
    items = _ENUM_ITEMS.get(media_type, [])
    if source != "BOTH":
        want = "remote" if source == "REMOTE" else "local"
        items = [it for it in items if _PLUGIN_ORIGIN.get(it[0]) == want]
    if not items:
        label = {
            "REMOTE": f"No remote {media_type} models — click Refresh",
            "BOTH":   f"No {media_type} models found",
        }.get(source, f"No {media_type} plugins found")
        return [(f"__none_{media_type}__", label, "")]
    return items


def clear_remote_models() -> None:
    """Remove all origin=='remote' plugins from the registry and enum lists."""
    remote_ids = [mid for mid, o in _PLUGIN_ORIGIN.items() if o == "remote"]
    for mid in remote_ids:
        PLUGIN_REGISTRY.pop(mid, None)
        _PLUGIN_ORIGIN.pop(mid, None)
    for items in _ENUM_ITEMS.values():
        items[:] = [it for it in items if it[0] not in remote_ids]


def register_remote_models(entries: list, prefs=None) -> int:
    """Build synthetic plugins from /v1/models entries and register them.

    Clears any previously registered remote models first (so Refresh is
    idempotent). Returns the number of remote models registered.
    """
    from .remote_base import make_remote_plugin

    clear_remote_models()
    count = 0
    for entry in entries or []:
        try:
            inst = make_remote_plugin(entry)
        except Exception as e:
            print(f"[Pallaidium] Skipping remote model {entry!r}: {e}")
            continue

        if inst.MODEL_ID in PLUGIN_REGISTRY:
            print(f"[Pallaidium] Remote model {inst.MODEL_ID!r} shadows an "
                  f"existing id — skipping.")
            continue

        errors = _validate_plugin(inst, "<remote>")
        if errors:
            for err in errors:
                print(f"[Pallaidium] remote {inst.MODEL_ID}: {err}")
            continue

        PLUGIN_REGISTRY[inst.MODEL_ID] = inst
        _PLUGIN_ORIGIN[inst.MODEL_ID] = "remote"
        if inst.MODEL_TYPE in _ENUM_ITEMS:
            _ENUM_ITEMS[inst.MODEL_TYPE].append(
                (inst.MODEL_ID, inst.DISPLAY_NAME, inst.DESCRIPTION)
            )
        count += 1
        print(f"[Pallaidium] Registered remote {inst.MODEL_TYPE} model: {inst.MODEL_ID}")

    print(f"[Pallaidium] Remote model refresh complete: {count} model(s).")
    return count


def get_plugin(model_id: str) -> "ModelPlugin | None":
    """Return the plugin instance for model_id, or None."""
    return PLUGIN_REGISTRY.get(model_id)


def plugin_types() -> dict[str, list[str]]:
    """Return {media_type: [MODEL_ID, …]} for all registered plugins."""
    result: dict[str, list[str]] = {}
    for model_id, plugin in PLUGIN_REGISTRY.items():
        result.setdefault(plugin.MODEL_TYPE, []).append(model_id)
    return result


# ---------------------------------------------------------------------------
# Run discovery at import time
# ---------------------------------------------------------------------------

discover()
