"""FLUX.3 layout prompts <-> Box Editor (Ideogram-4 style) JSON.

FLUX.3 takes layout as a caption that tags regions with ``<id>``, followed by a
JSON list of regions::

    A fox <obj_1> on a bench <obj_2>. [{"id": "obj_1", "bbox": [y0,x0,y1,x1], "desc": "..."}, ...]

In an edit every row uses ``{"id", "from", "src_bbox", "tgt_bbox", "desc"}``
instead of ``bbox``: ``from`` is "ref_image_N" for an item visible in reference
N (``src_bbox`` = where it is there) or "target" for newly created content
(``src_bbox`` = null). ``tgt_bbox`` is where it ends up, null for a removal.

Boxes are [top, left, bottom, right] on a 0-1000 grid, exactly the Box Editor's
Ideogram convention, so only the text around them needs translating. (BFL's CLI
boxes.json uses 0-1 instead; normalize_boxes() rescales those.) Pure Python (no
bpy) so the queue worker can call it too.
"""

import json
import re

_COLORS_RE = re.compile(r";\s*colors\s+(#[0-9A-Fa-f]{6}(?:\s*,\s*#[0-9A-Fa-f]{6})*)\s*$")
_TEXT_RE = re.compile(r'^Text\s+"(.*?)"\s*(?:;\s*(.*))?$', re.S)
_REGIONS_START_RE = re.compile(r"\[\s*\{")


def parse_ideogram_json(text: str):
    """Return the Box Editor / Ideogram dict in *text*, or None.

    ``//`` comment lines (the Box Editor's ``// image: path`` header) are skipped.
    """
    if not text or "compositional_deconstruction" not in text:
        return None
    body = "\n".join(l for l in text.splitlines() if not l.strip().startswith("//"))
    try:
        data = json.loads(body)
    except Exception:
        return None
    if isinstance(data, dict) and isinstance(data.get("compositional_deconstruction"), dict):
        return data
    return None


def split_flux3_layout(prompt: str):
    """Split a FLUX.3 layout prompt into (caption, regions); regions is None if absent."""
    for m in _REGIONS_START_RE.finditer(prompt or ""):
        try:
            regions = json.loads(prompt[m.start():].strip())
        except Exception:
            continue
        if (isinstance(regions, list) and regions
                and all(isinstance(r, dict) and "id" in r for r in regions)):
            return prompt[:m.start()].strip(), regions
    return (prompt or "").strip(), None


_BOX_KEYS = ("bbox", "src_bbox", "tgt_bbox")


def normalize_boxes(prompt: str) -> str:
    """Rescale a layout prompt whose boxes are all on BFL's 0-1 CLI scale to 0-1000.

    The model reads 0-1000; a pasted boxes.json-style table (every coordinate
    <= 1) would otherwise squeeze everything into the top-left corner.
    """
    caption, rows = split_flux3_layout(prompt)
    if not rows:
        return prompt
    values = [v for r in rows for k in _BOX_KEYS if isinstance(r.get(k), list) for v in r[k]]
    if not values or not all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in values):
        return prompt
    for r in rows:
        for k in _BOX_KEYS:
            if isinstance(r.get(k), list):
                r[k] = [round(float(v) * 1000) for v in r[k]]
    return f"{caption} {json.dumps(rows, ensure_ascii=False)}"


def _short(desc: str, words: int = 10) -> str:
    """First clause of *desc*, at most *words* words: a caption-sized mention."""
    clause = re.split(r"[.;:,\n]", desc or "", maxsplit=1)[0].strip()
    parts = clause.split()
    if not parts:
        return "an element"
    # Lower-case a leading capital so the mention reads mid-sentence; keep
    # acronyms ("UFO") as written.
    first = parts[0]
    if first[0].isupper() and (len(first) == 1 or first[1:].islower()):
        parts[0] = first.lower()
    return " ".join(parts[:words])


def _with_colors(desc: str, palette) -> str:
    return f"{desc}; colors {', '.join(palette)}" if palette else desc


def _target_region(rid: str, bbox, desc: str) -> dict:
    """Newly created content in an edit: no source, only where it goes."""
    return {"id": rid, "from": "target", "src_bbox": None, "tgt_bbox": list(bbox), "desc": desc}


def ideogram_to_flux3(data: dict, edit=None) -> str:
    """Box Editor / Ideogram dict -> FLUX.3 layout prompt (caption + region JSON).

    *edit* selects the edit-table schema, where new regions become "target"
    rows. None decides from the data: the layout's "edit" flag, or any element
    with a source box.
    """
    comp = data.get("compositional_deconstruction") or {}
    style = data.get("style_description") or {}
    caption = (data.get("high_level_description") or "").strip()
    if edit is None:
        edit = bool(comp.get("edit")) or any(e.get("src_bbox") for e in comp.get("elements") or [])

    regions, mentions = [], []
    background = (comp.get("background") or "").strip()
    if background:
        bg_id = comp.get("background_id") or "background_1"
        full = [0, 0, 1000, 1000]
        regions.append(_target_region(bg_id, full, background) if edit
                       else {"id": bg_id, "bbox": full, "desc": background})
        mentions.append((bg_id, f"the background, {_short(background, 8)}", "add", ""))

    used = {r["id"] for r in regions}
    counters = {"obj": 0, "text": 0}
    for elem in comp.get("elements") or []:
        kind = "text" if elem.get("type") == "text" else "obj"
        rid = (elem.get("id") or "").strip()
        if not rid or rid in used:
            while True:
                counters[kind] += 1
                rid = f"{kind}_{counters[kind]}"
                if rid not in used:
                    break
        used.add(rid)

        desc = (elem.get("desc") or "").strip()
        if kind == "text":
            text = (elem.get("text") or "").strip()
            full = f'Text "{text}"; {desc}' if desc else f'Text "{text}"'
            mention = f'the text "{text}"'
        else:
            full = desc or "an element"
            mention = _short(desc)
        full = _with_colors(full, elem.get("color_palette") or [])

        region = {"id": rid}
        bbox = list(elem.get("bbox") or [0, 0, 1000, 1000])
        src = elem.get("src_bbox")
        ref = ""
        if src:
            ref = f"ref_image_{int(elem.get('from_ref', 0) or 0)}"
            region["from"] = ref
            region["src_bbox"] = list(src)
            region["tgt_bbox"] = None if elem.get("remove") else bbox
            verb = "remove" if elem.get("remove") else ("keep" if list(src) == bbox else "move")
            region["desc"] = full
        elif edit:
            region = _target_region(rid, bbox, full)
            verb = "add"
        else:
            region["bbox"] = bbox
            region["desc"] = full
            verb = "add"
        regions.append(region)
        mentions.append((rid, mention, verb, ref))

    # Mention every region the caption does not tag yet: new regions in one
    # "Featuring" list, layout edits as instructions. A caption coming back from
    # flux3_to_ideogram already carries its tags, so a round trip is stable.
    missing = [m for m in mentions if f"<{m[0]}>" not in caption]
    clauses = []
    added = [f"{m} <{rid}>" for rid, m, verb, _ref in missing if verb == "add"]
    if added:
        clauses.append(f"{'Add' if edit else 'Featuring'} {'; '.join(added)}.")
    for rid, m, verb, ref in missing:
        if verb == "move":
            clauses.append(f"Move {m} <{rid}> from <{ref}> to its new position.")
        elif verb == "keep":
            clauses.append(f"Keep {m} <{rid}> from <{ref}> unchanged.")
        elif verb == "remove":
            clauses.append(f"Remove {m} <{rid}> from <{ref}>.")
    if clauses:
        lead = f"{caption.rstrip('. ')}. " if caption else ""
        caption = lead + " ".join(clauses)

    style_bits = [style.get(k, "") for k in ("medium", "art_style", "photo", "aesthetics", "lighting")]
    style_bits = [s.strip() for s in style_bits if s and s.strip()]
    if style.get("color_palette"):
        style_bits.append("palette " + ", ".join(style["color_palette"]))
    if style_bits:
        caption = f"{caption} Style: {'; '.join(style_bits)}."

    if not regions:
        return caption
    return f"{caption} {json.dumps(regions, ensure_ascii=False)}"


def flux3_to_ideogram(prompt: str):
    """FLUX.3 layout prompt -> Box Editor / Ideogram dict, or None without regions."""
    caption, regions = split_flux3_layout(normalize_boxes(prompt))
    if regions is None:
        return None

    edit = any("from" in r for r in regions)
    background, background_id, elements = "", "", []
    for r in regions:
        rid = str(r.get("id", ""))
        desc = str(r.get("desc", "") or "")
        new_in_edit = r.get("from") == "target"
        # New content ("target" rows) is a plain box in the Box Editor; the
        # layout's edit flag brings the target schema back on export.
        bbox = r.get("tgt_bbox") if new_in_edit else r.get("bbox")
        if (not background and rid.startswith("background") and ("from" not in r or new_in_edit)
                and list(bbox or []) == [0, 0, 1000, 1000]):
            background, background_id = desc, rid
            continue

        palette = []
        m = _COLORS_RE.search(desc)
        if m:
            palette = [c.strip().upper() for c in m.group(1).split(",")]
            desc = desc[:m.start()].rstrip()

        elem = {"id": rid, "type": "obj", "desc": desc}
        t = _TEXT_RE.match(desc)
        if t:
            elem.update(type="text", text=t.group(1), desc=(t.group(2) or "").strip())
        if palette:
            elem["color_palette"] = palette

        if not new_in_edit and ("from" in r or r.get("src_bbox")):
            src = r.get("src_bbox") or [0, 0, 1000, 1000]
            tgt = r.get("tgt_bbox")
            elem["src_bbox"] = list(src)
            ref = re.search(r"(\d+)$", str(r.get("from", "ref_image_0")))
            elem["from_ref"] = int(ref.group(1)) if ref else 0
            elem["remove"] = tgt is None
            elem["bbox"] = list(src if tgt is None else tgt)
        else:
            elem["bbox"] = list(bbox or [0, 0, 1000, 1000])
        elements.append(elem)

    comp = {"background": background, "elements": elements}
    if background_id:
        comp["background_id"] = background_id
    if edit:
        comp["edit"] = True
    return {
        "high_level_description": caption,
        "style_description": {},
        "compositional_deconstruction": comp,
    }


def to_flux3_prompt(prompt: str, n_refs: int = 0) -> str:
    """Prompt as FLUX.3 should receive it.

    Box Editor JSON is converted; with reference images it uses the edit-table
    schema (new boxes become "target" rows). A FLUX.3 layout prompt passes
    through, rescaled to 0-1000 if its boxes are on BFL's 0-1 CLI scale.
    """
    data = parse_ideogram_json(prompt)
    if data is not None:
        return ideogram_to_flux3(data, edit=True if n_refs else None)
    return normalize_boxes(prompt or "")


# ---------------------------------------------------------------------------
# Prompt upsampler (black-forest-labs/flux-3-image-klein-pup) request protocol,
# following BFL's reference inference code (flux3/pup.py, flux3/bbox.py).
# ---------------------------------------------------------------------------

_ID_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")

PUP_I2I_SCAFFOLD = (
    "Reference images follow in order (the first image, the second image, ...).\n"
    "User request: {instruction}\n"
    "Output the caption."
)
PUP_I2I_BBOX_SCAFFOLD = (
    "Reference images follow in attachment order: ref_image_0, ref_image_1, ...\n"
    "User editing instruction: {instruction}\n"
    "Decide the edit, imagine the target image, and output the annotation record."
)


def _valid_box(box):
    """A [y0, x0, y1, x1] box on the 0-1000 grid, or None."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or v != v for v in box):
        return None
    y1, x1, y2, x2 = box
    if y1 < 0 or x1 < 0 or y2 <= y1 or x2 <= x1 or y2 > 1000 or x2 > 1000 or max(box) <= 1:
        return None
    return list(box)


def _box_rows(prompt: str):
    """(caption, rows) when *prompt* carries a valid FLUX.3 box table, else None."""
    caption, rows = split_flux3_layout(prompt)
    if not rows or not all(
        isinstance(r.get("id"), str) and _ID_RE.fullmatch(r["id"])
        and any(_valid_box(r.get(k)) for k in ("bbox", "src_bbox", "tgt_bbox"))
        for r in rows
    ):
        return None
    return caption, rows


def is_finished_layout(prompt: str) -> bool:
    """True for an upsampler answer (every box id is cited as <id>): nothing left to upsample."""
    found = _box_rows(prompt)
    return found is not None and all(f"<{r['id']}>" in found[0] for r in found[1])


def pup_request(prompt: str, n_images: int, width: int = 0, height: int = 0):
    """(system prompt key, user text, locked rows or None) for the upsampler.

    Boxes in the prompt switch to the *_bbox prompts, which keep them as locked
    annotations. Text-to-image requests get a "TARGET CANVAS: W:H" line, which
    the t2i prompts read to compose for the output aspect ratio.
    """
    task = "i2i" if n_images else "t2i"
    found = _box_rows(prompt)
    if found is None:
        key, text, locked = task, prompt.strip(), None
    else:
        caption, locked = found
        key = f"{task}_bbox"
        text = (
            f"ORIGINAL PROMPT:\n{caption}\n\n"
            "LOCKED BOUNDING-BOX ANNOTATIONS ([y_min, x_min, y_max, x_max] on a 0-1000 canvas, with (0, 0) at "
            "the top-left). This is an authoritative JSON list of dictionaries; preserve every supplied identity, "
            f"provenance value, null, and coordinate exactly:\n{json.dumps(locked, ensure_ascii=False)}"
        )
    if task == "t2i" and width > 0 and height > 0:
        from math import gcd
        g = gcd(int(width), int(height)) or 1
        text = f"TARGET CANVAS: {int(width) // g}:{int(height) // g}\n{text}"
    elif task == "i2i":
        scaffold = PUP_I2I_BBOX_SCAFFOLD if locked is not None else PUP_I2I_SCAFFOLD
        text = scaffold.format(instruction=text)
    return key, text, locked


def _iou(a, b) -> float:
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _valid_added_row(row, caption: str):
    """A row the upsampler added, if it is cited in the caption and well formed."""
    if not isinstance(row, dict):
        return None
    rid, desc = row.get("id"), row.get("desc")
    if (not isinstance(rid, str) or not _ID_RE.fullmatch(rid) or f"<{rid}>" not in caption
            or not isinstance(desc, str) or not desc.strip()):
        return None
    box = _valid_box(row.get("bbox"))
    if box is not None:
        return {"id": rid, "bbox": box, "desc": desc.strip()}
    origin, src, tgt = row.get("from"), row.get("src_bbox"), row.get("tgt_bbox")
    src_box = _valid_box(src) if src is not None else None
    tgt_box = _valid_box(tgt) if tgt is not None else None
    if (not isinstance(origin, str) or not re.fullmatch(r"ref_image_\d+|target", origin)
            or (src is not None and src_box is None) or (tgt is not None and tgt_box is None)
            or (src_box is None and tgt_box is None) or (origin == "target") == (src_box is not None)):
        return None
    return {"id": rid, "from": origin, "src_bbox": src_box, "tgt_bbox": tgt_box, "desc": desc.strip()}


def pup_reconcile(answer: str, locked) -> str:
    """The upsampler's answer with the user's (locked) boxes re-imposed.

    A row that re-plans a locked box under its own id (IoU >= 0.9) is renamed to
    the locked id, in the rows and in the caption. Locked rows keep their boxes
    verbatim and adopt the upsampler's richer desc; valid added rows follow.
    """
    caption, rows = split_flux3_layout(answer)
    rows = [r for r in (rows or []) if isinstance(r, dict)]
    locked_ids = {r["id"] for r in locked}
    row_ids = {r.get("id") for r in rows}
    renamed = {}
    for lock in locked:
        lock_box = lock.get("bbox") or lock.get("tgt_bbox") or lock.get("src_bbox")
        if lock["id"] in row_ids or lock_box is None:
            continue
        for row in rows:
            rid = row.get("id")
            if rid in renamed or rid in locked_ids:
                continue
            box = _valid_box(row.get("bbox") or row.get("tgt_bbox") or row.get("src_bbox"))
            if box is not None and _iou(box, lock_box) >= 0.9:
                renamed[rid] = lock["id"]
                break
    for row in rows:
        if row.get("id") in renamed:
            row["id"] = renamed[row["id"]]
    for old, new in renamed.items():
        caption = caption.replace(f"<{old}>", f"<{new}>")

    by_id = {r.get("id"): r for r in rows if isinstance(r.get("id"), str)}
    out = []
    for lock in locked:
        desc = by_id.get(lock["id"], {}).get("desc")
        out.append({**lock, "desc": desc.strip() if isinstance(desc, str) and desc.strip() else lock.get("desc", "")})
    taken = {r["id"] for r in out}
    for row in rows:
        added = _valid_added_row(row, caption)
        if added is not None and added["id"] not in taken:
            out.append(added)
            taken.add(added["id"])
    return f"{caption} {json.dumps(out, ensure_ascii=False)}"


def pup_reference_size(width: int, height: int, n_refs: int):
    """Size budget the upsampler saw its references at in training (multiples of 32)."""
    import math
    cap = {1: 1024 * 1024, 2: 724 * 724}.get(n_refs, 512 * 512)
    floor32 = lambda v: max(32, int(math.floor(v / 32)) * 32)
    w = max(32, round(width / 32) * 32)
    h = max(32, round(height / 32) * 32)
    if w * h > cap:
        beta = math.sqrt(width * height / cap)
        w, h = floor32(width / beta), floor32(height / beta)
    if w * h > cap:
        if w == 32:
            h = floor32(cap / w)
        else:
            w = floor32(cap / h)
    return w, h
