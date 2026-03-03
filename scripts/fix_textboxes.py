"""Fix textboxes in DOCX: expand clipped containers, enable autofit, fix captions.

Three passes:
  1. Convert noAutofit → normAutofit on all wps:bodyPr (both mc:Choice + mc:Fallback)
  2. Adaptively expand textbox heights (DrawingML extent + VML shape style) based on text content
  3. Insert line breaks before "Рисунок NNN - Лист N" in caption textboxes

Usage:
    python scripts/fix_textboxes.py [--input FILE] [--output FILE]
"""
from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

from lxml import etree
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Namespaces
NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS_WPS = "http://schemas.microsoft.com/office/word/2010/wordprocessingShape"
NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS_V = "urn:schemas-microsoft-com:vml"
NS_WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
NS_MC = "http://schemas.openxmlformats.org/markup-compatibility/2006"

VML_HEIGHT_RE = re.compile(
    r"((?:^|;)\s*height\s*:\s*)([0-9]+(?:\.[0-9]+)?)(pt|in|cm|mm|px)?",
    re.IGNORECASE,
)

# Pattern: "Рисунок NNN - Лист N" or "Рисунок NNN" at the end of caption text
RISUNOK_PATTERN = re.compile(r"Рисунок\s+\d+")
RISUNOK_SHEET_PATTERN = re.compile(r"(Рисунок\s+\d+\s*-\s*Лист\s+\d+)")
RISUNOK_ONLY_PATTERN = re.compile(r"(Рисунок\s+\d+)$")


def _local_name(node) -> str:
    tag = str(getattr(node, "tag", "") or "")
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _is_in_mc_fallback(node) -> bool:
    for anc in node.iterancestors():
        if _local_name(anc) == "Fallback":
            return True
    return False


def _get_text(node) -> str:
    return "".join(e.text or "" for e in node.iter(f"{{{NS_W}}}t"))


def _get_max_font_half_pt(node) -> int:
    """Get the maximum font size (in half-pt) across all runs in a node."""
    max_sz = 0
    sz_tag = f"{{{NS_W}}}sz"
    val_attr = f"{{{NS_W}}}val"
    for sz in node.iter(sz_tag):
        raw = sz.get(val_attr, "")
        try:
            val = int(raw)
            if val > max_sz:
                max_sz = val
        except (TypeError, ValueError):
            pass
    return max_sz


def _count_paragraphs(node) -> int:
    return sum(1 for _ in node.iter(f"{{{NS_W}}}p"))


def _estimate_needed_height_emu(
    text: str,
    container_width_emu: int,
    font_half_pt: int,
    n_paragraphs: int,
) -> int:
    """Estimate the height (in EMU) needed to display text in a container.

    1 pt = 12700 EMU.  We assume avg char width ≈ 0.50 * font_size for Cyrillic.
    """
    if not text.strip() or container_width_emu <= 0 or font_half_pt <= 0:
        return 0

    font_pt = font_half_pt / 2.0
    # EMU per pt
    emu_per_pt = 12700.0

    # Available width for text (subtract small margin for safety)
    width_pt = container_width_emu / emu_per_pt

    # Use wider char estimate (0.55) to be conservative
    avg_char_width_pt = font_pt * 0.55

    # Characters per line
    chars_per_line = max(1, width_pt / avg_char_width_pt)

    # Number of lines needed
    text_len = len(text.strip())
    num_lines = max(1, text_len / chars_per_line)

    # Add extra lines for paragraph breaks
    if n_paragraphs > 1:
        num_lines += (n_paragraphs - 1) * 0.5

    # Line height (generous factor)
    line_height_pt = font_pt * 1.35

    # Total height needed + 30% safety margin
    height_pt = num_lines * line_height_pt
    return int(height_pt * emu_per_pt * 1.3)


# ============================================================================
# Pass 1: Enable normAutofit
# ============================================================================

def pass1_enable_norm_autofit(doc) -> dict[str, int]:
    """Convert noAutofit → normAutofit on all wps:bodyPr elements."""
    stats = {"noAutofit_to_normAutofit": 0, "already_normAutofit": 0}

    bodyPr_tag = f"{{{NS_WPS}}}bodyPr"
    normAutofit_tag = f"{{{NS_A}}}normAutofit"
    noAutofit_tag = f"{{{NS_A}}}noAutofit"

    for bp in doc.element.iter(bodyPr_tag):
        no_nodes = list(bp.iter(noAutofit_tag))
        # Only look at direct children
        no_children = [ch for ch in bp if _local_name(ch) == "noAutofit"]
        norm_children = [ch for ch in bp if _local_name(ch) == "normAutofit"]

        if norm_children:
            stats["already_normAutofit"] += 1
            continue

        if no_children:
            for node in no_children:
                bp.remove(node)
            bp.append(OxmlElement("a:normAutofit"))
            stats["noAutofit_to_normAutofit"] += 1

    return stats


# ============================================================================
# Pass 2: Adaptive height expansion
# ============================================================================

def pass2_expand_textbox_heights(doc, *, min_growth: float = 1.0, max_growth: float = 4.0) -> dict[str, int]:
    """Expand textbox containers where text overflows.

    Processes both DrawingML (wp:extent + a:ext) and VML (v:shape style height).

    Expands up to max_growth (2.5x). normAutofit acts as safety net for residual overflow.
    """
    stats = {
        "dml_extents_expanded": 0,
        "dml_spPr_ext_expanded": 0,
        "vml_heights_expanded": 0,
        "textboxes_analyzed": 0,
        "skipped_narrow": 0,
    }

    txbx_tag = f"{{{NS_W}}}txbxContent"
    wsp_tag = f"{{{NS_WPS}}}wsp"
    spPr_tag = f"{{{NS_WPS}}}spPr"
    txbx_wps_tag = f"{{{NS_WPS}}}txbx"
    extent_tag = f"{{{NS_WP}}}extent"
    a_ext_tag = f"{{{NS_A}}}ext"
    a_xfrm_tag = f"{{{NS_A}}}xfrm"

    # Process DrawingML textboxes (mc:Choice)
    for txbx_content in doc.element.iter(txbx_tag):
        if _is_in_mc_fallback(txbx_content):
            continue

        text = _get_text(txbx_content)
        if not text.strip():
            continue

        stats["textboxes_analyzed"] += 1
        n_paras = _count_paragraphs(txbx_content)
        font_half_pt = _get_max_font_half_pt(txbx_content) or 17  # default ~8.5pt

        # Find extent (container size)
        cx, cy = 0, 0
        extent_node = None
        for anc in txbx_content.iterancestors():
            for ext in anc.iter(extent_tag):
                try:
                    cx = int(ext.get("cx", "0"))
                    cy = int(ext.get("cy", "0"))
                    extent_node = ext
                except:
                    pass
                break
            if extent_node is not None:
                break

        if cx <= 0 or cy <= 0 or extent_node is None:
            continue

        # Skip very narrow textboxes (< 15pt wide)
        width_pt = cx / 12700.0
        if width_pt < 15:
            stats["skipped_narrow"] += 1
            continue

        # Estimate needed height
        needed_emu = _estimate_needed_height_emu(text, cx, font_half_pt, n_paras)
        if needed_emu <= cy:
            continue  # Fits already

        # Calculate growth factor
        ratio = needed_emu / cy
        growth = min(max_growth, max(min_growth, ratio))
        if growth <= 1.02:
            continue

        new_cy = int(round(cy * growth))

        # Expand wp:extent
        extent_node.set("cy", str(new_cy))
        stats["dml_extents_expanded"] += 1

        # Also expand a:ext in wps:spPr > a:xfrm > a:ext
        for anc in txbx_content.iterancestors():
            if _local_name(anc) == "wsp":
                spPr = anc.find(spPr_tag)
                if spPr is not None:
                    xfrm = spPr.find(a_xfrm_tag)
                    if xfrm is not None:
                        a_ext = xfrm.find(a_ext_tag)
                        if a_ext is not None:
                            a_ext.set("cy", str(new_cy))
                            stats["dml_spPr_ext_expanded"] += 1
                break

    # Process VML textboxes (incl. mc:Fallback)
    v_shape_tag = f"{{{NS_V}}}shape"
    v_textbox_tag = f"{{{NS_V}}}textbox"

    for shape in doc.element.iter(v_shape_tag):
        textbox = shape.find(v_textbox_tag)
        if textbox is None:
            continue

        text = _get_text(shape)
        if not text.strip():
            continue

        font_half_pt = _get_max_font_half_pt(shape) or 17
        n_paras = _count_paragraphs(shape)

        # Parse style for width and height
        style = shape.get("style", "")
        if not style:
            continue

        h_match = VML_HEIGHT_RE.search(style)
        if h_match is None:
            continue

        try:
            height_val = float(h_match.group(2))
        except (TypeError, ValueError):
            continue
        if height_val <= 0:
            continue

        unit = h_match.group(3) or "pt"

        # Convert height to EMU for comparison
        if unit == "pt":
            height_emu = int(height_val * 12700)
        elif unit == "in":
            height_emu = int(height_val * 914400)
        elif unit == "cm":
            height_emu = int(height_val * 360000)
        elif unit == "mm":
            height_emu = int(height_val * 36000)
        else:
            height_emu = int(height_val * 12700)  # default to pt

        # Get width similarly for estimation
        w_match = re.search(
            r"(?:^|;)\s*width\s*:\s*([0-9]+(?:\.[0-9]+)?)(pt|in|cm|mm|px)?",
            style,
            re.IGNORECASE,
        )
        if w_match:
            try:
                w_val = float(w_match.group(1))
                w_unit = w_match.group(2) or "pt"
                if w_unit == "pt":
                    width_emu = int(w_val * 12700)
                elif w_unit == "in":
                    width_emu = int(w_val * 914400)
                elif w_unit == "cm":
                    width_emu = int(w_val * 360000)
                elif w_unit == "mm":
                    width_emu = int(w_val * 36000)
                else:
                    width_emu = int(w_val * 12700)
            except:
                width_emu = 1000000  # fallback
        else:
            width_emu = 1000000  # fallback

        # Skip very narrow textboxes
        width_pt_vml = width_emu / 12700.0
        if width_pt_vml < 15:
            continue

        # Estimate needed height
        needed_emu = _estimate_needed_height_emu(text, width_emu, font_half_pt, n_paras)
        if needed_emu <= height_emu:
            continue

        ratio = needed_emu / height_emu
        growth = min(max_growth, max(min_growth, ratio))
        if growth <= 1.02:
            continue

        new_height = height_val * growth
        replacement = f"{h_match.group(1)}{new_height:.2f}{unit}"
        updated_style = style[: h_match.start()] + replacement + style[h_match.end() :]
        if updated_style != style:
            shape.set("style", updated_style)
            stats["vml_heights_expanded"] += 1

    return stats


# ============================================================================
# Pass 3: Caption line breaks
# ============================================================================

def pass3_caption_line_breaks(doc) -> dict[str, int]:
    """In caption textboxes, add line break before 'Рисунок NNN - Лист N'.

    Finds textboxes containing 'Рисунок NNN - Лист N' pattern and inserts
    a <w:br/> before it, so it appears on a new line. Also expands the container
    to accommodate the extra line.

    Processes BOTH mc:Choice and mc:Fallback for consistency.
    """
    stats = {"caption_breaks_inserted": 0, "caption_containers_expanded": 0}

    txbx_tag = f"{{{NS_W}}}txbxContent"
    t_tag = f"{{{NS_W}}}t"
    r_tag = f"{{{NS_W}}}r"
    br_tag = f"{{{NS_W}}}br"
    extent_tag = f"{{{NS_WP}}}extent"
    spPr_tag = f"{{{NS_WPS}}}spPr"
    a_xfrm_tag = f"{{{NS_A}}}xfrm"
    a_ext_tag = f"{{{NS_A}}}ext"

    # Pattern: text ends with "... Рисунок NNN - Лист N" or "... Рисунок NNN"
    # We want to split before "Рисунок"
    split_pattern = re.compile(r"^(.*?\S)\s*(Рисунок\s+\d+.*)$", re.DOTALL)

    for txbx_content in doc.element.iter(txbx_tag):
        full_text = _get_text(txbx_content)
        if not RISUNOK_SHEET_PATTERN.search(full_text):
            continue  # Only process captions with "Рисунок N - Лист N"

        # Find the run that contains "Рисунок"
        modified = False
        for p_elem in txbx_content.iter(f"{{{NS_W}}}p"):
            for run in p_elem.iter(r_tag):
                for t_elem in run.iter(t_tag):
                    t_text = t_elem.text or ""
                    if not t_text:
                        continue

                    # Check if this run's text contains "Рисунок" and has text before it
                    m = split_pattern.match(t_text)
                    if m and m.group(1).strip():
                        before_text = m.group(1)
                        risunok_text = m.group(2)

                        # Check if there's already a break just before (avoid double-breaks)
                        prev = t_elem.getprevious()
                        if prev is not None and _local_name(prev) == "br":
                            continue

                        # Split the text: set current to "before" part
                        t_elem.text = before_text

                        # Create a <w:br/> element
                        br_elem = OxmlElement("w:br")

                        # Create a new <w:t> element for the "Рисунок..." part
                        new_t = OxmlElement("w:t")
                        new_t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
                        new_t.text = risunok_text

                        # Insert break and new text after the current t_elem
                        idx = list(run).index(t_elem)
                        run.insert(idx + 1, br_elem)
                        run.insert(idx + 2, new_t)

                        modified = True
                        stats["caption_breaks_inserted"] += 1
                        break  # Only one split per run

                    # Also check cross-run: "Рисунок" at start of a run with previous text run
                    if t_text.strip().startswith("Рисунок") and RISUNOK_SHEET_PATTERN.search(t_text):
                        # Check preceding text
                        prev_run = run.getprevious()
                        if prev_run is not None and _local_name(prev_run) == "r":
                            prev_texts = list(prev_run.iter(t_tag))
                            if prev_texts:
                                last_prev_t = prev_texts[-1]
                                if (last_prev_t.text or "").strip():
                                    # Already separate run - just add break before
                                    prev = t_elem.getprevious()
                                    if prev is not None and _local_name(prev) == "br":
                                        continue  # already has break

                                    br_elem = OxmlElement("w:br")
                                    idx = list(run).index(t_elem)
                                    run.insert(idx, br_elem)

                                    modified = True
                                    stats["caption_breaks_inserted"] += 1
                                    break

                if modified:
                    break
            if modified:
                break

        # Expand container to fit the extra line
        if modified:
            # Find and expand extent for DrawingML
            for anc in txbx_content.iterancestors():
                # Expand wp:extent
                for ext in anc.iter(extent_tag):
                    try:
                        cy = int(ext.get("cy", "0"))
                        font_half_pt = _get_max_font_half_pt(txbx_content) or 17
                        extra_line_emu = int((font_half_pt / 2.0) * 1.25 * 12700)
                        new_cy = cy + extra_line_emu
                        ext.set("cy", str(new_cy))
                        stats["caption_containers_expanded"] += 1
                    except:
                        pass
                    break

                # Expand a:ext in spPr
                if _local_name(anc) == "wsp":
                    spPr = anc.find(spPr_tag)
                    if spPr is not None:
                        xfrm = spPr.find(a_xfrm_tag)
                        if xfrm is not None:
                            a_ext = xfrm.find(a_ext_tag)
                            if a_ext is not None:
                                try:
                                    cy = int(a_ext.get("cy", "0"))
                                    font_half_pt = _get_max_font_half_pt(txbx_content) or 17
                                    extra_line_emu = int((font_half_pt / 2.0) * 1.25 * 12700)
                                    a_ext.set("cy", str(cy + extra_line_emu))
                                except:
                                    pass
                break

            # Also expand VML shape height for fallback
            # Find the corresponding mc:Fallback VML shape
            for anc in txbx_content.iterancestors():
                if _local_name(anc) == "Fallback":
                    for shape in anc.iter(f"{{{NS_V}}}shape"):
                        style = shape.get("style", "")
                        h_match = VML_HEIGHT_RE.search(style)
                        if h_match:
                            try:
                                h_val = float(h_match.group(2))
                                unit = h_match.group(3) or "pt"
                                font_half_pt = _get_max_font_half_pt(shape) or 17
                                if unit == "pt":
                                    extra = (font_half_pt / 2.0) * 1.25
                                else:
                                    extra = h_val * 0.4  # ~40% for extra line
                                new_h = h_val + extra
                                repl = f"{h_match.group(1)}{new_h:.2f}{unit}"
                                updated = style[: h_match.start()] + repl + style[h_match.end() :]
                                shape.set("style", updated)
                            except:
                                pass
                        break
                    break

    return stats


def fix_textboxes(input_path: str, output_path: str) -> dict[str, int]:
    """Main fix function."""
    doc = Document(input_path)
    all_stats: dict[str, int] = {}

    print("Pass 1: Enable normAutofit on all textbox bodyPr...")
    s1 = pass1_enable_norm_autofit(doc)
    all_stats.update(s1)
    print(f"  Converted noAutofit→normAutofit: {s1['noAutofit_to_normAutofit']}")
    print(f"  Already normAutofit: {s1['already_normAutofit']}")

    print("\nPass 2: Adaptive height expansion...")
    s2 = pass2_expand_textbox_heights(doc)
    all_stats.update(s2)
    print(f"  Textboxes analyzed: {s2['textboxes_analyzed']}")
    print(f"  DML extents expanded: {s2['dml_extents_expanded']}")
    print(f"  DML spPr ext expanded: {s2['dml_spPr_ext_expanded']}")
    print(f"  VML heights expanded: {s2['vml_heights_expanded']}")
    print(f"  Skipped narrow: {s2['skipped_narrow']}")

    print("\nPass 3: Caption line breaks...")
    s3 = pass3_caption_line_breaks(doc)
    all_stats.update(s3)
    print(f"  Caption breaks inserted: {s3['caption_breaks_inserted']}")
    print(f"  Caption containers expanded: {s3['caption_containers_expanded']}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)
    print(f"\nSaved: {output_path}")

    return all_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix textboxes in DOCX")
    parser.add_argument(
        "--input", "-i",
        default=r"C:\Users\Urdul\Desktop\project\translator\for_test\Abby\merged_stylefix_fixed_gap05_nogrow_stripped_docwide.docx",
    )
    parser.add_argument(
        "--output", "-o",
        default=r"C:\Users\Urdul\Desktop\project\translator\for_test\Abby\merged_stylefix_fixed_gap05_nogrow_stripped_docwide_tbfix.docx",
    )
    args = parser.parse_args()
    fix_textboxes(args.input, args.output)
