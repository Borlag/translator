Ок, Владимир. По репо видно, что у тебя **layout_check/layout_fix уже есть**, но сейчас “скрытый текст” в рамках/текстбоксах часто остаётся из-за 3 вещей:

1. **Для textbox/frame без измеримых размеров** (нет `wp:extent`/cx/cy) overflow считается “вслепую” и фиксируется слишком слабо: в `layout_fix._estimate_overflow_ratio()` **игнорируется `ratio` из issue**, поэтому шрифт уменьшается “по дефолту 0.5pt”, даже если рост x2.
2. **VML-текстбоксы** (ABBYY/конвертеры) часто задают размеры в `style="width:...;height:...pt"` — текущий `layout_check` это не парсит, значит **capacity не считается** и риски/фиксы хуже.
3. В текстбоксах важен не только шрифт/spacing, но и **внутренние отступы (inset / bodyPr lIns/rIns/...)** + **lineRule=exact**. Сейчас эти штуки либо правятся только в `abbyy_profile=full`, либо не правятся точечно под overflow.

Ниже — патч, который:

* чинит `_estimate_overflow_ratio()` (использует `ratio`/`target_len/source_len` если нет `approx_capacity_chars`);
* добавляет парсинг VML `style width/height` в `layout_check` (значительно лучше детект overflow);
* делает точечный “анти-клиппинг” для текстбокса при overflow: **relax exact line spacing + включить normAutofit + ужать inset’ы**;
* делает агрессивный ABBYY-relax **только для page-anchored рамок НЕ на краях страницы** (верх/низ сохраняем);
* добавляет конфиг-параметр `layout_textbox_inset_cap_emu` (чтобы “расстояние между текстом” можно было регулировать; 25400 EMU ≈ 2pt).

---

```diff
diff --git a/src/docxru/layout_fix.py b/src/docxru/layout_fix.py
index 8c0c6a1..e7f3c2d 100644
--- a/src/docxru/layout_fix.py
+++ b/src/docxru/layout_fix.py
@@ -1,6 +1,7 @@
 from __future__ import annotations
 
+import re
 from docx.oxml import OxmlElement
 from docx.oxml.ns import qn
 from docx.shared import Pt
@@ -9,6 +10,9 @@ from docx.table import _Cell
 from .config import PipelineConfig
 from .models import Issue, Segment, Severity
 
+_TEXTBOX_INSET_DEFAULT_CAP_EMU = 25400  # ~2pt (1pt=12700 EMU)
+
 
 def reduce_font_size(paragraph, reduction_pt: float = 0.5, *, min_font_pt: float = 6.0) -> bool:
     changed = False
@@ -30,17 +34,27 @@ def reduce_font_size(paragraph, reduction_pt: float = 0.5, *, min_font_pt: float = 6.0) -> bool:
     if changed:
         return True
 
     # If explicit run sizes are missing, apply a small fallback size to text runs.
-    fallback = max(floor_pt, 10.0 - step)
     for run in paragraph.runs:
         if not (run.text or "").strip():
             continue
-        current_size = _resolve_run_size_pt(run, paragraph)
-        if current_size is not None and float(current_size) <= floor_pt + 1e-6:
-            continue
-        run.font.size = Pt(fallback)
+        current_size = _resolve_run_size_pt(run, paragraph)
+        if current_size is not None:
+            if float(current_size) <= floor_pt + 1e-6:
+                continue
+            fallback = max(floor_pt, float(current_size) - step)
+        else:
+            # Last-resort: assume ~10pt base if we can't resolve style size.
+            fallback = max(floor_pt, 10.0 - step)
+        run.font.size = Pt(float(fallback))
         changed = True
     return changed
 
@@ -102,6 +116,68 @@ def reduce_character_spacing(paragraph, twips: int = -10) -> bool:
     return changed
 
 
+def _local_name(node) -> str:
+    tag = str(getattr(node, "tag", "") or "")
+    if not tag:
+        return ""
+    if "}" in tag:
+        return tag.split("}", 1)[1]
+    if ":" in tag:
+        return tag.split(":", 1)[1]
+    return tag
+
+
+def _iter_textbox_body_pr_nodes(paragraph):
+    p_elm = getattr(paragraph, "_p", None)
+    if p_elm is None:
+        return
+    # Keep this tight to avoid scanning whole document tree.
+    container_hints = {"txbx", "textbox", "shape", "wsp", "drawing", "pict"}
+    for ancestor in [p_elm, *list(p_elm.iterancestors())]:
+        if _local_name(ancestor) not in container_hints:
+            continue
+        for node in ancestor.iter():
+            if _local_name(node) == "bodyPr":
+                yield node
+
+
+def _iter_vml_textbox_nodes(paragraph):
+    p_elm = getattr(paragraph, "_p", None)
+    if p_elm is None:
+        return
+    for ancestor in [p_elm, *list(p_elm.iterancestors())]:
+        if _local_name(ancestor) == "textbox":
+            yield ancestor
+
+
+def _set_body_pr_norm_autofit(body_pr) -> bool:
+    no_autofit_nodes = []
+    has_norm_autofit = False
+    for child in list(body_pr):
+        name = _local_name(child)
+        if name == "noAutofit":
+            no_autofit_nodes.append(child)
+        elif name == "normAutofit":
+            has_norm_autofit = True
+    if not no_autofit_nodes:
+        return False
+    for node in no_autofit_nodes:
+        body_pr.remove(node)
+    if not has_norm_autofit:
+        body_pr.append(OxmlElement("a:normAutofit"))
+    return True
+
+
+def _enable_textbox_norm_autofit(paragraph) -> bool:
+    changed = False
+    seen: set[int] = set()
+    for body_pr in _iter_textbox_body_pr_nodes(paragraph):
+        nid = id(body_pr)
+        if nid in seen:
+            continue
+        seen.add(nid)
+        changed = _set_body_pr_norm_autofit(body_pr) or changed
+    return changed
+
+
+def _tighten_textbox_insets(paragraph, *, cap_emu: int) -> bool:
+    changed = False
+    cap = max(0, int(cap_emu))
+    seen: set[int] = set()
+    for body_pr in _iter_textbox_body_pr_nodes(paragraph):
+        nid = id(body_pr)
+        if nid in seen:
+            continue
+        seen.add(nid)
+        for attr in ("lIns", "rIns", "tIns", "bIns"):
+            raw = body_pr.get(attr)
+            if raw is None:
+                continue
+            try:
+                val = int(str(raw))
+            except (TypeError, ValueError):
+                continue
+            if val <= cap:
+                continue
+            body_pr.set(attr, str(cap))
+            changed = True
+    # VML textboxes (if present): use inset="0,0,0,0" to reclaim padding.
+    for tb in _iter_vml_textbox_nodes(paragraph):
+        raw_inset = str(tb.get("inset", "") or "").strip()
+        if raw_inset == "0,0,0,0":
+            continue
+        # Only touch when we are already fixing overflow risk.
+        tb.set("inset", "0,0,0,0")
+        changed = True
+    return changed
+
+
 def _paragraph_cell(paragraph) -> _Cell | None:
     parent = getattr(paragraph, "_parent", None)
     if isinstance(parent, _Cell):
@@ -141,6 +217,25 @@ def _remove_paragraph_frame(paragraph) -> bool:
     p_pr.remove(frame_pr)
     return True
 
 
+def _relax_paragraph_exact_line_spacing(paragraph) -> bool:
+    p_elm = getattr(paragraph, "_p", None)
+    if p_elm is None:
+        return False
+    p_pr = getattr(p_elm, "pPr", None)
+    if p_pr is None:
+        return False
+    spacing = p_pr.find(qn("w:spacing"))
+    if spacing is None:
+        return False
+    line_rule_attr = qn("w:lineRule")
+    if str(spacing.get(line_rule_attr, "") or "").strip().lower() != "exact":
+        return False
+    spacing.set(line_rule_attr, "atLeast")
+    return True
+
+
 def _relax_table_row_exact_height(paragraph) -> bool:
     p_elm = getattr(paragraph, "_p", None)
     if p_elm is None:
@@ -236,6 +331,27 @@ def _estimate_overflow_ratio(issue: Issue | None) -> float:
     if issue is None:
         return 1.0
     details = issue.details or {}
-    try:
-        target_len = float(details.get("target_len", 0) or 0)
-        approx_capacity = float(details.get("approx_capacity_chars", 0) or 0)
-    except (TypeError, ValueError):
-        return 1.0
-    if approx_capacity <= 0.0 or target_len <= 0.0:
-        return 1.0
-    return max(1.0, target_len / approx_capacity)
+    # Preferred: explicit capacity-based ratio from layout checks.
+    try:
+        target_len = float(details.get("target_len", 0) or 0)
+        approx_capacity = float(details.get("approx_capacity_chars", 0) or 0)
+        if approx_capacity > 0.0 and target_len > 0.0:
+            return max(1.0, target_len / approx_capacity)
+    except (TypeError, ValueError):
+        pass
+    # Fallback: ratio from length_ratio_high / textbox fallback checks.
+    try:
+        ratio = float(details.get("ratio", 0) or 0)
+        if ratio > 0.0:
+            return max(1.0, ratio)
+    except (TypeError, ValueError):
+        pass
+    # Last resort: target/source lengths if present.
+    try:
+        source_len = float(details.get("source_len", 0) or 0)
+        target_len = float(details.get("target_len", 0) or 0)
+        if source_len > 0.0 and target_len > 0.0:
+            return max(1.0, target_len / source_len)
+    except (TypeError, ValueError):
+        pass
+    return 1.0
 
 
 def _paragraph_average_font_pt(paragraph) -> float | None:
@@ -311,6 +427,7 @@ def _fix_table_overflow(
     overflow_ratio = _estimate_overflow_ratio(issue)
     cell = _paragraph_cell(seg.paragraph_ref)
     spacing_factor = float(cfg.layout_spacing_factor)
@@ -326,6 +443,7 @@ def _fix_table_overflow(
         if len(cell.paragraphs) > 1:
             spacing_factor *= 0.85
         changed = reduce_cell_spacing(cell, factor=spacing_factor) or changed
         for paragraph in cell.paragraphs:
+            changed = _relax_paragraph_exact_line_spacing(paragraph) or changed
             if pass_number >= 3 and overflow_ratio >= 1.8:
                 char_spacing_twips = -12 if pass_number == 3 else -15
                 changed = reduce_character_spacing(paragraph, twips=char_spacing_twips) or changed
@@ -335,6 +453,7 @@ def _fix_table_overflow(
             if pass_number >= 3 and overflow_ratio >= 1.8:
                 changed = set_single_line_spacing(paragraph) or changed
     else:
+        changed = _relax_paragraph_exact_line_spacing(seg.paragraph_ref) or changed
         changed = reduce_paragraph_spacing(seg.paragraph_ref, factor=spacing_factor) or changed
         if pass_number >= 3 and overflow_ratio >= 1.8:
             char_spacing_twips = -12 if pass_number == 3 else -15
@@ -383,10 +502,15 @@ def _fix_textbox_overflow(
 
     changed = False
     overflow_ratio = _estimate_overflow_ratio(issue)
+    # Reclaim space before shrinking fonts too much.
+    inset_cap = int(getattr(cfg, "layout_textbox_inset_cap_emu", _TEXTBOX_INSET_DEFAULT_CAP_EMU))
+    changed = _relax_paragraph_exact_line_spacing(seg.paragraph_ref) or changed
+    changed = _enable_textbox_norm_autofit(seg.paragraph_ref) or changed
+    changed = _tighten_textbox_insets(seg.paragraph_ref, cap_emu=inset_cap) or changed
     textbox_spacing_factor = min(0.9, max(0.4, float(cfg.layout_spacing_factor)))
     if pass_number >= 2:
         textbox_spacing_factor *= 0.85
     changed = reduce_paragraph_spacing(seg.paragraph_ref, factor=textbox_spacing_factor) or changed
@@ -430,6 +554,7 @@ def _fix_frame_overflow(
 
     changed = False
     overflow_ratio = _estimate_overflow_ratio(issue)
+    changed = _relax_paragraph_exact_line_spacing(seg.paragraph_ref) or changed
     if (overflow_ratio >= 1.35 and bool(seg.context.get("in_table"))) or pass_number >= 3:
         changed = _remove_paragraph_frame(seg.paragraph_ref) or changed
     changed = _relax_paragraph_frame_height_rule(seg.paragraph_ref) or changed
@@ -476,6 +601,7 @@ def _fix_generic_overflow(seg: Segment, cfg: PipelineConfig, *, pass_number: int = 1) -> bool:
     if seg.paragraph_ref is None:
         return False
 
     changed = False
+    changed = _relax_paragraph_exact_line_spacing(seg.paragraph_ref) or changed
     spacing_factor = float(cfg.layout_spacing_factor)
     if pass_number >= 2:
         spacing_factor *= 0.9
diff --git a/src/docxru/layout_check.py b/src/docxru/layout_check.py
index 2b92a0b..df0d0d2 100644
--- a/src/docxru/layout_check.py
+++ b/src/docxru/layout_check.py
@@ -14,6 +14,7 @@ from .token_shield import strip_bracket_tokens
 
 _SPACE_RE = re.compile(r"\s+")
 _EMU_PER_TWIP = 635
+_VML_DIM_RE = re.compile(r"(?:^|;)\s*(width|height)\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*(pt|in|cm|mm|px)\b", re.IGNORECASE)
 _APPROX_CHAR_WIDTH_TWIPS = 120
 _APPROX_CHAR_HEIGHT_TWIPS = 220
 _FONT_WIDTH_FACTORS = {
@@ -176,6 +177,48 @@ def _textbox_extent_twips(seg: Segment) -> tuple[int | None, int | None]:
     except Exception:
         return None, None
 
+    def _local_name(node) -> str:
+        tag = str(getattr(node, "tag", "") or "")
+        if not tag:
+            return ""
+        if "}" in tag:
+            return tag.split("}", 1)[1]
+        if ":" in tag:
+            return tag.split(":", 1)[1]
+        return tag
+
+    def _parse_len_to_twips(value: float, unit: str) -> int | None:
+        if value <= 0:
+            return None
+        u = (unit or "").strip().lower()
+        if u == "pt":
+            return int(value * 20.0)
+        if u == "in":
+            return int(value * 1440.0)
+        if u == "cm":
+            return int(value * (1440.0 / 2.54))
+        if u == "mm":
+            return int(value * (1440.0 / 25.4))
+        if u == "px":
+            # assume 96 dpi => 1px = 0.75pt = 15 twips
+            return int(value * 15.0)
+        return None
+
+    def _parse_vml_style_extent(style: str) -> tuple[int | None, int | None]:
+        if not style:
+            return None, None
+        width_twips = None
+        height_twips = None
+        for m in _VML_DIM_RE.finditer(style):
+            key = (m.group(1) or "").strip().lower()
+            try:
+                val = float(m.group(2))
+            except (TypeError, ValueError):
+                continue
+            unit = (m.group(3) or "").strip().lower()
+            tw = _parse_len_to_twips(val, unit)
+            if tw is None or tw <= 0:
+                continue
+            if key == "width":
+                width_twips = tw
+            elif key == "height":
+                height_twips = tw
+        return width_twips, height_twips
+
     for ancestor in [p_elm, *list(p_elm.iterancestors())]:
         for node in ancestor.iter():
             tag = str(getattr(node, "tag", "")).lower()
@@ -193,6 +236,21 @@ def _textbox_extent_twips(seg: Segment) -> tuple[int | None, int | None]:
             if width_twips > 0 and height_twips > 0:
                 return width_twips, height_twips
+
+        # VML/Word shapes often encode width/height in style="width:..pt;height:..pt".
+        # Look for the nearest shape ancestor with a style attribute.
+        if _local_name(ancestor) == "shape":
+            style = str(ancestor.get("style") or "").strip()
+            w_tw, h_tw = _parse_vml_style_extent(style)
+            if w_tw is not None or h_tw is not None:
+                return w_tw, h_tw
     return None, None
 
 
@@ -280,10 +338,16 @@ def check_textbox_overflow(doc, segments: Iterable[Segment]) -> list[Issue]:
         overflow = False
         if approx_capacity is not None:
             overflow = len(target_text) > int(approx_capacity * 1.1)
         else:
-            overflow = ratio > 1.6 and (len(target_text) - len(source_text)) >= 20
+            # Fallback when container size is unknown: textbox layout is usually tight,
+            # so use a more sensitive heuristic than generic paragraph expansion.
+            delta = len(target_text) - len(source_text)
+            overflow = (len(source_text) >= 10 and ratio > 1.3 and delta >= 8) or (ratio > 1.6 and delta >= 20)
         if not overflow:
             continue
diff --git a/src/docxru/oxml_table_fix.py b/src/docxru/oxml_table_fix.py
index 2a6dcb0..7c6a0fa 100644
--- a/src/docxru/oxml_table_fix.py
+++ b/src/docxru/oxml_table_fix.py
@@ -167,6 +167,33 @@ def set_textbox_autofit(document) -> int:
     return updated
 
 
+def normalize_textbox_insets(document, *, max_inset_emu: int = 25400) -> int:
+    """Cap excessive DrawingML textbox insets (a:bodyPr lIns/rIns/tIns/bIns) to reclaim space."""
+    cap = max(0, int(max_inset_emu))
+    updated = 0
+    seen_node_ids: set[int] = set()
+    for txbx_content in _iter_non_empty_textbox_contents(document):
+        for body_pr in _iter_related_body_pr_nodes(txbx_content):
+            node_id = id(body_pr)
+            if node_id in seen_node_ids:
+                continue
+            seen_node_ids.add(node_id)
+            changed = False
+            for attr in ("lIns", "rIns", "tIns", "bIns"):
+                raw = body_pr.get(attr)
+                if raw is None:
+                    continue
+                try:
+                    val = int(str(raw))
+                except (TypeError, ValueError):
+                    continue
+                if val <= cap:
+                    continue
+                body_pr.set(attr, str(cap))
+                changed = True
+            if changed:
+                updated += 1
+    return updated
+
+
 def normalize_table_cell_margins(document, *, max_margin_twips: int = 108) -> int:
     """Cap excessive table-cell margins in w:tcMar to reduce avoidable text loss."""
     normalized = 0
@@ -208,6 +235,7 @@ def normalize_abbyy_oxml(document, *, profile: str) -> dict[str, int]:
         "line_spacing_exact_relaxed": 0,
         "textbox_autofit_updated": 0,
+        "textbox_insets_normalized": 0,
         "table_cell_margins_normalized": 0,
     }
     if mode == "off":
@@ -216,16 +244,16 @@ def normalize_abbyy_oxml(document, *, profile: str) -> dict[str, int]:
 
     stats["tr_height_exact_removed"] = remove_exact_tr_height(document)
     if mode in {"aggressive", "full"}:
         stats["frame_pr_exact_relaxed"] = relax_frame_pr_exact_height(
             document,
             preserve_page_edge_frames=True,
-            preserve_page_anchored_frames=True,
+            preserve_page_anchored_frames=False,
         )
         stats["line_spacing_exact_relaxed"] = relax_exact_line_spacing(
             document,
             preserve_page_edge_frames=True,
-            preserve_page_anchored_frames=True,
+            preserve_page_anchored_frames=False,
         )
     if mode == "full":
         stats["textbox_autofit_updated"] = set_textbox_autofit(document)
+        stats["textbox_insets_normalized"] = normalize_textbox_insets(document)
         stats["table_cell_margins_normalized"] = normalize_table_cell_margins(document)
     return stats
diff --git a/src/docxru/pipeline.py b/src/docxru/pipeline.py
index 6f8237d..b8f8e3a 100644
--- a/src/docxru/pipeline.py
+++ b/src/docxru/pipeline.py
@@ -657,10 +657,11 @@ def _apply_abbyy_and_layout_passes(doc: Document, segments: list[Segment], cfg: PipelineConfig, logger) -> None:
             logger.info(
                 "ABBYY OXML normalization (%s): trHeight_exact_removed=%d; framePr_removed=%d; "
                 "framePr_exact_relaxed=%d; lineSpacing_exact_relaxed=%d; textbox_autofit_updated=%d; "
-                "tableCellMargins_normalized=%d",
+                "textbox_insets_normalized=%d; tableCellMargins_normalized=%d",
                 cfg.abbyy_profile,
                 int(oxml_stats.get("tr_height_exact_removed", 0)),
                 int(oxml_stats.get("frame_pr_removed", 0)),
                 int(oxml_stats.get("frame_pr_exact_relaxed", 0)),
                 int(oxml_stats.get("line_spacing_exact_relaxed", 0)),
                 int(oxml_stats.get("textbox_autofit_updated", 0)),
+                int(oxml_stats.get("textbox_insets_normalized", 0)),
                 int(oxml_stats.get("table_cell_margins_normalized", 0)),
             )
diff --git a/src/docxru/config.py b/src/docxru/config.py
index 0b8a4d2..7bbd7f1 100644
--- a/src/docxru/config.py
+++ b/src/docxru/config.py
@@ -140,6 +140,10 @@ class PipelineConfig:
     layout_auto_fix_passes: int = 1
     layout_font_reduction_pt: float = 0.5
     layout_spacing_factor: float = 0.8
+    # Cap textbox internal padding (DrawingML a:bodyPr insets) in EMU.
+    # 0 -> максимально ужать (до 0), 25400 (~2pt) обычно безопасно.
+    layout_textbox_inset_cap_emu: int = 25400
     # Readability floor for layout auto-fix shrink operations.
     layout_min_font_pt: float = 6.0
     # Optional unconditional post-writeback font shrink.
@@ -156,6 +160,7 @@ _FORMATTING_PRESET_FIELDS: tuple[str, ...] = (
     "layout_auto_fix",
     "layout_auto_fix_passes",
     "font_shrink_body_pt",
     "font_shrink_table_pt",
+    "layout_textbox_inset_cap_emu",
     "layout_min_font_pt",
     "font_shrink_min_font_pt",
     "mode",
     "com_expand_overflowing_shapes",
 )
@@ -171,6 +176,7 @@ _FORMATTING_PRESET_DEFAULTS: dict[str, dict[str, Any]] = {
         "layout_auto_fix_passes": 1,
         "font_shrink_body_pt": 0.0,
         "font_shrink_table_pt": 0.0,
+        "layout_textbox_inset_cap_emu": 25400,
         "layout_min_font_pt": 6.0,
         "font_shrink_min_font_pt": 6.0,
         "mode": "reflow",
         "com_expand_overflowing_shapes": False,
@@ -185,6 +191,7 @@ _FORMATTING_PRESET_DEFAULTS: dict[str, dict[str, Any]] = {
         "layout_auto_fix_passes": 1,
         "font_shrink_body_pt": 0.0,
         "font_shrink_table_pt": 0.0,
+        "layout_textbox_inset_cap_emu": 25400,
         "layout_min_font_pt": 6.0,
         "font_shrink_min_font_pt": 6.0,
         "mode": "reflow",
         "com_expand_overflowing_shapes": False,
@@ -193,14 +200,15 @@ _FORMATTING_PRESET_DEFAULTS: dict[str, dict[str, Any]] = {
     "abbyy_standard": {
         "translate_enable_formatting_fixes": True,
-        "abbyy_profile": "aggressive",
+        "abbyy_profile": "full",
         "layout_check": True,
         "layout_auto_fix": True,
         "layout_auto_fix_passes": 2,
         "font_shrink_body_pt": 0.0,
         "font_shrink_table_pt": 0.5,
-        "layout_min_font_pt": 9.5,
-        "font_shrink_min_font_pt": 9.5,
+        "layout_textbox_inset_cap_emu": 25400,
+        "layout_min_font_pt": 9.0,
+        "font_shrink_min_font_pt": 9.0,
         "mode": "reflow",
         "com_expand_overflowing_shapes": False,
     },
@@ -215,6 +223,7 @@ _FORMATTING_PRESET_DEFAULTS: dict[str, dict[str, Any]] = {
         "layout_auto_fix_passes": 3,
         "font_shrink_body_pt": 0.5,
         "font_shrink_table_pt": 1.0,
+        "layout_textbox_inset_cap_emu": 25400,
         "layout_min_font_pt": 8.0,
         "font_shrink_min_font_pt": 8.0,
         "mode": "com",
         "com_expand_overflowing_shapes": True,
@@ -410,6 +419,7 @@ def load_config(path: str | Path) -> PipelineConfig:
     layout_auto_fix = bool(data.get("layout_auto_fix", preset_defaults["layout_auto_fix"]))
     layout_auto_fix_passes = max(1, int(data.get("layout_auto_fix_passes", preset_defaults["layout_auto_fix_passes"])))
     layout_font_reduction_pt = float(data.get("layout_font_reduction_pt", 0.5))
     layout_spacing_factor = float(data.get("layout_spacing_factor", 0.8))
+    layout_textbox_inset_cap_emu = max(0, int(data.get("layout_textbox_inset_cap_emu", preset_defaults.get("layout_textbox_inset_cap_emu", 25400))))
     layout_min_font_pt = max(6.0, float(data.get("layout_min_font_pt", preset_defaults["layout_min_font_pt"])))
     font_shrink_body_pt = max(0.0, float(data.get("font_shrink_body_pt", preset_defaults["font_shrink_body_pt"])))
     font_shrink_table_pt = max(0.0, float(data.get("font_shrink_table_pt", preset_defaults["font_shrink_table_pt"])))
@@ -470,6 +480,7 @@ def load_config(path: str | Path) -> PipelineConfig:
         layout_check=layout_check,
         layout_expansion_warn_ratio=layout_expansion_warn_ratio,
         layout_auto_fix=layout_auto_fix,
         layout_auto_fix_passes=layout_auto_fix_passes,
         layout_font_reduction_pt=layout_font_reduction_pt,
         layout_spacing_factor=layout_spacing_factor,
+        layout_textbox_inset_cap_emu=layout_textbox_inset_cap_emu,
         layout_min_font_pt=layout_min_font_pt,
         font_shrink_body_pt=font_shrink_body_pt,
         font_shrink_table_pt=font_shrink_table_pt,
         font_shrink_min_font_pt=font_shrink_min_font_pt,
         pattern_set=pattern_set,
     )
diff --git a/tests/test_oxml_table_fix.py b/tests/test_oxml_table_fix.py
index 5b6d8b8..d1e9b29 100644
--- a/tests/test_oxml_table_fix.py
+++ b/tests/test_oxml_table_fix.py
@@ -6,7 +6,7 @@ from docx.oxml import OxmlElement
 from docx.oxml.ns import qn
 
-from docxru.oxml_table_fix import normalize_abbyy_oxml, normalize_table_cell_margins, set_textbox_autofit
+from docxru.oxml_table_fix import normalize_abbyy_oxml, normalize_table_cell_margins, normalize_textbox_insets, set_textbox_autofit
 
 
 def _append_exact_tr_height(doc: Document):
@@ -131,6 +131,22 @@ def test_normalize_abbyy_oxml_aggressive_preserves_page_anchored_framed_lines():
     bottom_frame, bottom_spacing = _append_framed_paragraph_with_spacing(doc, y_twips=15100)
 
     stats = normalize_abbyy_oxml(doc, profile="aggressive")
 
-    assert stats["frame_pr_exact_relaxed"] == 0
-    assert stats["line_spacing_exact_relaxed"] == 0
+    # Preserve edge-anchored (header/footer-like) frames, but relax page-anchored body frames.
+    assert stats["frame_pr_exact_relaxed"] == 1
+    assert stats["line_spacing_exact_relaxed"] == 1
     assert top_frame.get(qn("w:hRule")) == "exact"
     assert top_spacing.get(qn("w:lineRule")) == "exact"
-    assert body_frame.get(qn("w:hRule")) == "exact"
-    assert body_spacing.get(qn("w:lineRule")) == "exact"
+    assert body_frame.get(qn("w:hRule")) == "atLeast"
+    assert body_spacing.get(qn("w:lineRule")) == "atLeast"
     assert bottom_frame.get(qn("w:hRule")) == "exact"
     assert bottom_spacing.get(qn("w:lineRule")) == "exact"
 
@@ -155,6 +171,7 @@ def test_normalize_abbyy_oxml_full_applies_textbox_autofit():
     assert stats["frame_pr_removed"] == 0
     assert stats["frame_pr_exact_relaxed"] == 1
     assert stats["line_spacing_exact_relaxed"] == 1
     assert stats["textbox_autofit_updated"] == 1
+    assert stats["textbox_insets_normalized"] == 0
     assert stats["table_cell_margins_normalized"] == 0
     assert tr_pr.find(qn("w:trHeight")) is None
     frame_pr = p_pr.find(qn("w:framePr"))
@@ -169,6 +186,21 @@ def test_normalize_abbyy_oxml_full_applies_textbox_autofit():
     assert _has_child(body_pr, "noAutofit") is False
     assert _has_child(body_pr, "normAutofit") is True
 
 
+def test_normalize_textbox_insets_caps_excessive_bodypr_insets():
+    doc = Document()
+    body_pr, _ = _append_textbox(doc, text="Textbox text")
+    body_pr.set("lIns", "91440")  # 0.1in
+    body_pr.set("rIns", "91440")
+    body_pr.set("tIns", "91440")
+    body_pr.set("bIns", "91440")
+
+    changed = normalize_textbox_insets(doc, max_inset_emu=25400)
+
+    assert changed == 1
+    assert body_pr.get("lIns") == "25400"
+    assert body_pr.get("bIns") == "25400"
+
diff --git a/tests/test_layout_check.py b/tests/test_layout_check.py
index 71e9ad2..cbd71d1 100644
--- a/tests/test_layout_check.py
+++ b/tests/test_layout_check.py
@@ -1,6 +1,7 @@
 from __future__ import annotations
 
 from docx import Document
+from docx.text.paragraph import Paragraph as DocxParagraph
 from docx.oxml import OxmlElement
 from docx.oxml.ns import qn
 
@@ -45,6 +46,29 @@ def _attach_extent(paragraph, *, width_twips: int, height_twips: int) -> None:
     run._r.append(extent)
 
 
+def _make_vml_textbox_paragraph(doc: Document, *, style: str, text: str):
+    host = doc.add_paragraph("Host")
+    run = host.add_run("")
+    shape = OxmlElement("w:shape")
+    shape.set("style", style)
+    txbx_content = OxmlElement("w:txbxContent")
+    p = OxmlElement("w:p")
+    r = OxmlElement("w:r")
+    t = OxmlElement("w:t")
+    t.text = text
+    r.append(t)
+    p.append(r)
+    txbx_content.append(p)
+    shape.append(txbx_content)
+    run._r.append(shape)
+    return DocxParagraph(p, doc)
+
+
 def _attach_frame(paragraph, *, width_twips: int, height_twips: int) -> None:
     p_pr = paragraph._p.get_or_add_pPr()
     frame_pr = OxmlElement("w:framePr")
     frame_pr.set(qn("w:w"), str(int(width_twips)))
@@ -82,6 +106,23 @@ def test_check_textbox_overflow_uses_extent_dimensions():
     assert details["approx_capacity_chars"] < len(seg.target_tagged or "")
 
 
+def test_check_textbox_overflow_parses_vml_style_dimensions():
+    doc = Document()
+    tb_p = _make_vml_textbox_paragraph(doc, style="width:100pt;height:10pt", text="Textbox")
+    seg = _make_segment(
+        seg_id="2b",
+        source="Bolt",
+        target="Very long translated text that should overflow vml style sized textbox",
+        paragraph_ref=tb_p,
+        in_textbox=True,
+    )
+    issues = check_textbox_overflow(doc, [seg])
+    assert len(issues) == 1
+    assert issues[0].details["width_twips"] == 2000
+    assert issues[0].details["height_twips"] == 200
+
+
 def test_check_textbox_overflow_fallback_by_ratio():
     seg = _make_segment(
         seg_id="3",
         source="Bearing",
```

---

### Как это использовать (минимально)

1. Если документ “ABBYY-подобный” (много рамок/текстбоксов):

   * ставь `formatting_preset: abbyy_standard` (после патча он уже `abbyy_profile: full`, min font ~9 и inset cap включён).
2. Если всё равно где-то режет: увеличь “агрессивность” постфикса:

   ```yaml
   translate_enable_formatting_fixes: true
   layout_check: true
   layout_auto_fix: true
   layout_auto_fix_passes: 3
   layout_min_font_pt: 9.0
   layout_textbox_inset_cap_emu: 0        # если хочешь максимально ужать внутренние поля
   ```
3. На Windows можно добить совсем тяжёлые места COM-режимом:

   ```yaml
   mode: com
   com_textbox_min_font_pt: 9.0
   com_expand_overflowing_shapes: true
   com_textbox_max_height_growth: 1.4
   ```

---

### Next steps

* Применить патч, прогнать `pytest`.
* Перевести проблемный DOCX с `abbyy_standard`, посмотреть `format_report.html` (там будут сегменты с “layout_*_overflow_risk”).
* Если остаются клипы — включить `mode: com` + `com_expand_overflowing_shapes: true`.

ИМХО: основной реальный выигрыш даст связка **(1) нормальный overflow_ratio → адаптивный shrink** + **(2) VML width/height парсинг** + **(3) inset cap/relax exact lineRule**.
