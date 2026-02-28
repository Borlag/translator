from __future__ import annotations

import io
import json
import math
import os
import re
from collections import Counter
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from docx.document import Document as DocxDocument
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.opc.part import Part
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
WPS_NS = "http://schemas.microsoft.com/office/word/2010/wordprocessingShape"

NS = {"w": W_NS, "r": R_NS, "wp": WP_NS, "wps": WPS_NS}

W_P = qn("w:p")
W_TBL = qn("w:tbl")
W_PPR = qn("w:pPr")
W_SECTPR = qn("w:sectPr")
W_TYPE = qn("w:type")
W_PGSZ = qn("w:pgSz")
W_PGMAR = qn("w:pgMar")
W_PSTYLE = qn("w:pStyle")
W_RSTYLE = qn("w:rStyle")
W_TBLSTYLE = qn("w:tblStyle")
W_STYLE = qn("w:style")
W_STYLE_ID_ATTR = qn("w:styleId")
W_DEFAULT_ATTR = qn("w:default")
W_BASEDON = qn("w:basedOn")
W_NEXT = qn("w:next")
W_LINK = qn("w:link")
W_VAL_ATTR = qn("w:val")

R_ID_ATTR = qn("r:id")
R_EMBED_ATTR = qn("r:embed")
R_LINK_ATTR = qn("r:link")

_TOKEN_SPLIT_RE = re.compile(r"[\W_]+", flags=re.UNICODE)
_ALIGN_TOKEN_SPLIT_RE = re.compile(r"[^0-9A-Za-z\u0400-\u04FF]+", flags=re.UNICODE)
_PARTNAME_NUMBER_RE = re.compile(r"\d+(?=\.[^.\/]+$)")
_ALIGN_STOP_TOKENS = {
    "safran",
    "landing",
    "systems",
    "uk",
    "ltd",
    "код",
    "cage",
    "k0654",
    "руководство",
    "техническому",
    "обслуживанию",
    "компонентов",
    "стойка",
    "основного",
    "шасси",
    "mar",
    "2025",
    "18",
}


@dataclass
class SectionRange:
    section_index: int
    start_body_index: int
    end_body_index: int
    blocks: list[Any]
    sect_pr: Any
    sect_pr_owner: str


@dataclass
class PageMapping:
    mapping: dict[int, list[int]]
    scores: dict[int, float]
    base_sections: int
    overlay_pages: int
    mode: str = "unknown"

    def to_json_dict(
        self,
        *,
        base_path: Path | None = None,
        overlay_path: Path | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "base_sections": int(self.base_sections),
            "overlay_pages": int(self.overlay_pages),
            "mode": str(self.mode),
            "mapping": {str(k): v for k, v in sorted(self.mapping.items())},
            "scores": {str(k): float(v) for k, v in sorted(self.scores.items())},
        }
        if base_path is not None:
            payload["base_path"] = str(base_path)
        if overlay_path is not None:
            payload["overlay_path"] = str(overlay_path)
        return payload


def parse_pages_spec(spec: str) -> list[int]:
    if not spec.strip():
        return []

    pages: set[int] = set()
    for chunk in spec.split(","):
        part = chunk.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left.strip())
            end = int(right.strip())
            if start <= 0 or end <= 0:
                raise ValueError(f"Page numbers must be >= 1: {part}")
            if start > end:
                raise ValueError(f"Invalid page range: {part}")
            pages.update(range(start, end + 1))
        else:
            value = int(part)
            if value <= 0:
                raise ValueError(f"Page numbers must be >= 1: {part}")
            pages.add(value)
    return sorted(pages)


def get_section_ranges(doc: DocxDocument) -> list[SectionRange]:
    body = doc._element.body
    body_children = list(body.iterchildren())

    sections: list[SectionRange] = []
    current_blocks: list[Any] = []
    current_start_idx = 0

    for child_idx, child in enumerate(body_children):
        if child.tag in {W_P, W_TBL}:
            current_blocks.append(child)

        if child.tag != W_P:
            continue

        sect_pr = _paragraph_sect_pr(child)
        if sect_pr is None:
            continue

        sections.append(
            SectionRange(
                section_index=len(sections) + 1,
                start_body_index=current_start_idx,
                end_body_index=child_idx + 1,
                blocks=list(current_blocks),
                sect_pr=sect_pr,
                sect_pr_owner="paragraph",
            )
        )
        current_blocks = []
        current_start_idx = child_idx + 1

    body_sect_pr = body.find("./w:sectPr", namespaces=NS)
    if body_sect_pr is not None:
        sections.append(
            SectionRange(
                section_index=len(sections) + 1,
                start_body_index=current_start_idx,
                end_body_index=len(body_children),
                blocks=list(current_blocks),
                sect_pr=body_sect_pr,
                sect_pr_owner="body",
            )
        )
    elif current_blocks:
        # Last-resort fallback for malformed files without body-level sectPr.
        sections.append(
            SectionRange(
                section_index=len(sections) + 1,
                start_body_index=current_start_idx,
                end_body_index=len(body_children),
                blocks=list(current_blocks),
                sect_pr=_clone_minimal_sect_pr(),
                sect_pr_owner="paragraph",
            )
        )

    return sections


def build_page_mapping(
    base_doc: DocxDocument,
    overlay_doc: DocxDocument,
    *,
    mode: str = "dp_one_to_one",
) -> PageMapping:
    normalized_mode = mode.strip().lower()
    if normalized_mode == "greedy_group":
        return _build_page_mapping_greedy_group(base_doc, overlay_doc)
    if normalized_mode == "dp_one_to_one":
        return _build_page_mapping_dp_one_to_one(base_doc, overlay_doc)
    raise ValueError(f"Unknown mapping mode: {mode}")


def _build_page_mapping_greedy_group(base_doc: DocxDocument, overlay_doc: DocxDocument) -> PageMapping:
    base_sections = get_section_ranges(base_doc)
    overlay_sections = get_section_ranges(overlay_doc)

    base_counters = [_token_counter(_extract_base_section_text(section)) for section in base_sections]
    overlay_counters = [_token_counter(_extract_overlay_section_text(section)) for section in overlay_sections]

    mapping: dict[int, list[int]] = {}
    scores: dict[int, float] = {}

    base_pos = 0
    base_count = len(base_counters)
    overlay_count = len(overlay_counters)

    for page_idx, overlay_counter in enumerate(overlay_counters):
        page_no = page_idx + 1
        remaining_overlay = overlay_count - page_idx - 1
        remaining_base = base_count - base_pos

        if remaining_base <= 0:
            mapping[page_no] = []
            scores[page_no] = 0.0
            continue

        if remaining_base <= remaining_overlay:
            mapping[page_no] = []
            scores[page_no] = 0.0
            continue

        if page_idx == overlay_count - 1:
            take = remaining_base
            group_counter = _combine_counters(base_counters, base_pos, take)
            score = _token_overlap_score(group_counter, overlay_counter)
        else:
            max_take = max(1, remaining_base - remaining_overlay)
            avg_take = remaining_base / float(remaining_overlay + 1)

            if not overlay_counter:
                take = int(round(avg_take))
                take = max(1, min(max_take, take))
                group_counter = _combine_counters(base_counters, base_pos, take)
                score = _token_overlap_score(group_counter, overlay_counter)
            else:
                running = Counter[str]()
                best_take = 1
                best_adj_score = -1.0
                best_raw_score = 0.0
                for take_candidate in range(1, max_take + 1):
                    running.update(base_counters[base_pos + take_candidate - 1])
                    raw = _token_overlap_score(running, overlay_counter)
                    adjusted = raw - (abs(take_candidate - avg_take) * 0.002)
                    if adjusted > best_adj_score:
                        best_take = take_candidate
                        best_adj_score = adjusted
                        best_raw_score = raw

                take = best_take
                score = best_raw_score

        mapping[page_no] = list(range(base_pos + 1, base_pos + take + 1))
        scores[page_no] = round(float(score), 4)
        base_pos += take

    if overlay_count > 0 and base_pos < base_count:
        last_page = overlay_count
        leftover = list(range(base_pos + 1, base_count + 1))
        mapping.setdefault(last_page, [])
        mapping[last_page].extend(leftover)
        group_counter = _combine_counters(
            base_counters,
            start=(mapping[last_page][0] - 1 if mapping[last_page] else 0),
            take=len(mapping[last_page]),
        )
        scores[last_page] = round(float(_token_overlap_score(group_counter, overlay_counters[-1])), 4)

    return PageMapping(
        mapping=mapping,
        scores=scores,
        base_sections=len(base_sections),
        overlay_pages=len(overlay_sections),
        mode="greedy_group",
    )


def _build_page_mapping_dp_one_to_one(base_doc: DocxDocument, overlay_doc: DocxDocument) -> PageMapping:
    base_sections = get_section_ranges(base_doc)
    overlay_sections = get_section_ranges(overlay_doc)

    base_count = len(base_sections)
    overlay_count = len(overlay_sections)
    if overlay_count == 0:
        return PageMapping(
            mapping={},
            scores={},
            base_sections=base_count,
            overlay_pages=0,
            mode="dp_one_to_one",
        )
    if base_count == 0:
        return PageMapping(
            mapping={page + 1: [] for page in range(overlay_count)},
            scores={page + 1: 0.0 for page in range(overlay_count)},
            base_sections=0,
            overlay_pages=overlay_count,
            mode="dp_one_to_one",
        )
    if overlay_count > base_count:
        raise ValueError(
            "dp_one_to_one mapping requires base sections >= overlay pages "
            f"(base={base_count}, overlay={overlay_count})"
        )

    base_texts = [_extract_section_text_all(section) for section in base_sections]
    overlay_texts = [_extract_section_text_all(section) for section in overlay_sections]

    base_tokens = [_tokenize_for_alignment(text) for text in base_texts]
    overlay_tokens = [_tokenize_for_alignment(text) for text in overlay_texts]

    all_docs = base_tokens + overlay_tokens
    doc_count = len(all_docs)
    doc_freq = Counter[str]()
    for tokens in all_docs:
        for token in set(tokens):
            doc_freq[token] += 1

    informative_tokens: set[str] = set()
    for token, freq in doc_freq.items():
        if freq <= 2 or (freq <= (doc_count * 0.25) and len(token) >= 2):
            informative_tokens.add(token)

    base_vectors = [_alignment_vector(tokens, informative_tokens) for tokens in base_tokens]
    overlay_vectors = [_alignment_vector(tokens, informative_tokens) for tokens in overlay_tokens]

    idf: dict[str, float] = {}
    for token in informative_tokens:
        idf[token] = math.log((doc_count + 1) / (doc_freq[token] + 1)) + 1.0

    base_norms = [_weighted_vector_norm(vector, idf) for vector in base_vectors]
    overlay_norms = [_weighted_vector_norm(vector, idf) for vector in overlay_vectors]

    sim_matrix = [[0.0 for _ in range(base_count + 1)] for _ in range(overlay_count + 1)]
    for page_idx in range(1, overlay_count + 1):
        overlay_vector = overlay_vectors[page_idx - 1]
        overlay_norm = overlay_norms[page_idx - 1]
        for section_idx in range(1, base_count + 1):
            sim_matrix[page_idx][section_idx] = _weighted_cosine_similarity(
                overlay_vector,
                base_vectors[section_idx - 1],
                overlay_norm,
                base_norms[section_idx - 1],
                idf,
            )

    ratio = base_count / float(overlay_count)
    gap_penalty = 0.01
    position_penalty = 0.0015
    negative_inf = -1e18

    dp = [[negative_inf for _ in range(base_count + 1)] for _ in range(overlay_count + 1)]
    prev = [[-1 for _ in range(base_count + 1)] for _ in range(overlay_count + 1)]

    first_max = base_count - (overlay_count - 1)
    for section_idx in range(1, first_max + 1):
        expected = ratio
        score = sim_matrix[1][section_idx] - (abs(section_idx - expected) * position_penalty)
        dp[1][section_idx] = score

    for page_idx in range(2, overlay_count + 1):
        section_min = page_idx
        section_max = base_count - (overlay_count - page_idx)
        best_transition = negative_inf
        best_prev_idx = -1
        for section_idx in range(section_min, section_max + 1):
            candidate_prev = section_idx - 1
            candidate_score = dp[page_idx - 1][candidate_prev] + (gap_penalty * candidate_prev)
            if candidate_score > best_transition:
                best_transition = candidate_score
                best_prev_idx = candidate_prev

            if best_prev_idx < 1:
                continue

            expected = ratio * page_idx
            current = (
                best_transition
                + sim_matrix[page_idx][section_idx]
                - (gap_penalty * (section_idx - 1))
                - (abs(section_idx - expected) * position_penalty)
            )
            dp[page_idx][section_idx] = current
            prev[page_idx][section_idx] = best_prev_idx

    last_page = overlay_count
    last_section_min = overlay_count
    last_section_max = base_count
    best_last_section = max(range(last_section_min, last_section_max + 1), key=lambda idx: dp[last_page][idx])

    assignment = [0 for _ in range(overlay_count + 1)]
    page_idx = overlay_count
    section_idx = best_last_section
    while page_idx >= 1 and section_idx >= 1:
        assignment[page_idx] = section_idx
        section_idx = prev[page_idx][section_idx]
        page_idx -= 1

    if any(value <= 0 for value in assignment[1:]):
        raise RuntimeError("Failed to build one-to-one page mapping")

    mapping: dict[int, list[int]] = {}
    scores: dict[int, float] = {}
    for page_no in range(1, overlay_count + 1):
        section_no = assignment[page_no]
        mapping[page_no] = [section_no]
        scores[page_no] = round(float(sim_matrix[page_no][section_no]), 4)

    return PageMapping(
        mapping=mapping,
        scores=scores,
        base_sections=base_count,
        overlay_pages=overlay_count,
        mode="dp_one_to_one",
    )


def save_page_mapping_json(
    path: Path,
    mapping: PageMapping,
    *,
    base_path: Path | None = None,
    overlay_path: Path | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = mapping.to_json_dict(base_path=base_path, overlay_path=overlay_path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_page_mapping_json(path: Path) -> dict[int, list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("mapping", payload)
    if not isinstance(raw, dict):
        raise ValueError(f"Mapping JSON must be a dict or contain 'mapping': {path}")

    mapping: dict[int, list[int]] = {}
    for key, value in raw.items():
        page = int(key)
        if page <= 0:
            raise ValueError(f"Invalid page number in mapping: {key}")
        if isinstance(value, int):
            sections = [value]
        elif isinstance(value, list):
            sections = [int(v) for v in value]
        else:
            raise ValueError(f"Invalid mapping value for page {key}: {type(value)!r}")
        cleaned = sorted({section for section in sections if int(section) > 0})
        mapping[page] = cleaned
    return mapping


def collect_relationship_ids(elements: Sequence[Any], sect_pr: Any | None = None) -> list[str]:
    roots = list(elements)
    if sect_pr is not None:
        roots.append(sect_pr)

    result: list[str] = []
    seen: set[str] = set()
    for root in roots:
        for rid in root.xpath(".//@r:id | .//@r:embed | .//@r:link"):
            key = str(rid).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            result.append(key)
    return result


def collect_style_ids(elements: Sequence[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for root in elements:
        for style_id in root.xpath(".//w:pStyle/@w:val | .//w:rStyle/@w:val | .//w:tblStyle/@w:val"):
            key = str(style_id).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            result.append(key)
    return result


def copy_styles_from_source(
    src_doc: DocxDocument,
    dst_doc: DocxDocument,
    style_ids: Sequence[str],
) -> dict[str, str]:
    initial = [str(style_id).strip() for style_id in style_ids if str(style_id).strip()]
    if not initial:
        return {}

    src_styles_root = src_doc.part._styles_part.element
    dst_styles_root = dst_doc.part._styles_part.element

    src_styles_by_id = _styles_by_id(src_styles_root)
    dst_styles_by_id = _styles_by_id(dst_styles_root)

    required = _expand_style_dependencies(initial, src_styles_by_id)
    if not required:
        return {}

    used_ids = set(dst_styles_by_id.keys())
    style_map: dict[str, str] = {}
    for style_id in sorted(required):
        if style_id not in src_styles_by_id:
            continue
        if style_id in dst_styles_by_id:
            style_map[style_id] = _next_style_id(f"{style_id}_ovl", used_ids)
        else:
            style_map[style_id] = style_id
            used_ids.add(style_id)

    for style_id in sorted(required):
        if style_id not in src_styles_by_id:
            continue
        target_style_id = style_map[style_id]
        if target_style_id in dst_styles_by_id:
            continue

        style_clone = deepcopy(src_styles_by_id[style_id])
        style_clone.set(W_STYLE_ID_ATTR, target_style_id)
        if W_DEFAULT_ATTR in style_clone.attrib:
            del style_clone.attrib[W_DEFAULT_ATTR]
        _remap_style_dependencies_in_style(style_clone, style_map)
        dst_styles_root.append(style_clone)
        dst_styles_by_id[target_style_id] = style_clone

    return style_map


def copy_section_relationships(
    src_doc: DocxDocument,
    dst_doc: DocxDocument,
    elements: Sequence[Any],
    sect_pr: Any | None,
) -> dict[str, str]:
    src_part = src_doc.part
    dst_part = dst_doc.part
    dst_package = dst_part.package

    clone_memo: dict[str, Part] = {}
    rid_map: dict[str, str] = {}

    for old_rid in collect_relationship_ids(elements, sect_pr):
        if old_rid in rid_map:
            continue
        if old_rid not in src_part.rels:
            continue

        src_rel = src_part.rels[old_rid]
        if src_rel.is_external:
            new_rid = dst_part.rels.get_or_add_ext_rel(src_rel.reltype, src_rel.target_ref)
            rid_map[old_rid] = new_rid
            continue

        if src_rel.reltype == RT.IMAGE:
            image_part = dst_package.get_or_add_image_part(io.BytesIO(src_rel.target_part.blob))
            new_rid = dst_part.relate_to(image_part, RT.IMAGE)
            rid_map[old_rid] = new_rid
            continue

        cloned_target = _clone_part_tree(src_rel.target_part, dst_package, clone_memo)
        new_rid = dst_part.relate_to(cloned_target, src_rel.reltype)
        rid_map[old_rid] = new_rid

    return rid_map


def remap_relationship_ids(elements: Sequence[Any], rid_map: dict[str, str]) -> None:
    if not rid_map:
        return
    for root in elements:
        for node in root.iter():
            for attr_name in (R_ID_ATTR, R_EMBED_ATTR, R_LINK_ATTR):
                rid = node.get(attr_name)
                if rid and rid in rid_map:
                    node.set(attr_name, rid_map[rid])


def remap_style_ids(elements: Sequence[Any], style_map: dict[str, str]) -> None:
    if not style_map:
        return
    for root in elements:
        for node in root.iter():
            if node.tag not in {W_PSTYLE, W_RSTYLE, W_TBLSTYLE}:
                continue
            style_id = node.get(W_VAL_ATTR)
            if style_id and style_id in style_map:
                node.set(W_VAL_ATTR, style_map[style_id])


def renumber_drawing_ids(elements: Sequence[Any], start_id: int) -> int:
    next_id = max(1, int(start_id))
    for root in elements:
        for doc_pr in root.xpath(".//wp:docPr"):
            doc_pr.set("id", str(next_id))
            next_id += 1
    return next_id


def replace_pages(
    base_doc: DocxDocument,
    overlay_doc: DocxDocument,
    *,
    pages: Sequence[int],
    page_mapping: dict[int, list[int]],
    keep_page_size: str = "target",
) -> dict[str, Any]:
    if keep_page_size not in {"target", "source"}:
        raise ValueError("keep_page_size must be 'target' or 'source'")

    base_sections = get_section_ranges(base_doc)
    overlay_sections = get_section_ranges(overlay_doc)

    selected_pages = sorted({int(page) for page in pages})
    for page in selected_pages:
        if page <= 0 or page > len(overlay_sections):
            raise ValueError(f"Page out of range for overlay: {page} (1..{len(overlay_sections)})")

    used_base_sections: set[int] = set()
    operations: list[tuple[int, int, list[int]]] = []
    for page in selected_pages:
        raw_sections = page_mapping.get(page, [])
        if not raw_sections:
            continue
        section_ids = sorted({int(sec) for sec in raw_sections})
        if section_ids[0] <= 0 or section_ids[-1] > len(base_sections):
            raise ValueError(
                f"Mapping for page {page} points outside base sections: {section_ids[0]}..{section_ids[-1]}"
            )
        expected = list(range(section_ids[0], section_ids[-1] + 1))
        if section_ids != expected:
            raise ValueError(f"Mapping for page {page} must be contiguous, got: {section_ids}")
        overlap = used_base_sections.intersection(section_ids)
        if overlap:
            raise ValueError(f"Base sections mapped more than once: {sorted(overlap)}")
        used_base_sections.update(section_ids)
        operations.append((section_ids[0], page, section_ids))

    operations.sort(key=lambda item: item[0], reverse=True)

    body = base_doc._element.body
    next_drawing_id = _next_drawing_id(base_doc)
    replaced_pages: list[int] = []
    replaced_section_count = 0

    source_style_ids: set[str] = set()
    for _, page, _ in operations:
        source_style_ids.update(collect_style_ids(overlay_sections[page - 1].blocks))
    style_map = copy_styles_from_source(
        src_doc=overlay_doc,
        dst_doc=base_doc,
        style_ids=sorted(source_style_ids),
    )

    for _, page, section_ids in operations:
        source_section = overlay_sections[page - 1]
        target_sections = [base_sections[sec_id - 1] for sec_id in section_ids]
        target_last = target_sections[-1]

        insertion_index = _resolve_insertion_index(body, target_sections[0])
        target_blocks = [block for section in target_sections for block in section.blocks]
        for block in target_blocks:
            if block.getparent() is body:
                body.remove(block)

        copied_blocks = [deepcopy(block) for block in source_section.blocks]
        copied_sect_pr = deepcopy(source_section.sect_pr)
        target_sect_pr = deepcopy(target_last.sect_pr)
        merged_sect_pr = _merge_section_properties(
            source_sect_pr=copied_sect_pr,
            target_sect_pr=target_sect_pr,
            keep_page_size=keep_page_size,
        )

        rid_map = copy_section_relationships(
            src_doc=overlay_doc,
            dst_doc=base_doc,
            elements=source_section.blocks,
            sect_pr=source_section.sect_pr,
        )
        remap_relationship_ids(copied_blocks, rid_map)
        remap_relationship_ids([merged_sect_pr], rid_map)
        remap_style_ids(copied_blocks, style_map)

        _strip_paragraph_level_sect_pr(copied_blocks)
        if target_last.sect_pr_owner == "paragraph":
            _attach_sect_pr_to_last_paragraph(copied_blocks, merged_sect_pr)
        else:
            _set_body_sect_pr(base_doc, merged_sect_pr)

        next_drawing_id = renumber_drawing_ids(copied_blocks, next_drawing_id)
        for offset, block in enumerate(copied_blocks):
            body.insert(insertion_index + offset, block)

        replaced_pages.append(page)
        replaced_section_count += len(section_ids)

    return {
        "replaced_pages": sorted(replaced_pages),
        "replaced_pages_count": len(replaced_pages),
        "replaced_base_sections_count": int(replaced_section_count),
        "overlay_pages_total": len(overlay_sections),
        "base_sections_total": len(base_sections),
    }


def _paragraph_sect_pr(paragraph: Any) -> Any | None:
    p_pr = paragraph.find("./w:pPr", namespaces=NS)
    if p_pr is None:
        return None
    return p_pr.find("./w:sectPr", namespaces=NS)


def _clone_minimal_sect_pr() -> Any:
    return OxmlElement("w:sectPr")


def _extract_section_text_all(section: SectionRange) -> str:
    chunks: list[str] = []
    for block in section.blocks:
        chunks.extend(block.xpath(".//w:t/text()"))
    return " ".join(chunks)


def _styles_by_id(styles_root: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for style in styles_root.iterchildren():
        if style.tag != W_STYLE:
            continue
        style_id = style.get(W_STYLE_ID_ATTR)
        if not style_id:
            continue
        result[str(style_id)] = style
    return result


def _expand_style_dependencies(initial: Sequence[str], styles_by_id: dict[str, Any]) -> set[str]:
    required: set[str] = set()
    stack = list(initial)
    while stack:
        style_id = stack.pop()
        if style_id in required:
            continue
        required.add(style_id)
        style = styles_by_id.get(style_id)
        if style is None:
            continue
        for dep in (W_BASEDON, W_NEXT, W_LINK):
            dep_node = style.find(f"./w:{_local_name(dep)}", namespaces=NS)
            if dep_node is None:
                continue
            dep_style_id = dep_node.get(W_VAL_ATTR)
            if dep_style_id and dep_style_id not in required:
                stack.append(str(dep_style_id))
    return required


def _next_style_id(prefix: str, used_ids: set[str]) -> str:
    candidate = prefix
    idx = 1
    while candidate in used_ids:
        candidate = f"{prefix}_{idx}"
        idx += 1
    used_ids.add(candidate)
    return candidate


def _remap_style_dependencies_in_style(style: Any, style_map: dict[str, str]) -> None:
    for dep in (W_BASEDON, W_NEXT, W_LINK):
        dep_node = style.find(f"./w:{_local_name(dep)}", namespaces=NS)
        if dep_node is None:
            continue
        dep_style_id = dep_node.get(W_VAL_ATTR)
        if dep_style_id and dep_style_id in style_map:
            dep_node.set(W_VAL_ATTR, style_map[dep_style_id])


def _local_name(tag: str) -> str:
    if "}" not in tag:
        return tag
    return tag.rsplit("}", 1)[1]


def _extract_base_section_text(section: SectionRange) -> str:
    chunks: list[str] = []
    for block in section.blocks:
        chunks.extend(block.xpath(".//w:t[not(ancestor::w:txbxContent)]/text()"))
    return " ".join(chunks)


def _extract_overlay_section_text(section: SectionRange) -> str:
    chunks: list[str] = []
    for block in section.blocks:
        chunks.extend(block.xpath(".//w:txbxContent//w:t/text()"))
    return " ".join(chunks)


def _token_counter(text: str) -> Counter[str]:
    tokens: list[str] = []
    for token in _TOKEN_SPLIT_RE.split(text.lower()):
        cleaned = token.strip()
        if not cleaned:
            continue
        if cleaned.isdigit():
            tokens.append(cleaned)
            continue
        if len(cleaned) >= 2:
            tokens.append(cleaned)
    return Counter(tokens)


def _tokenize_for_alignment(text: str) -> list[str]:
    tokens: list[str] = []
    for token in _ALIGN_TOKEN_SPLIT_RE.split(text.lower()):
        cleaned = token.strip()
        if not cleaned:
            continue
        if cleaned in _ALIGN_STOP_TOKENS:
            continue
        if len(cleaned) == 1 and not cleaned.isdigit():
            continue
        tokens.append(cleaned)
    return tokens


def _alignment_vector(tokens: list[str], informative_tokens: set[str]) -> Counter[str]:
    vector = Counter(token for token in tokens if token in informative_tokens)
    for token in list(vector.keys()):
        if token.startswith("рисунок") or token.startswith("лист"):
            vector[token] *= 2
        if token.isdigit():
            value = int(token)
            if 500 <= value <= 999:
                vector[token] *= 1.5
    return vector


def _weighted_vector_norm(vector: Counter[str], idf: dict[str, float]) -> float:
    total = 0.0
    for token, value in vector.items():
        weighted = float(value) * idf.get(token, 1.0)
        total += weighted * weighted
    return math.sqrt(total)


def _weighted_cosine_similarity(
    left: Counter[str],
    right: Counter[str],
    left_norm: float,
    right_norm: float,
    idf: dict[str, float],
) -> float:
    if not left or not right:
        return 0.0
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    common = set(left.keys()).intersection(right.keys())
    if not common:
        return 0.0
    dot = 0.0
    for token in common:
        weight = idf.get(token, 1.0)
        dot += (float(left[token]) * weight) * (float(right[token]) * weight)
    denominator = left_norm * right_norm
    if denominator <= 0.0:
        return 0.0
    return dot / denominator


def _combine_counters(counters: Sequence[Counter[str]], start: int, take: int) -> Counter[str]:
    out = Counter[str]()
    for idx in range(start, start + take):
        out.update(counters[idx])
    return out


def _token_overlap_score(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = left & right
    shared = float(sum(intersection.values()))
    if shared <= 0.0:
        return 0.0
    left_total = float(sum(left.values()))
    right_total = float(sum(right.values()))
    if left_total <= 0.0 or right_total <= 0.0:
        return 0.0
    precision = shared / left_total
    recall = shared / right_total
    denom = precision + recall
    if denom <= 0.0:
        return 0.0
    return (2.0 * precision * recall) / denom


def _partname_template(source_part: Part) -> str:
    partname = str(source_part.partname)
    replaced, count = _PARTNAME_NUMBER_RE.subn("%d", partname, count=1)
    if count > 0:
        return replaced
    root, ext = os.path.splitext(partname)
    ext_value = ext or ".xml"
    return f"{root}%d{ext_value}"


def _clone_part_tree(source_part: Part, dst_package: Any, memo: dict[str, Part]) -> Part:
    key = str(source_part.partname)
    existing = memo.get(key)
    if existing is not None:
        return existing

    new_partname = dst_package.next_partname(_partname_template(source_part))
    source_blob = source_part.blob if source_part.blob is not None else b""
    cloned = Part(new_partname, source_part.content_type, source_blob, dst_package)
    memo[key] = cloned

    for rid, rel in source_part.rels.items():
        if rel.is_external:
            cloned.rels.add_relationship(rel.reltype, rel.target_ref, rid, is_external=True)
            continue
        if rel.reltype == RT.IMAGE:
            image_part = dst_package.get_or_add_image_part(io.BytesIO(rel.target_part.blob))
            cloned.rels.add_relationship(rel.reltype, image_part, rid, is_external=False)
            continue
        nested = _clone_part_tree(rel.target_part, dst_package, memo)
        cloned.rels.add_relationship(rel.reltype, nested, rid, is_external=False)

    return cloned


def _next_drawing_id(doc: DocxDocument) -> int:
    max_value = 0
    for raw in doc._element.body.xpath(".//wp:docPr/@id"):
        try:
            max_value = max(max_value, int(raw))
        except (TypeError, ValueError):
            continue
    return max_value + 1


def _resolve_insertion_index(body: Any, section: SectionRange) -> int:
    if section.blocks:
        first = section.blocks[0]
        if first.getparent() is body:
            return body.index(first)
    child_count = len(list(body.iterchildren()))
    return min(section.start_body_index, child_count)


def _merge_section_properties(source_sect_pr: Any, target_sect_pr: Any, keep_page_size: str) -> Any:
    merged = source_sect_pr
    # Preserve target pagination semantics (e.g., NEW_PAGE vs CONTINUOUS).
    _replace_child(merged, W_TYPE, target_sect_pr)
    if keep_page_size == "target":
        _replace_child(merged, W_PGSZ, target_sect_pr)
        _replace_child(merged, W_PGMAR, target_sect_pr)
    return merged


def _replace_child(target_parent: Any, child_tag: str, source_parent: Any) -> None:
    for child in list(target_parent):
        if child.tag == child_tag:
            target_parent.remove(child)
    replacement = None
    for child in source_parent:
        if child.tag == child_tag:
            replacement = deepcopy(child)
            break
    if replacement is not None:
        target_parent.append(replacement)


def _strip_paragraph_level_sect_pr(blocks: Sequence[Any]) -> None:
    for block in blocks:
        if block.tag != W_P:
            continue
        p_pr = block.find("./w:pPr", namespaces=NS)
        if p_pr is None:
            continue
        for sect_pr in p_pr.findall("./w:sectPr", namespaces=NS):
            p_pr.remove(sect_pr)


def _attach_sect_pr_to_last_paragraph(blocks: list[Any], sect_pr: Any) -> None:
    if not blocks or blocks[-1].tag != W_P:
        paragraph = OxmlElement("w:p")
        blocks.append(paragraph)
    else:
        paragraph = blocks[-1]

    p_pr = paragraph.find("./w:pPr", namespaces=NS)
    if p_pr is None:
        p_pr = OxmlElement("w:pPr")
        paragraph.insert(0, p_pr)

    for existing in p_pr.findall("./w:sectPr", namespaces=NS):
        p_pr.remove(existing)
    p_pr.append(sect_pr)


def _set_body_sect_pr(doc: DocxDocument, sect_pr: Any) -> None:
    body = doc._element.body
    current = body.find("./w:sectPr", namespaces=NS)
    if current is not None:
        body.remove(current)
    body.append(sect_pr)


__all__ = [
    "PageMapping",
    "SectionRange",
    "build_page_mapping",
    "collect_relationship_ids",
    "copy_section_relationships",
    "get_section_ranges",
    "load_page_mapping_json",
    "parse_pages_spec",
    "remap_relationship_ids",
    "renumber_drawing_ids",
    "replace_pages",
    "save_page_mapping_json",
]
