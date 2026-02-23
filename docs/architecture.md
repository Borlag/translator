# docxru Architecture

## Overview

`docxru` translates technical DOCX and PDF content (EN -> RU) while preserving layout-sensitive structure.
Pipeline stages:

1. Segment extraction (`docx_reader.py`)
2. Tagging and shielding (`tagging.py`, `token_shield.py`)
3. TM lookup + LLM translation (`tm.py`, `llm.py`, `pipeline.py`)
4. Validation + repair (`validator.py`, `pipeline.py`)
5. Write-back + QA artifacts (`pipeline.py`, `qa_report.py`)
6. Optional post-layout normalization/fixes (`oxml_table_fix.py`, `layout_check.py`, `layout_fix.py`)
7. PDF-specific extraction/layout/write path (`pdf_reader.py`, `pdf_layout.py`, `pdf_writer.py`, `pdf_pipeline.py`)

## Data Flow

1. `collect_segments()` scans body, tables, optional headers/footers, and textboxes (`w:txbxContent`).
2. Each segment is transformed into tagged text with style markers and placeholder shielding.
3. Exact TM is checked first; optional fuzzy TM references are added to prompt context.
4. LLM translation runs in:
   - docxru grouped-request mode (`batch_segments > 1`, internal batching by nearby segments) for supported providers, or
   - sequential context-window mode (`context_window_chars > 0`) with `recent_translations`.
5. Output is validated (placeholders/tags/numbers + optional glossary-lemma checks), repaired if supported.
6. Successful output is written back to runs, then QA reports (`qa_report.html`, `qa.jsonl`) are generated.

PDF flow:

1. `extract_pdf_pages()` extracts text blocks/spans + font metadata from each PDF page.
2. `group_all_pages()` merges adjacent text blocks into translation segments (with simple table/column heuristics).
3. Existing TM + LLM + validator stack is reused in `pdf_pipeline.py` for segment translation.
4. `replace_block_text()` applies redaction and `insert_htmlbox` write-back (with optional scale-down).
5. Optional bilingual mode creates an OCG layer (`Russian Translation`) and writes RU overlay into that layer.
6. QA artifacts are generated with the same report writers as DOCX.

## Key Modules

- `src/docxru/pipeline.py`: orchestration, retries/fallbacks, TM profile keys, write-back and QA.
- `src/docxru/llm.py`: provider clients (`mock`, `openai`, `google`, `ollama`) and prompt construction.
- `src/docxru/tm.py`: exact/fuzzy SQLite TM with progress tracking.
- `src/docxru/docx_reader.py`: segment enumeration, TOC heuristics, textbox extraction.
- `src/docxru/validator.py`: marker/number validation and glossary morphology checks.
- `src/docxru/consistency.py`: glossary-term consistency checks across document segments.
- `src/docxru/layout_check.py`: overflow-risk heuristics (expansion/table/textbox).
- `src/docxru/layout_fix.py`: optional spacing/font reductions for risky segments.
- `src/docxru/eval.py`: batch evaluation harness and aggregate report generation.
- `src/docxru/pdf_models.py`: PDF dataclasses for spans/blocks/pages/segments.
- `src/docxru/pdf_reader.py`: PDF extraction via PyMuPDF `get_text("dict")`.
- `src/docxru/pdf_layout.py`: block grouping, table/column heuristics, segment creation.
- `src/docxru/pdf_font_map.py`: replacement-font selection and mapping logic.
- `src/docxru/pdf_writer.py`: redaction + HTML insertion + optional OCG layer integration.
- `src/docxru/pdf_pipeline.py`: end-to-end PDF translation orchestration.

## Important Configuration Knobs

- LLM behavior:
  - `llm.structured_output_mode`
  - `llm.glossary_prompt_mode`, `llm.glossary_match_limit`
  - `llm.batch_segments`, `llm.batch_max_chars`
  - `llm.context_window_chars`
- TM behavior:
  - `tm.fuzzy_enabled`, `tm.fuzzy_top_k`, `tm.fuzzy_min_similarity`
- Pipeline safeguards:
  - `abbyy_profile`
  - `glossary_lemma_check`
  - `layout_check`, `layout_expansion_warn_ratio`
  - `layout_auto_fix`, `layout_font_reduction_pt`, `layout_spacing_factor`
- PDF behavior:
  - `pdf.bilingual_mode`
  - `pdf.ocr_fallback`
  - `pdf.max_pages`
  - `pdf.max_font_shrink_ratio`
  - `pdf.block_merge_threshold_pt`
  - `pdf.skip_headers_footers`, `pdf.table_detection`
  - `pdf.font_map`, `pdf.default_sans_font`, `pdf.default_serif_font`, `pdf.default_mono_font`

## Known Failure Modes and Mitigations

- Marker corruption from LLM:
  - Strict validation + optional repair pass.
  - Fallback from batch to single-segment mode.
- Batch JSON instability:
  - Contract parser + issue codes (`batch_json_schema_violation`, `batch_fallback_single`).
- OCR/ABBYY layout rigidity:
  - Optional OXML normalization profiles (`safe`, `aggressive`).
- Terminology drift:
  - Matched glossary prompts + document glossary accumulation.
  - Optional consistency analysis (`consistency_term_variation`).
- Text expansion overflow:
  - Optional layout validation and auto-fix heuristics.
- Scanned PDFs with no text layer:
  - Optional OCR fallback through `ocrmypdf`.
- PDF font mismatch (missing Cyrillic glyphs):
  - Replacement font map + fallback defaults in PDF config.

## Plan Status (Claude Plan)

- P0 Critical fixes: implemented.
- P1 quality extensions: implemented (consistency checks, layout validation/auto-fix, ABBYY spacing normalization).
- P2 evaluation/docs: implemented (`docxru eval` harness + architecture documentation).
- P3 PDF track (from claude_2 plan): implemented baseline (`translate-pdf`, OCR fallback, bilingual OCG, PDF QA).
