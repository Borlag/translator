Context
The project is a DOCX EN→RU aviation/technical document translation pipeline (src/docxru/). It uses paragraph-level segmentation, inline style tagging (⟦S_n|B|I⟧), token shielding for part numbers/dimensions, multi-provider LLM backends (OpenAI, Google, Ollama, Mock), translation memory (SQLite), and QA reporting.
The user wants three deliverables:

T1: Repo audit (architecture, failure modes, prioritized fix plan)
T2: Implement improvements (better context, glossary enforcement, layout validation, auto-fix)
T3: Evaluation harness (python -m docxru eval)

This plan focuses on the implementation work (T2 + T3), with T1 delivered as documentation.

Identified Bugs & Root Causes
Bug 1: No neighbor context for short/medium paragraphs

File: pipeline.py:364-383 — _should_attach_neighbor_context()
Paragraphs 25-79 chars (common procedural steps like "Remove the bolt.") get NO prev_text/next_text
Table cell segments always get False (line 365-366)

Bug 2: No cross-segment translation context

pipeline.py:1119-1127 attaches SOURCE neighbor text before translation, never previous TRANSLATED text
Each segment is translated without knowledge of how neighbors were rendered in Russian

Bug 3: Document-level glossary terms not accumulated

pipeline.py:1155-1159 — glossary_prompt_mode="matched" only injects terms matching the current segment
A term defined in segment 5 is invisible to segments 6-100

Bug 4: Final cleanup runs on untranslated segments

pipeline.py:386-406 — _apply_final_run_level_cleanup() iterates ALL segments, not just translated ones
Rules like \bTable\b → Таблица could fire on intentionally-skipped English content

Bug 5: Text box content not extracted

docx_reader.py only walks <w:body>, tables, headers/footers
Text inside <w:txbxContent> (text boxes from PDF→DOCX conversion) is invisible

Bug 6: No layout validation or auto-fix

No mechanism detects post-translation overflow, clipping, or off-page elements
No auto-fixes for text expansion (RU is typically 20-40% longer than EN)


Implementation Plan
Phase 1 — P0: Critical Fixes
1.1 Sliding context window + sequential mode
Files: pipeline.py, llm.py, config.py
Changes in pipeline.py:

Relax _should_attach_neighbor_context(): always return True for segments with Latin text; use shorter budget (100 chars) for table cells instead of blocking them entirely
Add recent_translations: list[tuple[str, str]] ring buffer (max 3 entries) in translate_docx()
After each segment is translated, append (source_plain, target_plain) to the buffer
Before translating each segment, attach seg.context["recent_translations"] with the last 3 entries
When context_window_chars > 0, force sequential processing (override concurrency to 1) and log a message

Changes in llm.py:

In build_user_prompt() (line 165), add a RECENT_TRANSLATIONS block after TM_REFERENCES formatting the recent source→target pairs

Changes in config.py:

Add context_window_chars: int = 600 to LLMConfig

1.2 Document-level glossary accumulation
Files: pipeline.py, llm.py
Changes in pipeline.py:

Add document_glossary: dict[str, str] = {} accumulator in translate_docx()
After processing each segment, merge its matched_glossary_terms into document_glossary
Before translating each segment, attach seg.context["document_glossary"] (limited to last N entries by config glossary_match_limit)

Changes in llm.py:

In build_user_prompt(), add a DOCUMENT_GLOSSARY block (separate from MATCHED_GLOSSARY) showing accumulated terms

1.3 Text box content extraction
Files: docx_reader.py

Add _iter_textbox_paragraphs(doc) function: walk document XML for <w:txbxContent> elements, yield (location_prefix, Paragraph) pairs
Modify collect_segments(): call _iter_textbox_paragraphs() after body/header/footer passes, with location prefix textbox{n}/p{m}
Add safety: skip if structure is unexpected, log warnings

1.4 Fix final cleanup applying to untranslated segments
Files: pipeline.py

In _apply_final_run_level_cleanup(), skip segments where seg.target_shielded_tagged is None


Phase 2 — P1: Significant Quality Gains
2.1 Cross-segment consistency checking
New file: src/docxru/consistency.py (~150 lines)
Functions:

build_phrase_translation_map(segments, glossary_matchers) -> dict[str, set[str]] — for each EN glossary phrase found in source, collect all RU translations used
detect_inconsistencies(phrase_map) -> list[Issue] — flag cases where same EN phrase got multiple distinct RU renderings
report_consistency(segments, glossary_matchers) -> list[Issue] — top-level orchestrator

Changes in pipeline.py:

After Stage 2, call report_consistency() and log summary
Append issues to QA report

2.2 Layout validation module
New file: src/docxru/layout_check.py (~200 lines)
Functions:

check_text_expansion(segments, warn_ratio=1.5) -> list[Issue] — flag segments where RU/EN char ratio exceeds threshold
check_table_cell_overflow(doc) -> list[Issue] — parse <w:tcW> cell widths, estimate if translated text fits (heuristic: ~120 twips/char at 10pt)
check_textbox_overflow(doc) -> list[Issue] — parse text box dimensions from <wp:extent>, compare against content length
validate_layout(doc, segments, cfg) -> list[Issue] — orchestrator

Changes in pipeline.py:

After Stage 3 write-back, call validate_layout() if enabled
Log results and add issues to QA report

Changes in config.py:

Add to PipelineConfig: layout_check: bool = False, layout_expansion_warn_ratio: float = 1.5

2.3 Layout auto-fix for text expansion
New file: src/docxru/layout_fix.py (~200 lines)
Functions:

reduce_font_size(paragraph, reduction_pt=0.5) — reduce all run font sizes by N points
reduce_cell_spacing(cell, factor=0.8) — reduce paragraph before/after spacing in a cell
fix_expansion_issues(doc, issues, cfg) -> int — for each overflow issue: first try spacing reduction, then font size reduction; return fix count

Changes in pipeline.py:

After layout validation, call fix_expansion_issues() if layout_auto_fix is enabled

Changes in config.py:

Add: layout_auto_fix: bool = False, layout_font_reduction_pt: float = 0.5, layout_spacing_factor: float = 0.8

2.4 Enhanced ABBYY artifact cleanup
Files: oxml_table_fix.py

Add normalize_line_spacing(document) -> int: convert <w:spacing w:lineRule="exact"> to lineRule="atLeast" to allow RU text expansion
Update normalize_abbyy_oxml(): include line spacing normalization in aggressive profile


Phase 3 — P2: Evaluation Harness & Documentation
3.1 Evaluation harness
New file: src/docxru/eval.py (~250 lines)
Data structures:

EvalResult dataclass: file path, segment count, TM hits, error/warning counts, glossary compliance rate, text expansion stats, layout issues, auto-fixes applied, timing

Functions:

evaluate_single(input_path, output_dir, cfg) -> EvalResult — run translation, collect all metrics
evaluate_batch(input_dir, output_dir, cfg) -> list[EvalResult] — process all .docx files
write_eval_report(results, report_path) — JSON report with per-file + aggregate metrics
Exit non-zero if error thresholds fail

Changes in cli.py:

Add eval subcommand: --input-dir, --output-dir, --config, --report, --threshold-errors

CLI:
python -m docxru eval --input-dir samples/ --output-dir out/ --config config/config.example.yaml --report out/report.json
3.2 Architecture documentation
New file: docs/architecture.md

Module descriptions with data flow diagram
Configuration knobs reference
Known failure modes and mitigations
Fix plan summary (P0/P1/P2 with status)


Files Modified/Created Summary
FileActionPhasesrc/docxru/pipeline.pyModify (context window, glossary accum, consistency hook, layout hook, cleanup fix)P0+P1src/docxru/llm.pyModify (new prompt blocks for recent_translations, document_glossary)P0src/docxru/config.pyModify (new config fields)P0+P1src/docxru/docx_reader.pyModify (text box extraction)P0src/docxru/oxml_table_fix.pyModify (line spacing normalization)P1src/docxru/cli.pyModify (eval subcommand)P2src/docxru/consistency.pyCreateP1src/docxru/layout_check.pyCreateP1src/docxru/layout_fix.pyCreateP1src/docxru/eval.pyCreateP2docs/architecture.mdCreateP2
Existing Code to Reuse

token_shield.strip_bracket_tokens() — for extracting plain text in consistency checks
validator.validate_all() pattern — for layout_check module structure
qa_report.write_qa_report() / write_qa_jsonl() — for eval report output
oxml_table_fix.normalize_abbyy_oxml() — extend with new normalization rules
models.Issue / models.Severity — for all new validation modules
scripts/render_docx_pages.py — for optional visual comparison in eval
config.PipelineConfig dataclass pattern — for new config fields

Verification Plan

Unit tests: Add tests for each new module (test_consistency.py, test_layout_check.py, test_layout_fix.py, test_eval.py)
Integration test: Run python -m docxru translate on samples/test_1.docx with mock LLM before and after changes, compare QA reports
Eval harness: Run python -m docxru eval --input-dir samples/ --output-dir out/ --config config/config.example.yaml --report out/report.json with mock LLM, verify JSON report structure
Existing tests: Run pytest tests/ to ensure no regressions
Manual spot-check: Translate one sample with OpenAI, open output in Word, visually inspect formatting