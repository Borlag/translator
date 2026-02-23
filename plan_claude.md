Context
The user previously translated aviation technical documents manually in ChatGPT — providing a large prompt with glossary, sending screenshots of 15-20 pages per chat session, and translating conversationally with full context. This produced excellent translations. The automated pipeline (docxru) produces significantly worse results because:

Critical bug: Batch translation mode (used for most segments) completely bypasses the glossary, custom system prompt, and all per-segment context — using only a 5-line generic English prompt instead of the full professional Russian translator prompt with glossary.
Formatting: Russian text is 30-50% longer than English, but font sizes are preserved exactly from the source, causing layout overflow in tables and textboxes. The user wants proactive font shrinking.


Part 1: Fix Translation Quality (Batch Mode + Context)
1A. Use full system prompt in batch mode
Root cause: When task == "batch_translate", the LLM clients use BATCH_SYSTEM_PROMPT_TEMPLATE (a 5-line English-only prompt with no glossary/expertise). This means ~90% of segments translated in batch mode get no glossary, no domain context, no professional translator instructions.
Files to modify:

src/docxru/llm.py — lines 65-70, 787-788, 1007-1008

Changes:

In OpenAIChatCompletionsClient.translate(): when task == "batch_translate", use self.translation_system_prompt (which already contains the full base prompt + custom prompt + glossary) and append the batch-specific JSON output instructions to it, instead of using BATCH_SYSTEM_PROMPT_TEMPLATE.
Same change in OllamaChatClient.translate().
Keep BATCH_SYSTEM_PROMPT_TEMPLATE as a fallback for providers that don't have translation_system_prompt.

Concrete change in OpenAIChatCompletionsClient.translate():
pythonelif task == "batch_translate":
    # Use the full translation prompt (with glossary + custom instructions)
    # and append batch-specific JSON output instructions
    system_prompt = self.translation_system_prompt + "\n\n" + BATCH_JSON_INSTRUCTIONS
Where BATCH_JSON_INSTRUCTIONS is a new constant containing only the JSON output format rules (extracted from the current BATCH_SYSTEM_PROMPT_TEMPLATE):
pythonBATCH_JSON_INSTRUCTIONS = """BATCH MODE — return ONLY valid JSON in the requested schema.
Do not add commentary outside the JSON.
Preserve all marker tokens exactly (⟦...⟧).
Preserve numbers, units, and punctuation."""
1B. Add per-segment context to batch items
Root cause: _build_batch_translation_prompt() sends only {"id": "...", "text": "..."} per segment — no section header, no table/textbox flag, no matched glossary.
Files to modify:

src/docxru/pipeline.py — _build_batch_translation_prompt() (line 908) and _translate_batch_once() (line 1049)

Changes:

Extend each batch item to include context:

python  {"id": seg_id, "text": text, "context": "SECTION: 32-10-00 | TABLE_CELL", "glossary": "term1->перевод1; term2->перевод2"}

In _translate_batch_once(), build matched glossary terms for the combined batch text and include section headers from each segment's context.
Update the batch prompt instructions to tell the LLM to use the per-item context for disambiguation.

1C. Update config defaults for better quality
File: config/config.agent_openai.yaml
SettingCurrentNewReasonreasoning_effortminimallowGives GPT-5 more room to think about contextbatch_segments64Smaller batches = better per-segment attentionfuzzy_enabledfalsetrueEnable reference translations from TM cachecontext_window_chars00Keep batch mode but improve its quality first
1D. Improve batch translation prompt
File: src/docxru/pipeline.py — _build_batch_translation_prompt() (line 908)
Update the prompt to:

Reference the per-item context and glossary fields
Instruct the LLM to use context for disambiguation and glossary for terminology
Keep the JSON output format requirements


Part 2: Formatting — Unconditional Font Size Reduction
2A. Add new config options
File: src/docxru/config.py
Add two new fields to the config dataclass:

font_shrink_body_pt: float = 0.0 (default 0 = disabled; user sets to 2.0)
font_shrink_table_pt: float = 0.0 (default 0 = disabled; user sets to 3.0)

2B. Implement global font shrink function
File: src/docxru/layout_fix.py
Add new function apply_global_font_shrink(segments, cfg) -> int:
pythondef apply_global_font_shrink(segments: list[Segment], cfg: PipelineConfig) -> int:
    """Unconditionally reduce font sizes for all translated segments."""
    body_shrink = float(cfg.font_shrink_body_pt)
    table_shrink = float(cfg.font_shrink_table_pt)
    if body_shrink <= 0 and table_shrink <= 0:
        return 0

    MIN_FONT_PT = 6.0
    changed_count = 0
    for seg in segments:
        if seg.paragraph_ref is None or not seg.target_tagged:
            continue

        shrink = table_shrink if seg.context.get("in_table") or seg.context.get("in_textbox") else body_shrink
        if shrink <= 0:
            continue

        para = seg.paragraph_ref
        changed = False
        for run in para.runs:
            if run.font.size is not None:
                current = run.font.size.pt
                new_size = max(MIN_FONT_PT, current - shrink)
                if new_size < current:
                    run.font.size = Pt(new_size)
                    changed = True
        if changed:
            changed_count += 1
    return changed_count
2C. Call the function in the pipeline
File: src/docxru/pipeline.py
Insert the call after the write-back loop and before layout checks:
python# After write-back, before layout check
if cfg.font_shrink_body_pt > 0 or cfg.font_shrink_table_pt > 0:
    shrunk = apply_global_font_shrink(segments, cfg)
    logger.info(f"Global font shrink applied to {shrunk} segments")
2D. Set config values
File: config/config.agent_openai.yaml
Add:
yamlfont_shrink_body_pt: 2.0
font_shrink_table_pt: 3.0

Files to Modify (Summary)
FileChangessrc/docxru/llm.pyNew BATCH_JSON_INSTRUCTIONS constant; modify batch system prompt in OpenAI and Ollama clientssrc/docxru/pipeline.pyUpdate _build_batch_translation_prompt() with context/glossary; update _translate_batch_once() to build per-item context; add global font shrink callsrc/docxru/layout_fix.pyAdd apply_global_font_shrink() functionsrc/docxru/config.pyAdd font_shrink_body_pt and font_shrink_table_pt config fieldsconfig/config.agent_openai.yamlUpdate reasoning_effort, batch_segments, fuzzy_enabled; add font shrink settings

Verification

Unit test: Run existing tests to ensure no regressions: python -m pytest
Dry run with mock: Run pipeline with provider: mock to verify:

Font shrink function is called and modifies font sizes correctly
Batch prompt now includes full system prompt text
Config parsing works for new fields


Integration test: Run a real translation of a small DOCX file with the updated config and verify:

Translated text uses correct glossary terms (check QA report for glossary_lemma_mismatch)
Font sizes in output DOCX are reduced by expected amounts
Tables have smaller font than body text


Compare output: Translate the same document with old config (batch_segments=6, minimal reasoning, no fuzzy TM) vs new config and compare translation quality in QA reports