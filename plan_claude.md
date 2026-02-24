рассмотри проект, найлди слабы е места и возможности для улучшения качества перевода, сделать его блестящимPlan: Improve Translation Quality to Brilliant Level
Context
The docxru translator is a sophisticated EN→RU aviation documentation translation system. After deep review of the codebase, I identified several critical weak spots where translation quality degrades silently. The biggest issues are: (1) batch mode loses nearly all quality instructions, (2) the checker has zero domain knowledge, (3) there are no few-shot examples, (4) untranslated English fragments go undetected. Fixing these will produce a measurably better, more consistent translation.

Weak Spots Found
CRITICAL — Batch system prompt is nearly empty
File: src/docxru/llm.py:70-81
BATCH_SYSTEM_PROMPT_TEMPLATE is a 5-line English stub. When batch_segments > 1 (the common production config), the rich Russian aviation instructions from SYSTEM_PROMPT_TEMPLATE are completely lost unless a custom system prompt file is configured. The batch prompt doesn't mention aviation context, imperative voice, glossary inflection, or safety at all.

CRITICAL — Checker system prompt has zero domain knowledge
File: src/docxru/checker.py:22-25
CHECKER_SYSTEM_PROMPT is 3 lines: "You are a bilingual EN->RU translation quality checker." It doesn't know about aviation terminology, Russian technical writing conventions, imperative voice, safety implications, or any of the translation rules the translator itself follows. The checker literally doesn't know what "good" looks like for this domain.

HIGH — No few-shot examples in any prompt
Neither the translation prompt nor the checker prompt include concrete input→output examples. Few-shot examples are one of the most effective ways to improve LLM output quality, especially for domain-specific tasks.

HIGH — Untranslated English fragments go undetected
File: src/docxru/validator.py
The validator checks markers, numbers, and length — but never detects when the LLM leaves entire English words/phrases untranslated in the output. This is the most common visible defect in machine translation.

MEDIUM — Temperature 0.1 reduces consistency
File: src/docxru/llm.py:942 (OpenAI client default)
For deterministic technical translation, temperature 0 is optimal. The current 0.1 introduces minor variation that can cause the same term to be translated differently on repeated runs.

MEDIUM — Consistency enforcement is post-hoc only
File: src/docxru/consistency.py
Term consistency issues are detected after the entire document is translated, but there's no mechanism to enforce consistency during translation. The recent_translations window (500 chars) is too small to reliably maintain term consistency across a 100+ page manual.

LOW — Hardcoded DOMAIN_TERM_PAIRS (120+ regex rules)
File: src/docxru/llm.py:597-914
The apply_glossary_replacements function has ~120 lines of hardcoded Safran-specific regex. This is a maintenance burden and mixes concerns.

Implementation Plan
1. Enrich batch system prompt with full aviation instructions
File: src/docxru/llm.py

Rewrite BATCH_SYSTEM_PROMPT_TEMPLATE to include the same core translation rules as SYSTEM_PROMPT_TEMPLATE: aviation domain, imperative voice, marker preservation, number/unit preservation, glossary inflection
Add JSON output format instructions inline (not as a separate weak appendix)
Keep the prompt in Russian for consistency with the main prompt
Update BATCH_JSON_INSTRUCTIONS to be a more detailed supplement, not a replacement
2. Enrich checker system prompt with domain knowledge
File: src/docxru/checker.py

Rewrite CHECKER_SYSTEM_PROMPT to include:
Aviation domain expertise (CMM/AMM/IPC context)
Russian technical writing standards (imperative for procedures, no literary style)
Key terminology rules (bolt≠болт vs винт, torque=момент затяжки, etc.)
What constitutes a defect: untranslated fragments, wrong register, wrong case agreement, ambiguous instructions
What is NOT a defect: minor stylistic preferences, valid inflection variants
Instruction to preserve all marker tokens in suggestions
3. Add few-shot examples to translation prompts
File: src/docxru/llm.py

Add 3-4 concrete examples to SYSTEM_PROMPT_TEMPLATE covering:
A procedure step (imperative voice): "Remove the bolt..." → "Снимите болт..."
A table cell/label (concise, nominative): "Torque Values" → "Значения момента затяжки"
A segment with markers preserved: ⟦S_1|B⟧Remove⟦/S_1⟧ the ⟦PN_1⟧ → ⟦S_1|B⟧Снимите⟦/S_1⟧ ⟦PN_1⟧
A warning/caution: "WARNING: Do not exceed..." → "ПРЕДУПРЕЖДЕНИЕ: Не превышайте..."
Add 2 examples to BATCH_SYSTEM_PROMPT_TEMPLATE showing correct JSON batch output format with markers preserved
4. Add untranslated-fragment detection to validator
File: src/docxru/validator.py

Add new function validate_untranslated_fragments(source_plain, target_plain) that:
Strips bracket tokens and numbers from target
Detects remaining Latin words (>2 chars, not in a whitelist of allowed terms: PN codes, standard abbreviations like MLG/NLG/CMM/SB, brand names like Safran)
If Latin word ratio exceeds threshold (e.g., >15% of total words), emit WARN issue untranslated_fragments
Add a configurable whitelist of allowed English terms (acronyms, brand names, standard codes)
Wire into validate_all()
5. Add doubled-word detection to validator
File: src/docxru/validator.py

Add validate_repeated_words(target_plain) that detects common LLM artifacts:
Same Russian word repeated consecutively ("болт болт", "Снимите Снимите")
Same phrase repeated ("момент затяжки момент затяжки")
Emit WARN issue repeated_words with details
6. Increase recent_translations budget for consistency
File: src/docxru/pipeline.py, src/docxru/llm.py

Increase recent_translations deque max size and char budget from 500 to 1500 chars
In _format_recent_translations_block, increase default max_chars to 1500
This gives the LLM much more context about how terms were previously translated in the same document
7. Lower default temperature to 0
File: src/docxru/config.py

Change default temperature from 0.1 to 0.0 in LLMConfig
This produces maximally deterministic translations, improving consistency
Files to Modify
File	Changes
src/docxru/llm.py	Rewrite BATCH_SYSTEM_PROMPT_TEMPLATE, BATCH_JSON_INSTRUCTIONS; add few-shot examples to SYSTEM_PROMPT_TEMPLATE; increase recent_translations budget
src/docxru/checker.py	Rewrite CHECKER_SYSTEM_PROMPT with full domain knowledge
src/docxru/validator.py	Add validate_untranslated_fragments(), validate_repeated_words(); wire into validate_all()
src/docxru/config.py	Change default temperature to 0.0
src/docxru/pipeline.py	Increase recent_translations budget; wire new validators
Verification
Run existing tests: pytest tests/ — all should pass
Specifically run:
pytest tests/test_validator.py — verify new validators don't break existing ones
pytest tests/test_llm_clients.py — verify prompt changes don't break client contracts
pytest tests/test_batch_mode.py — verify batch prompt changes work
pytest tests/test_checker.py — verify checker prompt changes work
Manual spot check: Run a mock translation (provider: mock) on a sample file to verify no structural regressions
Review prompt quality: Read the enriched prompts to verify they are clear, non-contradictory, and appropriately detailed