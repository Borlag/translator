# Max Quality Aviation EN->RU Profile

This project now includes a dedicated profile for high-quality technical aviation translation:

- Config: `config/config.max_quality_aviation.yaml`
- Scope: DOCX and PDF
- Focus: glossary consistency, checker pass, and formatting/layout safeguards

## What This Profile Enables

- OpenAI translation with strict structured output.
- Uses:
  - `general_prompt.md`
  - `glossary.md`
- Glossary-aware translation in prompt (`matched` mode).
- Fuzzy TM hints for consistency from prior translations.
- LLM checker enabled with safe auto-apply for DOCX edits.
- Layout and formatting safeguards enabled for DOCX:
  - layout checks
  - auto-fix passes
  - controlled font shrink
- Run outputs isolated under `output/max_quality/<run_id>/...`.

## Requirements

- `OPENAI_API_KEY` is set.
- For best DOCX formatting (fields, TOC, textboxes), Microsoft Word + COM are available.
  - Profile default: `mode: com`
  - If COM is unavailable, run with `--mode reflow` override.

## DOCX Translation (Recommended)

```powershell
docxru translate `
  --input "C:\path\source.docx" `
  --output "C:\path\source.ru.docx" `
  --config "config\config.max_quality_aviation.yaml" `
  --resume
```

Optional explicit fallback without COM:

```powershell
docxru translate `
  --input "C:\path\source.docx" `
  --output "C:\path\source.ru.docx" `
  --config "config\config.max_quality_aviation.yaml" `
  --mode reflow `
  --resume
```

## DOCX Postformat Pass

For difficult layouts (ABBYY-converted, dense tables/textboxes), run an extra pass:

```powershell
docxru postformat `
  --input "C:\path\source.ru.docx" `
  --output "C:\path\source.ru.postformat.docx" `
  --config "config\config.max_quality_aviation.yaml"
```

## PDF Translation

```powershell
docxru translate-pdf `
  --input "C:\path\source.pdf" `
  --output "C:\path\source.ru.pdf" `
  --config "config\config.max_quality_aviation.yaml" `
  --resume
```

For scanned PDFs:

```powershell
docxru translate-pdf `
  --input "C:\path\scan.pdf" `
  --output "C:\path\scan.ru.pdf" `
  --config "config\config.max_quality_aviation.yaml" `
  --ocr-fallback `
  --resume
```

## Quality Checks You Should Review

After each run, inspect artifacts in `output/max_quality/<run_id>/`:

- `qa_report.html`
- `qa.jsonl`
- `checker_suggestions.json`
- `checker_suggestions_safe.json`
- `run_status.json`
- `dashboard.html`

For quick visual page checks (DOCX):

```powershell
python scripts\compare_docx_pages.py `
  --left "C:\path\source.docx" `
  --right "C:\path\source.ru.docx" `
  --output-dir "tmp\docs\compare_source_vs_ru" `
  --backend auto `
  --dpi 150
```

