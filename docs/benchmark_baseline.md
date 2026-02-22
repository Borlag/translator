# Benchmark Baseline (Stage 0)

Date: 2026-02-22

## Scope
- Sample: `samples/32-12-22_pn201587001,201587002_R.69_SAFRAN (1-469) — abby_short_2.docx`
- Profiles:
  - OpenAI baseline config: `tmp/baseline/config.openai.baseline120.yaml`
  - Ollama baseline config: `tmp/baseline/config.ollama.baseline120.yaml`
- Controlled slice: `--max-segments 120` (same for both providers).

## Why 120 segments
- Full-file run for OpenAI (`5754` segments) exceeded a 30-minute local timeout in this session.
- To keep measurements reproducible right now, baseline is fixed to the first 120 segments for both profiles.

## Run metadata

### OpenAI
- Timestamp: `2026-02-22 12:53:57`
- Command: `docxru translate --input samples/...abby_short_2.docx --output tmp/baseline/out_openai_120.docx --config tmp/baseline/config.openai.baseline120.yaml --max-segments 120`
- Config SHA256: `d76de58634b9e0c7659de4aeec4e23008f45210f30f103adde80faa42cd4e5b1`
- Artifacts:
  - `tmp/baseline/out_openai_120.docx`
  - `tmp/baseline/qa_openai_120.jsonl`
  - `tmp/baseline/qa_openai_120.html`
  - `tmp/baseline/run_openai_120.log`

### Ollama
- Timestamp: `2026-02-22 12:56:07`
- Command: `docxru translate --input samples/...abby_short_2.docx --output tmp/baseline/out_ollama_120.docx --config tmp/baseline/config.ollama.baseline120.yaml --max-segments 120`
- Config SHA256: `dee63effea67a5ee6535b31af6f68af80e33649ad6ce6d32b042cf02222a9a6e`
- Artifacts:
  - `tmp/baseline/out_ollama_120.docx`
  - `tmp/baseline/qa_ollama_120.jsonl`
  - `tmp/baseline/qa_ollama_120.html`
  - `tmp/baseline/run_ollama_120.log`

## Metrics
Counts were extracted from `qa_*_120.jsonl` by `code`.

| Profile | placeholders_mismatch | style_tags_mismatch | batch_fallback_single | batch_validation_fallback | writeback_skipped_due_to_errors | Notes |
|---|---:|---:|---:|---:|---:|---|
| openai | 0 | 0 | 0 | 0 | 0 | grouped mode on (`batch_segments=6`) |
| ollama | 0 | 0 | 0 | 0 | 0 | single-segment mode (`batch_segments=1`) |

## Next baseline step
- Re-run the same methodology on full-file scope (remove `--max-segments`) when runtime budget permits.
