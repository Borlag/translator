# Benchmark Baseline (Stage 0)

Status: draft template (baseline values not yet collected)

## Scope
- Sample: `samples/...abby_short_2.docx`
- Profiles:
  - `config/config.agent_openai.yaml`
  - `config/config.ollama.hy-mt.yaml`

## Metrics to capture
- `placeholders_mismatch`
- `style_tags_mismatch`
- `batch_fallback_single`
- `batch_validation_fallback`
- `writeback_skipped_due_to_errors`

## Run log checklist
1. Run translator with OpenAI profile on the fixed sample.
2. Run translator with Ollama profile on the same sample.
3. Save command line, config hash, and timestamp.
4. Copy metric values from `run.log` and QA issue summary.

## Baseline table
| Profile | placeholders_mismatch | style_tags_mismatch | batch_fallback_single | batch_validation_fallback | writeback_skipped_due_to_errors | Notes |
|---|---:|---:|---:|---:|---:|---|
| openai | TBD | TBD | TBD | TBD | TBD | |
| ollama | TBD | TBD | TBD | TBD | TBD | |
