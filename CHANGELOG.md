# Changelog

Notable changes per release. Versions follow [semantic versioning](https://semver.org);
dates are the tag date.

## 0.4.0 — 2026-07-04

- New rule `deprecated-model`: flags a `model` id that is retired (the call 404s —
  raised as an error) or deprecated and scheduled for removal (a warning), and names
  the current model to switch to. Exact-match against a known list, so a live model
  or alias is never flagged.
- New rule `unsupported-params`: flags `temperature` / `top_p` / `top_k` set on a
  model that rejects them — the OpenAI reasoning (`o1`, `o3`, ...) series and the
  newest Anthropic models, where the parameter is a no-op or a 400.
- Both key off facts the parser already sees (the model string, the parameter names),
  so they stay silent whenever the model isn't a plain string literal.

## 0.3.0 — 2026-07-04

- Findings now carry a severity (`error`, `warning`, `info`). overllm raises only
  `warning` and above by default, so `static-prompt` (info) is silent unless you pass
  `--all` or `--min-severity info`. This is what makes it quiet on everyday code.
- Corpus scanner gained an "amateur / AI-authored" profile that biases toward the code
  most likely to contain waste: pre-1.0 OpenAI API usage, bot scripts, and repos built
  with Claude Code or Cursor.

## 0.2.1 — 2026-07-04

- JavaScript/TypeScript: `generateText` / `streamText` / `generateObject` are only
  treated as LLM calls when imported from the `ai` package, so a function of the same
  name from elsewhere is not flagged.
- Fixed a false positive where a "count the lines" style prompt matched the mechanical rule.

## 0.2.0 — 2026-07-04

- Added JavaScript and TypeScript support through tree-sitter, behind the optional
  `overllm[js]` extra. The Python core stays dependency-free.

## 0.1.3 — 2026-07-04

- Prompts held in a variable are now resolved and read, not just inline literals
  (`p = f"..."; create(messages=[{"content": p}])`).
- Detect the AWS Bedrock, Cohere, HuggingFace, and Replicate SDKs.
- `prompt-injection` is now web-only: it fires on untrusted remote request input
  (`request.args`, `request.json`, ...), not on local single-user input like `input()`
  or `sys.argv`, which is not an attack surface.

## 0.1.2 — 2026-07-03

- Split `llm-in-loop` into *batchable* (a real per-item map — flagged) and *necessary*
  (a `while` loop, a `range(...)` retry, or a conversation loop that feeds prior answers
  back in — not flagged). Only batchable loops are reported.
- Skip loops that already iterate over batches or chunks, so it does not tell
  already-batched code to batch.
- Corpus scanner skips vendored virtualenvs and `site-packages`.

## 0.1.1 — 2026-07-03

- Added the `prompt-injection` rule.
- Hardened the extraction patterns against false positives found by scanning real repos
  (bounded match windows so the verb and the datum have to be near each other).
- Added the corpus scanner (`scripts/scan_corpus.py`) used to measure precision on real code.

## 0.1.0 — 2026-07-03

- First release. Python detection via the standard-library `ast`, the `llm-mechanical`,
  `llm-extraction`, `llm-in-loop`, and `static-prompt` rules, and `human` / `json` /
  `sarif` / `markdown` output.
