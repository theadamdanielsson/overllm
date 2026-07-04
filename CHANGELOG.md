# Changelog

Notable changes per release. Versions follow [semantic versioning](https://semver.org);
dates are the tag date.

## 0.6.2 — 2026-07-04

Test-only fix; the 0.6.1 runtime code is unchanged. The deep-recursion regression
test parsed a pathologically nested prompt, which tripped CPython's own parser
recursion limit during AST construction on Python 3.11 (a parser limit, not an
overllm one, so CI was red there). The test now builds the AST directly, so it
exercises overllm's prompt-walk depth guard on every version. CI is green across
3.9–3.13.

## 0.6.1 — 2026-07-04

Correctness patch from an adversarial red-team of 0.6.0. **If you ran
`overllm --fix` on 0.6.0, re-check any file that had a non-ASCII character
(emoji, accented letter) on the same line as an edited parameter — the edit
could have been misplaced.** 0.6.1 fixes that.

- **`--fix` no longer corrupts files with non-ASCII on the edited line.** AST
  column offsets are UTF-8 *byte* offsets; the edit math now runs in byte space,
  so a multibyte character earlier on the line can no longer shift — or crash
  (`IndexError`) — an edit. All edits are bounds-checked and re-parsed before
  writing.
- **A single pathological file can no longer abort a whole scan.** Bounded the
  recursion in prompt extraction and contained any detector error per file (the
  Python path now matches the JS path's isolation).
- **Baseline: distinct multi-line calls no longer fingerprint-collide** (keyed on
  the whole call, not just its first line, so a genuinely new finding isn't
  masked), and **paths are normalized cwd-relative** so a baseline written locally
  matches the same repo in CI.

## 0.6.0 — 2026-07-04

- **Wrapper recall.** overllm now follows a project's own LLM wrapper. When a
  function calls the SDK directly and one of its parameters feeds the prompt
  (`def ask(prompt): client.chat.completions.create(...)`), calls to that
  function are analyzed too — their prompt argument is read and loop/cost rules
  apply. This catches the common case where the waste is at the call site, not
  inside the wrapper. Same-file, unambiguous names only; method wrappers match
  `self.<name>(...)`; a `**kwargs` splat is left alone (a precision choice).
- **`llm_calls` config allowlist.** Declare your own wrapper by name
  (`[tool.overllm] llm_calls = ["myapp.llm.ask"]`) to reach calls that go through
  a framework, a provider layer, or a `**kwargs` splat, where the SDK call isn't
  statically visible. Validated on real repos: unlocked 29 previously-invisible
  calls in a provider-mediated codebase.
- **Baseline / ratchet mode.** `--write-baseline` snapshots current findings to
  `overllm-baseline.json`; `--baseline` then reports only findings new since the
  snapshot, and `--update-baseline` prunes ones you've fixed. Fingerprints key on
  rule + file + code + model (no line number), so they survive edits elsewhere,
  and count occurrences instead of netting them, so a new wasteful call still
  fires even when an old one was removed. This lets a legacy repo adopt overllm
  in one command without drowning in pre-existing findings.
- **Wrapper recall.** overllm now follows a project's own LLM wrapper. When a
  function calls the SDK directly and one of its parameters feeds the prompt
  (`def ask(prompt): client.chat.completions.create(...)`), calls to that
  function are analyzed too — their prompt argument is read and loop/cost rules
  apply. This catches the common case where the waste is at the call site, not
  inside the wrapper. Same-file, unambiguous names only; method wrappers match
  `self.<name>(...)`; a `**kwargs` splat is left alone (a precision choice).
- **`llm_calls` config allowlist.** Declare your own wrapper by name
  (`[tool.overllm] llm_calls = ["myapp.llm.ask"]`) to reach calls that go through
  a framework, a provider layer, or a `**kwargs` splat, where the SDK call isn't
  statically visible. Validated on real repos: unlocked 29 previously-invisible
  calls in a provider-mediated codebase.
- **Concurrency-aware `llm-in-loop`.** A call dispatched per item into an
  `asyncio.gather` / `as_completed` now says the cost scales N× but latency is
  amortized, instead of claiming N latencies.
- **`--fix` (safe autofix).** Applies only the two mechanically-exact fixes:
  removes a sampling param a model rejects (`unsupported-params`, safe), and —
  gated behind `--unsafe-fixes` — swaps a retired/deprecated model id for its
  replacement (`deprecated-model`). `--diff` previews the patch without writing.
  Edits are AST-anchored (never a regex on source, so strings and comments are
  untouched), non-overlapping, and the result is re-parsed so a fix can never
  leave a file that won't compile. The five judgment rules are never auto-applied.
  Validated: on a 508-file repo, all files still parsed and every targeted
  finding cleared, idempotently.
- **`--format github`.** Emits GitHub Actions workflow-command annotations, so
  findings show inline on the PR diff. The Action now prints these too.
- **SARIF → code scanning.** A documented workflow to upload `--format sarif`
  results via `github/codeql-action/upload-sarif` (free on public repos).

## 0.5.1 — 2026-07-04

- Gate the `mcp` dependency to Python 3.10+ (the MCP SDK requires it), so the
  linter still installs and tests clean on 3.9. Core overllm remains 3.9+.
- Add the MCP registry ownership marker to the README and publish `server.json`
  at 0.5.1.

## 0.5.0 — 2026-07-04

- overllm now ships an MCP server (`pip install "overllm[mcp]"`, run with
  `overllm-mcp`), so an AI agent can call it directly. Two tools: `scan_path`
  for files and directories on disk, `scan_code` for a snippet passed inline.
  Both return the same deterministic findings as the CLI and change nothing.
- `server.json` for publishing to the official MCP registry.

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
