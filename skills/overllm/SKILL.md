---
name: overllm
description: Find the AI/LLM calls you didn't need — where a model does what plain code does better. Static, local, no API key.
version: 1.0.1
metadata:
  openclaw:
    emoji: 🔍
    homepage: https://github.com/theadamdanielsson/overllm
    os: [darwin, linux, win32]
    requires:
      bins: [overllm]
---

# overllm — the needless-LLM-call linter

Use this skill when the user wants to **audit a codebase for wasteful or
unnecessary LLM/AI calls** — cases where they're paying latency, money, and
nondeterminism for something a library or a regex already does (parsing a date,
extracting JSON, sorting a list, classifying with a fixed set of labels).

overllm is a static linter. It parses code locally (Python via the stdlib `ast`,
JavaScript/TypeScript via tree-sitter), runs **no model**, makes **no network
call**, and needs **no API key**. Same code in, same result out.

## When to use

- "Audit this repo for unnecessary GPT/Claude/LLM calls."
- "Where am I calling a model to do something plain code could do?"
- "Check this file before I commit — am I wasting an API call here?"
- Pre-commit / CI gating on needless AI calls.

## How to run

Check the binary is available first; if missing, install it:

```bash
overllm --version || pip install "overllm[js]"
```

`overllm[js]` adds JavaScript/TypeScript support; plain `pip install overllm`
covers Python only.

Then scan. **Always pass `--format json` and `--exit-zero`** so you get
structured findings and overllm's non-zero "findings exist" exit code is not
mistaken for a crash:

```bash
overllm <path> --format json --exit-zero
```

- One file: `overllm app.py --format json --exit-zero`
- A folder: `overllm src/ --format json --exit-zero`
- Whole project: `overllm . --format json --exit-zero`

Useful flags:

- `--all` — also surface the `static-prompt` info-level rule (quiet by default).
- `--min-severity {error,warning,info}` — default shows `warning` and above.
- `--select <ids>` / `--ignore <ids>` — comma-separated rule ids to run or skip.
- `--format {human,json,sarif,markdown}` — use `json` to reason over findings,
  `markdown` when the user wants a report to paste somewhere.

## How to present findings

Each finding has a file, line, column, rule id, a message, and a concrete
deterministic replacement (`-> use sorted()`, `-> batch the inputs into a single
call`). When you report back:

1. Lead with the count: "N needless LLM calls across M files."
2. For each finding, give `file:line`, what it's doing wrong in one line, and the
   suggested fix. The fix is the value — always include it.
3. If overllm prints nothing, say so plainly: a clean codebase exits 0 with no
   findings. That's a pass, not a failure.

overllm is quiet by design and only fires on concrete patterns with a named
replacement — if it floods the user, treat that as a bug worth reporting, not
signal.

## What it does not do

It does not lint the code an AI wrote, does not run or call any model, and does
not change files — it only reads and reports. Running it is safe: worst case is
a few lines of output.
