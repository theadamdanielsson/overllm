---
name: overllm
description: Find the LLM call you didn't need — your GPT call is a regex. Static, zero-config linter that flags unnecessary GPT/Claude/AI API calls a regex, the stdlib, or a library already does (parsing dates, extracting emails/JSON, sorting, classifying with fixed labels). Runs locally, no model, no API key, nothing phones home. Python + JS/TS. Observability tells you what you spent; overllm finds the call you never should have made — cutting LLM/API cost as a result, not a promise.
version: 1.1.0
metadata:
  openclaw:
    emoji: 🔍
    homepage: https://github.com/theadamdanielsson/overllm
    os: [darwin, linux, win32]
    requires:
      bins: [overllm]
---

# overllm — find the LLM call you didn't need

**Your GPT call is a regex.** Use this skill to audit a codebase for
**wasteful or unnecessary GPT/Claude/LLM API calls** — the ones where you pay
latency, money, and nondeterminism for something a regex, the stdlib, or a
library already does (parsing a date, extracting an email or JSON, sorting a
list, classifying with a fixed set of labels). overllm finds the model calls you
can delete outright; **cutting your LLM/API bill is the result, not the pitch.**

Where the other tools sit: observability (Helicone, Langfuse) tells you what you
already spent; routers and caches make a call cheaper. **overllm is the only one
that asks whether the call should exist at all — statically, before you ship.**

overllm is a static, zero-config linter. It parses code locally (Python via the
stdlib `ast`, JavaScript/TypeScript via tree-sitter), runs **no model**, makes
**no network call**, needs **no API key**, and **sends no telemetry** — nothing
phones home. Same code in, same result out.

## Your GPT call is a regex — see it

```python
# before — a paid, nondeterministic API round-trip to pull an email out of text
email = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": f"extract the email from: {text}"}],
).choices[0].message.content

# after — free, instant, deterministic
import re
email = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
```

overllm flags the first as `llm-extraction` and points you straight at the fix.
It is **not** an "AI slop" / AI-generated-code detector — it flags wasteful AI
*usage* (calls that shouldn't exist), not AI-written code smell.

## When to use

- "How do I reduce my OpenAI/Claude API bill / cut token usage?"
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
- `--format {human,json,sarif,markdown,github}` — use `json` to reason over
  findings, `markdown` for a pasteable report, `github` for CI annotations.
- `--baseline` / `--write-baseline` — adopt on an existing repo: snapshot current
  findings, then report only new ones.
- `--fix` (safe) / `--fix --unsafe-fixes` / `--diff` — auto-remove a rejected
  sampling param, or (unsafe) swap a retired model id; `--diff` previews.

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
