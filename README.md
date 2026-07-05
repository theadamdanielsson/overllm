<!-- mcp-name: io.github.theadamdanielsson/overllm -->

# overllm

**Catch the LLM/AI calls you didn't need.**

overllm is a small, fast linter with one job: find the places in your code where you call an AI model to do something plain code does better. You called GPT to parse a date. You called a model to extract JSON that `json.loads` already handles. You are paying latency, money, and nondeterminism for a regex.

It reads your code with a real parser: Python through the standard-library `ast`, and JavaScript and TypeScript through tree-sitter. No model runs, no network, no API key. Same code in, same result out. Fast enough for a pre-commit hook.

Cost dashboards and caches deal with calls you've already decided to make. overllm asks the earlier question — did you need the call at all — and answers it from the source, before you run anything. Everyone else lints the code the AI wrote; overllm catches where you're paying an AI to do what a library already does.

## Install

```bash
pip install overllm          # Python
pip install "overllm[js]"    # adds JavaScript / TypeScript support
```

## Use it

Point it at one file first, so you can see what it flags before you run it on everything:

```bash
overllm app.py       # one file
overllm src/         # a folder
overllm .            # the whole project
```

It reads your code and prints what it finds. By default it writes nothing and changes nothing — the worst case of a plain run is a few lines of output. The one mode that edits files is the opt-in `--fix` (below), and only for the two mechanically-safe fixes.

Example output:

```
app.py:42:5 llm-mechanical  LLM call asks the model to sort
    resp = client.chat.completions.create(model="gpt-4o", messages=[...])
    -> use sorted()

app.py:88:1 llm-in-loop  LLM call inside a loop: one API round-trip per iteration
    completion(model="gpt-4o", messages=[{"role": "user", "content": f"tag {x}"}])
    -> batch the inputs into a single call, cache repeated results, or use a function

2 needless LLM calls in 1 file.
```

It is quiet by default. Only `warning` and `error` findings show, so a clean project prints nothing and exits 0 — most codebases surface a handful or none. If it floods you, treat that as a bug and [open an issue](https://github.com/theadamdanielsson/overllm/issues).

overllm exits non-zero when it finds something, so it gates a commit or a CI check. Pass `--exit-zero` to report without failing.

## Your code stays on your machine

overllm is static analysis. It parses your files locally, then prints what it found. It never uploads your code, never calls an API, needs no key, and sends no telemetry — there is no model in the loop and nothing phones home. Pull your network cable and it runs exactly the same.

The core is a couple thousand lines of Python with no required dependencies, so you can read all of it before you trust it. `overllm[js]` adds tree-sitter to parse JavaScript and TypeScript; that is the only optional dependency.

## Rules

Every rule fires only on a concrete code pattern, and every finding names the deterministic replacement. It stays silent when it is not sure.

By default overllm only raises **warning** and above, so it is quiet on your everyday code. `static-prompt` is **info** and stays silent unless you ask for it with `--all` or `--min-severity info`.

| Rule | Severity | Fires when | Suggests |
| --- | --- | --- | --- |
| `llm-mechanical` | error | The prompt asks for a mechanical transform: sort, reverse, count, sum, deduplicate, change case, base64, arithmetic on literals. | the one-line stdlib equivalent |
| `llm-extraction` | error | The prompt asks the model to extract an email, URL, date, or number. | a regex, `datetime`, or `urllib.parse` |
| `prompt-injection` | error | Untrusted web-request input (`request.args`, `request.json`, ...) flows straight into the prompt. | keep it in a separate user message, validate it, constrain the model |
| `llm-in-loop` | warning | An LLM call runs once per loop iteration (real N calls, not streaming). | batch, cache, or move it out of the loop |
| `deprecated-model` | error / warning | The `model` id is a retired model (the call 404s) or one that is deprecated and scheduled for removal. | switch to the current model it names |
| `unsupported-params` | warning | `temperature` / `top_p` / `top_k` is set on a model that rejects them — the OpenAI reasoning (`o1`, `o3`, ...) series and the newest Anthropic models. | remove the parameter; steer with the prompt instead |
| `json-mode-missing-json` | error | `response_format={"type": "json_object"}` is set but the fully-static prompt never contains the word "json" — a guaranteed OpenAI 400. | add "json" to a message, or use a `json_schema` format |
| `static-prompt` | info | The user prompt is a compile-time constant, no variables. The input is fixed, so the call buys nothing. | precompute or cache the result |

The last two check the call itself, not the prompt: a model id that no longer exists, or a knob the model ignores. Both are matched exactly against a known list, so a live model or alias is never flagged. The lists track provider deprecation pages and need updating over time.

It detects the OpenAI, Anthropic, Google, Mistral, Cohere, Groq, AWS Bedrock, HuggingFace, Replicate, LangChain, LiteLLM, and Ollama SDKs in Python, the Vercel AI SDK (`generateText`, `streamText`, `generateObject`) and the openai / anthropic node SDKs in JavaScript and TypeScript, and raw HTTP requests to those hosts. It also follows a model through **LCEL composition** — a `chain = prompt | model | parser` pipe, a bound model (`.with_structured_output(...)`), or an alias — so `chain.invoke(...)` is seen; embeddings calls (`embeddings.create`) count too. When a call goes through your own wrapper or a framework overllm can't see, name it in `llm_calls` (below).

## Silence a false positive

```python
resp = client.chat.completions.create(...)  # overllm: ignore
resp = client.chat.completions.create(...)  # overllm: ignore=llm-in-loop
```

Put `# overllm: ignore-file` at the top of a file to skip the whole file.

## Configure

In `pyproject.toml` (Python 3.11+):

```toml
[tool.overllm]
ignore = ["llm-in-loop"]
exclude = ["examples/", "migrations/"]
llm_calls = ["myapp.llm.ask", "chat_service.complete"]
```

Or on the command line: `--select`, `--ignore`, `--min-severity`, `--all`, and `--config PATH` (`exclude` is config-only). Run `overllm --help` for the full list.

### Teaching overllm your own wrapper

Most code doesn't call the SDK inline — it wraps it (`def ask(prompt): client.chat.completions.create(...)`). overllm follows that wrapper on its own when it lives in the same file. When the call goes through a framework, a provider layer, or a `**kwargs` splat, overllm can't see the SDK call, so tell it the wrapper's name in `llm_calls`. After that, calls to `ask(...)` are treated like LLM calls — it reads the prompt argument and runs the loop and cost rules. A name matches bare (`ask`), dotted (`myapp.llm.ask`), or that dotted path imported under its short name.

## Adopt on an existing codebase (baseline)

Dropping overllm on an old repo gives you a wall of findings you'll never get
through. Snapshot them once and have CI flag only what's new after that:

```bash
overllm . --write-baseline        # writes overllm-baseline.json — commit it
overllm . --baseline              # reports only findings new since the snapshot
overllm . --update-baseline       # ratchet: also drop entries you've since fixed
```

The snapshot keys each finding on rule + file + code + model, not the line
number, so unrelated edits above it don't invalidate it. And it counts how many
times each one shows up instead of just diffing totals, so a new bad call still
trips the check even if you happened to delete an old one somewhere else.

## Fix what's safe (`--fix`)

Two of the rules have one obvious fix, so overllm can just do it for you:

```bash
overllm . --fix                  # drop a sampling param the model rejects (safe)
overllm . --fix --unsafe-fixes   # also swap a retired model id for its replacement
overllm . --fix --diff           # print the patch, don't touch anything
```

Plain `--fix` only does the safe one (`unsupported-params`). Swapping a model id
changes what your code actually does at runtime, so that's behind
`--unsafe-fixes`. Fixes edit the syntax tree, not the raw text, so your comments
and strings are left alone, and overllm re-parses the file before saving — if the
edit would break it, it's dropped. The other five rules need a human call, so it
never touches them.

## Pre-commit hook

In `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/theadamdanielsson/overllm
    rev: v0.6.0
    hooks:
      - id: overllm
```

## GitHub Action

overllm ships an Action that scans a pull request and leaves one grounded comment. It stays silent when there is nothing to say.

```yaml
name: overllm
on:
  pull_request:

permissions:
  contents: read
  pull-requests: write

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: theadamdanielsson/overllm@v1
        with:
          paths: "."
```

## Other output formats

```bash
overllm --format json .      # machine-readable
overllm --format sarif .     # upload to GitHub code scanning
overllm --format github .    # GitHub Actions inline annotations
overllm --format markdown .  # the PR-comment body
```

## GitHub code scanning (SARIF)

overllm can output SARIF, so findings show up in the Security tab and inline on
the diff. It's free on public repos:

```yaml
name: overllm-scan
on: [push, pull_request]
permissions:
  contents: read
  security-events: write   # required to upload results
jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pipx run overllm . --format sarif --exit-zero > overllm.sarif
      - uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: overllm.sarif
          category: overllm
```

## Use it from an agent (MCP server)

overllm ships an MCP server, so an AI agent (Claude Desktop, Cursor, or anything that speaks MCP) can call it to audit code for needless model calls. It exposes two tools: `scan_path` (a file or directory on disk) and `scan_code` (a snippet passed inline). Both return the same findings the CLI does — the rule, the location, and the concrete replacement — and neither writes or changes anything.

Install the server variant and point your client at it (the server needs Python 3.10+; the linter itself still runs on 3.9):

```bash
pip install "overllm[mcp]"
```

```json
{
  "mcpServers": {
    "overllm": {
      "command": "uvx",
      "args": ["--from", "overllm[mcp]", "overllm-mcp"]
    }
  }
}
```

Then ask the agent things like "scan this repo for unnecessary LLM calls" or "is this function wasting a model call?" and it gets grounded, deterministic findings instead of guessing.

## Why not just use an AI code reviewer?

AI reviewers and AI-slop linters look at the code the model produced: comments, dead code, structure. None of them ask the question overllm asks, which is whether you needed the model at all. It is a different axis, and it is one plain static analysis can answer with high precision and zero cost.

## Contributing

The most useful thing you can send is a false positive: a real line of code where overllm flags a call it should not. A linter is only worth running if it is right, so one concrete bad flag is worth more than a feature request. [CONTRIBUTING.md](CONTRIBUTING.md) covers how to report one and how to run the tests.

Past releases and what changed are in [CHANGELOG.md](CHANGELOG.md).

## License

MIT © Adam Danielsson
