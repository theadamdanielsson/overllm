# overllm

**Catch the LLM/AI calls you didn't need.**

overllm is a small, fast linter with one job: find the places in your code where you call an AI model to do something plain code does better. You called GPT to parse a date. You called a model to extract JSON that `json.loads` already handles. You are paying latency, money, and nondeterminism for a regex.

It reads your code with Python's own `ast` module. No model runs, no network, no API key. Same code in, same result out. Fast enough for a pre-commit hook.

Everyone else lints the code the AI wrote. overllm catches where you are paying an AI to do what a library already does.

## Install

```bash
pip install overllm
```

## Use it

```bash
overllm .            # scan the current project
overllm src/         # scan a folder
overllm app.py       # scan one file
```

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

overllm exits non-zero when it finds something, so it gates a commit or a CI check. Pass `--exit-zero` to report without failing.

## Rules

Every rule fires only on a concrete code pattern, and every finding names the deterministic replacement. It stays silent when it is not sure.

| Rule | Fires when | Suggests |
| --- | --- | --- |
| `static-prompt` | The user prompt is a compile-time constant (no variables). The input is fixed, so the call buys nothing. | precompute or cache the result |
| `llm-extraction` | The prompt asks the model to extract or parse an email, URL, date, number, or JSON. | a regex, `json`, `datetime`, or the SDK's structured-output mode |
| `llm-mechanical` | The prompt asks for a mechanical transform: sort, reverse, count, sum, deduplicate, change case, base64, arithmetic on literals. | the one-line stdlib equivalent |
| `llm-in-loop` | An LLM call sits inside a `for`/`async for`/comprehension. One API round-trip per iteration. | batch, cache, or move it out of the loop |

It detects calls to the OpenAI, Anthropic, Google, Mistral, Cohere, Groq, LangChain, LiteLLM, and Ollama SDKs, and raw HTTP requests to those hosts.

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
```

Or on the command line: `--select`, `--ignore`, `--exclude` via config, `--config PATH`.

## Pre-commit hook

In `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/theadamdanielsson/overllm
    rev: v0.1.0
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
overllm --format markdown .  # the PR-comment body
```

## Why not just use an AI code reviewer?

AI reviewers and AI-slop linters look at the code the model produced: comments, dead code, structure. None of them ask the question overllm asks, which is whether you needed the model at all. It is a different axis, and it is one plain static analysis can answer with high precision and zero cost.

## License

MIT © Adam Danielsson
