# Contributing

overllm is a linter, so its whole value is being right. A finding you can't trust is
worse than no finding — you learn to ignore the tool, then you uninstall it. That shapes
what is useful to contribute.

## The most valuable thing you can send: a false positive

If overllm flagged a line it should not have, that is the bug that matters most. To report
one, [open an issue](https://github.com/theadamdanielsson/overllm/issues) with:

- the smallest snippet that still gets flagged (ideally something runnable),
- the rule id it fired (e.g. `llm-in-loop`),
- what you expected instead and why the call is not actually waste,
- `overllm --version` and whether it is Python or JavaScript/TypeScript.

A single real-world case where the tool is wrong is worth more than a feature request. Most
of the hardening in the changelog came from exactly these — see the bounded match windows in
`rules.py`, each of which exists because a loose pattern fired on real code.

A missed call (a false negative) is welcome too, same format — but precision comes first. It
is better to miss an exotic wrapper than to false-positive on an unrelated `.create()`.

## Developing

Python 3.9+; the core has no runtime dependencies. Install with the dev extra (adds pytest
and tree-sitter for the JS/TS tests):

```bash
git clone https://github.com/theadamdanielsson/overllm
cd overllm
pip install -e ".[dev]"
pytest -q
```

CI runs the suite on Python 3.9, 3.11, and 3.13, and dogfoods `overllm src/` — the tool must
be clean on its own source. Run both before you open a PR.

## Where things live

| File | What it does |
| --- | --- |
| `src/overllm/detector.py` | Finds LLM calls in a Python AST and extracts the prompt (variable resolution, taint, loop classification). |
| `src/overllm/jsdetector.py` | The same, for JavaScript and TypeScript via tree-sitter. |
| `src/overllm/rules.py` | Turns a detected call into findings. This is where the rule patterns live. |
| `src/overllm/config.py` | Loads `[tool.overllm]` / `.overllm.toml`. |
| `scripts/scan_corpus.py` | Runs overllm across real GitHub repos to measure precision. |

## Adding or changing a rule

The bar is deliberately high, because a noisy rule costs more trust than it earns.

- Fire on an observable code pattern, never on taste. No generic complexity, dead code, or
  style — other tools own that ground.
- Every finding names a concrete deterministic replacement (`use sorted()`), not a vague
  "consider not doing this."
- Stay silent when unsure. Bound your patterns so the verb and the datum have to be close;
  a loose `.*` will find a false positive on real code.
- Add two tests: one snippet that should fire, and a near-miss that must stay silent. The
  second is the one that keeps the rule honest.

Before proposing a rule, sanity-check it against real code:

```bash
python scripts/scan_corpus.py --lang py --limit 80 --workdir /tmp/overllm-corpus
```

It clones repos that call LLM SDKs, runs the rules, and prints a sample to spot-check for
false positives. If your rule lights up on code that is fine, it is not ready.

## Pull requests

Keep them focused — one rule or one fix per PR. Say what real code motivated the change, and
make sure `pytest -q` and `overllm src/` are both clean.
