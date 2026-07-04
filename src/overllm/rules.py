"""The rule set. Every rule keys off an observable code pattern, not taste, and
every finding names a concrete deterministic replacement.

v1 is deliberately about one thing: LLM calls you did not need. It does not do
generic complexity, dead code, or style - that ground is already covered by
other tools, and it is where an opinionated linter turns into noise.
"""

from __future__ import annotations

import re

from .detector import LLMCall
from .models import Finding

# Rule ids
STATIC_PROMPT = "static-prompt"
LLM_EXTRACTION = "llm-extraction"
LLM_IN_LOOP = "llm-in-loop"
LLM_MECHANICAL = "llm-mechanical"
PROMPT_INJECTION = "prompt-injection"

ALL_RULES = (STATIC_PROMPT, LLM_EXTRACTION, LLM_IN_LOOP, LLM_MECHANICAL, PROMPT_INJECTION)

# error   = clearly unnecessary call, or a security risk (raise by default)
# warning = a real cost pattern worth reviewing (raise by default)
# info    = a minor smell, mostly demo/tutorial code (quiet unless asked for)
RULE_SEVERITY = {
    LLM_EXTRACTION: "error",
    LLM_MECHANICAL: "error",
    PROMPT_INJECTION: "error",
    LLM_IN_LOOP: "warning",
    STATIC_PROMPT: "info",
}

# (compiled pattern, human message, suggestion)
# Bounded windows (not `.*`) so the verb and the datum have to be near each other;
# a loose `.*` matched "get the weather ... on a specific date" on a real repo.
_EXTRACTION_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"\bextract\b.{0,30}\be-?mail\b"),
     "asks the model to extract an email address",
     "match it with a regex, or use the `email-validator` package"),
    (re.compile(r"\bextract\b.{0,30}\b(url|hyperlink)\b"),
     "asks the model to extract a URL",
     "use a regex or `urllib.parse`"),
    (re.compile(r"\b(extract|parse)\b.{0,25}\b(date|datetime)\b"),
     "asks the model to extract or parse a date",
     "use `datetime.strptime` or `dateutil.parser`"),
    (re.compile(r"\bextract\b.{0,30}\b(phone number|price|amount|dollar)\b"),
     "asks the model to extract a number",
     "use a regex, then `int()` / `float()`"),
]

_MECHANICAL_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"\b(sort|order)\b.*\b(these|this|them|the (list|items|array|names|numbers))\b|\balphabetize\b"),
     "asks the model to sort",
     "use `sorted()`"),
    (re.compile(r"\breverse\b.*\b(the\s+)?(string|list|order|text|these|it)\b"),
     "asks the model to reverse a sequence",
     "use `reversed()` or `[::-1]`"),
    (re.compile(r"\bcount\b.{0,20}\b(the\s+)?(number of\s+)?(words|characters|letters|items|occurrences|elements)\b"),
     "asks the model to count",
     "use `len()`, `str.count()`, or `collections.Counter`"),
    (re.compile(r"\b(sum|add up|total|average|mean)\b.*\b(these|the)\b.*\b(numbers|values|amounts)\b|\bcalculate the (sum|total|average|mean)\b"),
     "asks the model to do arithmetic over values",
     "use `sum()` / `statistics.mean()`"),
    # "unique" must sit on a collection ("unique values/items/emails"), not stand
    # alone as an adjective -- "unique features", "a unique logo", "makes it unique"
    # are not dedup requests. Found by scanning real repos.
    (re.compile(r"\b(remove|drop|eliminate)\s+(the\s+)?duplicates?\b|\bde-?duplicate\b|"
                r"\bunique\s+(values|items|elements|entries|rows|lines|list|set|strings|numbers|ids|keys|names)\b"),
     "asks the model to deduplicate",
     "use `set()` or `dict.fromkeys()`"),
    (re.compile(r"\b(uppercase|lowercase|to upper|to lower|capitalize|title[- ]?case)\b"),
     "asks the model to change letter case",
     "use `str.upper()` / `.lower()` / `.title()`"),
    (re.compile(r"\bbase64\b"),
     "asks the model to base64 encode/decode",
     "use the `base64` module"),
    (re.compile(r"\b(what is|calculate|compute)\b[^.]*\b\d+\s*[-+*/x×]\s*\d+"),
     "asks the model to compute arithmetic on literal numbers",
     "just compute it in code"),
    (re.compile(r"\bformat\b.*\b(the\s+)?date\b|\bconvert\b.*\bdate\b"),
     "asks the model to format a date",
     "use `datetime.strftime` / `strptime`"),
]

_MIN_STATIC_LEN = 12  # ignore trivial/placeholder prompts for the static rule


def _finding(call: LLMCall, path: str, rule: str, message: str, suggestion: str) -> Finding:
    return Finding(
        path=path,
        line=call.line,
        col=call.col,
        rule=rule,
        message=message,
        suggestion=suggestion,
        severity=RULE_SEVERITY.get(rule, "warning"),
        snippet=call.snippet,
    )


def run_rules(call: LLMCall, path: str) -> list[Finding]:
    out: list[Finding] = []
    text = call.prompt_text

    # R1: fully static user prompt - constant input, so the call buys nothing.
    if (
        call.prompt_resolved
        and call.prompt_static
        and len(text) >= _MIN_STATIC_LEN
        and re.search(r"[a-z]{3,}", text)
    ):
        out.append(_finding(
            call, path, STATIC_PROMPT,
            "LLM call with a fully static prompt (no variables). The input is "
            "constant, so this pays latency, money, and nondeterminism for a fixed result",
            "precompute or cache the result; if you meant to include runtime data, interpolate it",
        ))

    # R2: the prompt asks for something a parser/regex does deterministically.
    if text:
        for pat, msg, sug in _EXTRACTION_PATTERNS:
            if pat.search(text):
                out.append(_finding(call, path, LLM_EXTRACTION, "LLM call " + msg, sug))
                break

    # R4: the prompt asks for a mechanical transform with a stdlib one-liner.
    if text:
        for pat, msg, sug in _MECHANICAL_PATTERNS:
            if pat.search(text):
                out.append(_finding(call, path, LLM_MECHANICAL, "LLM call " + msg, sug))
                break

    # R3: an LLM call in a batchable loop - one API round-trip per item, where the
    # items are independent (not a while/retry/conversation loop). Real savings.
    if call.loop_kind == "batchable":
        out.append(_finding(
            call, path, LLM_IN_LOOP,
            "LLM call inside a per-item loop: one API round-trip per iteration (N calls, N latencies, N times the cost)",
            "batch the items into a single call, cache repeated results, or if the per-item work is deterministic use a function",
        ))

    # R5: untrusted external input flows straight into the prompt.
    if call.tainted:
        out.append(_finding(
            call, path, PROMPT_INJECTION,
            "untrusted web-request input flows straight into this prompt, a prompt-injection risk",
            "keep external input in a separate user message (never the system prompt), validate it, and constrain what the model is allowed to do",
        ))

    return out
