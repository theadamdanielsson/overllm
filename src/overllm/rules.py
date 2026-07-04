"""The rule set. Every rule keys off an observable code pattern, not taste, and
every finding names a concrete deterministic replacement.

The core question is: LLM calls you did not need (mechanical/extraction/static),
should not make as written (in-loop cost, prompt-injection), or configured wrong
in a code-visible way (a retired model id, a sampling param a model rejects). All
of it is deterministic and provable from the source. It deliberately does NOT do
generic complexity, dead code, style, or "is this model too big for the task" --
that last one needs judgement, which is where a precise linter turns into noise.
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
DEPRECATED_MODEL = "deprecated-model"
UNSUPPORTED_PARAMS = "unsupported-params"
JSON_MODE_MISSING = "json-mode-missing-json"

ALL_RULES = (
    STATIC_PROMPT, LLM_EXTRACTION, LLM_IN_LOOP, LLM_MECHANICAL, PROMPT_INJECTION,
    DEPRECATED_MODEL, UNSUPPORTED_PARAMS, JSON_MODE_MISSING,
)

# error   = clearly unnecessary call, a broken call, or a security risk (raise by default)
# warning = a real cost pattern worth reviewing (raise by default)
# info    = a minor smell, mostly demo/tutorial code (quiet unless asked for)
RULE_SEVERITY = {
    LLM_EXTRACTION: "error",
    LLM_MECHANICAL: "error",
    PROMPT_INJECTION: "error",
    LLM_IN_LOOP: "warning",
    STATIC_PROMPT: "info",
    DEPRECATED_MODEL: "warning",   # per-finding: error when retired, warning when deprecated
    UNSUPPORTED_PARAMS: "warning",
    JSON_MODE_MISSING: "error",     # a provable OpenAI 400
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

# --- Model hygiene -----------------------------------------------------------
# Deterministic facts about the call: the model id is a known-old string, or a
# sampling parameter is set on a model that rejects it. Exact-match on the model
# id (lowercased), so live aliases -- davinci-002, claude-3-haiku, gpt-4o -- never
# trip. Sourced from provider deprecation pages; this list needs periodic updating.

# Retired models: the call 404s today.
_RETIRED_MODELS = {
    "claude-1": "claude-sonnet-4-6", "claude-1.0": "claude-sonnet-4-6",
    "claude-1.2": "claude-sonnet-4-6", "claude-1.3": "claude-sonnet-4-6",
    "claude-instant-1": "claude-haiku-4-5", "claude-instant-1.1": "claude-haiku-4-5",
    "claude-instant-1.2": "claude-haiku-4-5",
    "claude-2": "claude-sonnet-4-6", "claude-2.0": "claude-sonnet-4-6",
    "claude-2.1": "claude-sonnet-4-6",
    "claude-3-sonnet-20240229": "claude-sonnet-4-6",
    "claude-3-5-sonnet-20240620": "claude-sonnet-4-6",
    "claude-3-5-sonnet-20241022": "claude-sonnet-4-6",
    "claude-3-opus-20240229": "claude-opus-4-8",
    "claude-3-5-haiku-20241022": "claude-haiku-4-5",
    "claude-3-7-sonnet-20250219": "claude-sonnet-4-6",
    "text-davinci-003": "gpt-4o-mini", "text-davinci-002": "gpt-4o-mini",
    "text-davinci-001": "gpt-4o-mini", "text-curie-001": "gpt-4o-mini",
    "text-babbage-001": "gpt-4o-mini", "text-ada-001": "gpt-4o-mini",
    "davinci": "gpt-4o-mini", "curie": "gpt-4o-mini",
    "babbage": "gpt-4o-mini", "ada": "gpt-4o-mini",
    "code-davinci-002": "gpt-4o-mini", "code-cushman-001": "gpt-4o-mini",
    "gpt-3.5-turbo-0301": "gpt-4o-mini", "gpt-4-0314": "gpt-4o", "gpt-4-32k-0314": "gpt-4o",
}
# Deprecated models: still served, scheduled for removal.
_DEPRECATED_MODELS = {
    "claude-3-haiku-20240307": "claude-haiku-4-5",
    "claude-opus-4-20250514": "claude-opus-4-8",
    "claude-sonnet-4-20250514": "claude-sonnet-4-6",
    "claude-opus-4-1-20250805": "claude-opus-4-8",
    "gpt-3.5-turbo-0613": "gpt-4o-mini", "gpt-3.5-turbo-16k-0613": "gpt-4o-mini",
    "gpt-4-32k": "gpt-4o", "gpt-4-32k-0613": "gpt-4o",
    "gpt-4-vision-preview": "gpt-4o", "gpt-4-1106-vision-preview": "gpt-4o",
}

# Models that reject sampling params: a 400 on the newest Anthropic models, and on
# the OpenAI reasoning (o-) series, which only accept the default temperature.
_NO_SAMPLING_MODELS = {
    "claude-opus-4-8", "claude-opus-4-7", "claude-fable-5", "claude-mythos-5",
}
_OPENAI_REASONING_RE = re.compile(r"^o[1-9][0-9]?(?:-|$)")
_SAMPLING_PARAMS = ("temperature", "top_p", "top_k")


def _rejects_sampling(model: str) -> bool:
    return model in _NO_SAMPLING_MODELS or bool(_OPENAI_REASONING_RE.match(model))


def _finding(call: LLMCall, path: str, rule: str, message: str, suggestion: str,
             severity: str | None = None) -> Finding:
    return Finding(
        path=path,
        line=call.line,
        col=call.col,
        rule=rule,
        message=message,
        suggestion=suggestion,
        severity=severity or RULE_SEVERITY.get(rule, "warning"),
        snippet=call.snippet,
        model=call.model,
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
        if call.loop_concurrent:
            # dispatched via asyncio.gather/as_completed: cost still scales N x, but
            # latency is already amortized, so do not claim N latencies.
            msg = ("LLM call dispatched once per item into a concurrent gather: N calls, "
                   "so N times the token cost (latency is amortized by the concurrency)")
            sug = ("batch the items into one call or cache repeated results to cut token "
                   "cost; the concurrency already handles latency")
        else:
            msg = ("LLM call inside a per-item loop: one API round-trip per iteration "
                   "(N calls, N latencies, N times the cost)")
            sug = ("batch the items into a single call, cache repeated results, or if the "
                   "per-item work is deterministic use a function")
        out.append(_finding(call, path, LLM_IN_LOOP, msg, sug))

    # R5: untrusted external input flows straight into the prompt.
    if call.tainted:
        out.append(_finding(
            call, path, PROMPT_INJECTION,
            "untrusted web-request input flows straight into this prompt, a prompt-injection risk",
            "keep external input in a separate user message (never the system prompt), validate it, and constrain what the model is allowed to do",
        ))

    # R8: JSON mode requested but the fully-static prompt never contains "json".
    # OpenAI rejects response_format=json_object unless a message says "json" -- a
    # guaranteed 400. Only fires when every message is statically readable, so the
    # absence of the word is proven, not merely unobserved.
    if call.json_object_mode and call.all_text_static and "json" not in call.all_text:
        out.append(_finding(
            call, path, JSON_MODE_MISSING,
            'requests JSON mode (response_format={"type": "json_object"}) but no message contains '
            'the word "json" -- OpenAI rejects this with a 400',
            'add "json" to the system or user message, or use a json_schema response_format',
            severity="error",
        ))

    # R6/R7: model hygiene -- a retired/deprecated model id, or a sampling
    # parameter set on a model that rejects it. Both key off the literal model
    # string, so they stay silent when the model is not a plain string.
    model = (call.model or "").lower()
    if model:
        if model in _RETIRED_MODELS:
            out.append(_finding(
                call, path, DEPRECATED_MODEL,
                f"calls \"{call.model}\", a retired model that no longer exists (this request will 404)",
                f"switch to `{_RETIRED_MODELS[model]}`",
                severity="error",
            ))
        elif model in _DEPRECATED_MODELS:
            out.append(_finding(
                call, path, DEPRECATED_MODEL,
                f"calls \"{call.model}\", a deprecated model scheduled for retirement",
                f"switch to `{_DEPRECATED_MODELS[model]}`",
                severity="warning",
            ))
        if _rejects_sampling(model):
            bad = [p for p in _SAMPLING_PARAMS if p in call.params]
            if bad:
                out.append(_finding(
                    call, path, UNSUPPORTED_PARAMS,
                    f"sets {', '.join(bad)} on \"{call.model}\", which does not accept sampling "
                    "parameters -- reasoning/thinking models reject them (a 400), so the setting is a no-op or an error",
                    "remove the parameter; steer these models with the prompt (and effort/reasoning depth) instead",
                ))

    return out
