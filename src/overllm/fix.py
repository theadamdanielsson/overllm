"""Safe autofix for the two mechanically-fixable rules.

Only two findings have an exact, code-visible replacement:

  * unsupported-params -- a sampling param on a model that rejects it -> remove
    it. SAFE: the call currently 400s or the param is a no-op, so removing it
    cannot break a working call.
  * deprecated-model -- a retired/deprecated model id -> its mapped replacement.
    UNSAFE: a model swap changes runtime behavior, so it is gated behind
    include_unsafe (the CLI's --unsafe-fixes).

The five judgment rules (mechanical, extraction, in-loop, static, injection) are
never auto-applied. Edits are anchored to AST node spans and applied as minimal
byte-range replacements -- never a regex on source, so nothing inside a string or
comment is touched -- sorted descending and non-overlapping, and the result is
re-parsed so we never write a file that no longer compiles.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from .detector import find_llm_calls
from .rules import _DEPRECATED_MODELS, _RETIRED_MODELS, _SAMPLING_PARAMS, _rejects_sampling


@dataclass
class Edit:
    start: int
    end: int
    replacement: str
    rule: str
    unsafe: bool


def _line_starts(source: str) -> list[int]:
    starts = [0]
    for line in source.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    return starts


def _pos(starts: list[int], line: int, col: int) -> int:
    return starts[line - 1] + col


def _kw_node(call: ast.Call, name: str) -> ast.keyword | None:
    for k in call.keywords:
        if k.arg == name:
            return k
    return None


def _remove_kwarg_edit(source: str, starts: list[int], kw: ast.keyword, rule: str) -> Edit | None:
    if kw.end_lineno is None:
        return None
    start = _pos(starts, kw.lineno, kw.col_offset)
    end = _pos(starts, kw.end_lineno, kw.end_col_offset)
    # Prefer eating a trailing comma (+ surrounding space); if there is none this
    # was the last argument, so eat a leading comma instead. Either way we leave
    # exactly the right number of separators and never create a dangling comma.
    j = end
    while j < len(source) and source[j] in " \t":
        j += 1
    if j < len(source) and source[j] == ",":
        j += 1
        while j < len(source) and source[j] in " \t":
            j += 1
        return Edit(start, j, "", rule, False)
    i = start
    while i > 0 and source[i - 1] in " \t":
        i -= 1
    if i > 0 and source[i - 1] == ",":
        return Edit(i - 1, end, "", rule, False)
    return Edit(start, end, "", rule, False)  # a lone keyword


def _replace_model_edit(source: str, starts: list[int], value: ast.expr, new_id: str, rule: str) -> Edit | None:
    if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
        return None
    start = _pos(starts, value.lineno, value.col_offset)
    end = _pos(starts, value.end_lineno, value.end_col_offset)
    q = source[start]
    if q not in ("'", '"'):
        return None  # f-string / concatenation / prefixed literal -> leave it
    return Edit(start, end, f"{q}{new_id}{q}", rule, True)


def compute_edits(source: str, tree: ast.AST | None = None, include_unsafe: bool = False) -> list[Edit]:
    if tree is None:
        tree = ast.parse(source)
    lines = source.splitlines()
    starts = _line_starts(source)
    edits: list[Edit] = []
    for call in find_llm_calls(tree, lines):
        model = (call.model or "").lower()
        if _rejects_sampling(model):
            for param in _SAMPLING_PARAMS:
                kw = _kw_node(call.node, param)
                if kw is not None:
                    e = _remove_kwarg_edit(source, starts, kw, "unsupported-params")
                    if e is not None:
                        edits.append(e)
        if include_unsafe and model:
            new_id = _RETIRED_MODELS.get(model) or _DEPRECATED_MODELS.get(model)
            kw = _kw_node(call.node, "model")
            if new_id and kw is not None:
                e = _replace_model_edit(source, starts, kw.value, new_id, "deprecated-model")
                if e is not None:
                    edits.append(e)
    return edits


def apply_edits(source: str, edits: list[Edit]) -> tuple[str, list[Edit]]:
    """Apply non-overlapping edits, highest offset first so earlier edits do not
    shift later ones. Overlapping edits (rare: two removals sharing a comma) are
    skipped this pass rather than risk corruption."""
    chosen: list[Edit] = []
    boundary = len(source) + 1
    for e in sorted(edits, key=lambda x: x.start, reverse=True):
        if e.end <= boundary:
            chosen.append(e)
            boundary = e.start
    out = source
    for e in chosen:
        out = out[: e.start] + e.replacement + out[e.end:]
    return out, chosen


def fix_source(source: str, include_unsafe: bool = False) -> tuple[str, list[Edit]]:
    """Return (new_source, applied_edits). Unchanged source and [] when there is
    nothing safe to do or the result would not parse."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source, []
    edits = compute_edits(source, tree, include_unsafe)
    if not edits:
        return source, []
    new, applied = apply_edits(source, edits)
    if new == source:
        return source, []
    try:
        ast.parse(new)
    except SyntaxError:
        return source, []  # never write something that no longer compiles
    return new, applied
