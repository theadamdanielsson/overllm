"""Safe autofix for the two mechanically-fixable rules.

Only two findings have an exact, code-visible replacement:

  * unsupported-params -- a sampling param on a model that rejects it -> remove
    it. SAFE: the call currently 400s or the param is a no-op, so removing it
    cannot break a working call.
  * deprecated-model -- a retired/deprecated model id -> its mapped replacement.
    UNSAFE: a model swap changes runtime behavior, so it is gated behind
    include_unsafe (the CLI's --unsafe-fixes).

The five judgment rules are never auto-applied. Edits are anchored to AST node
spans and applied as minimal replacements -- never a regex on source, so nothing
inside a string or comment is touched -- sorted descending and non-overlapping,
and the result is re-parsed so we never write a file that no longer compiles.

Offsets: CPython's `col_offset`/`end_col_offset` are UTF-8 BYTE offsets, so all
edit math is done on `source.encode("utf-8")` and line starts are split on real
Python newlines only (`\\r\\n | \\r | \\n`). Doing this in character space would
shift every edit past any non-ASCII byte earlier on the line -- silent corruption.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from .detector import find_llm_calls
from .rules import _DEPRECATED_MODELS, _RETIRED_MODELS, _SAMPLING_PARAMS, _rejects_sampling

_LINE_BOUNDARY = re.compile(rb"\r\n|\r|\n")


@dataclass
class Edit:
    start: int          # byte offset
    end: int            # byte offset
    replacement: bytes
    rule: str
    unsafe: bool


def _byte_line_starts(data: bytes) -> list[int]:
    """Byte offset of the start of each 1-based line, splitting only on the
    newline sequences the Python tokenizer counts (not form feed / U+2028 / etc.)."""
    starts = [0]
    for m in _LINE_BOUNDARY.finditer(data):
        starts.append(m.end())
    return starts


def _bpos(starts: list[int], line: int, col: int) -> int | None:
    if not (1 <= line <= len(starts)):
        return None
    return starts[line - 1] + col


def _kw_node(call: ast.Call, name: str) -> ast.keyword | None:
    for k in call.keywords:  # top-level kwargs only; never a nested call's
        if k.arg == name:
            return k
    return None


def _remove_kwarg_edit(data: bytes, starts: list[int], kw: ast.keyword, rule: str) -> Edit | None:
    if kw.end_lineno is None:
        return None
    start = _bpos(starts, kw.lineno, kw.col_offset)
    end = _bpos(starts, kw.end_lineno, kw.end_col_offset)
    if start is None or end is None or not (0 <= start <= end <= len(data)):
        return None
    # Prefer eating a trailing comma (+ surrounding space); if there is none this
    # was the last argument, so eat a leading comma instead -- never a dangle.
    j = end
    while j < len(data) and data[j:j + 1] in (b" ", b"\t"):
        j += 1
    if j < len(data) and data[j:j + 1] == b",":
        j += 1
        while j < len(data) and data[j:j + 1] in (b" ", b"\t"):
            j += 1
        return Edit(start, j, b"", rule, False)
    i = start
    while i > 0 and data[i - 1:i] in (b" ", b"\t"):
        i -= 1
    if i > 0 and data[i - 1:i] == b",":
        return Edit(i - 1, end, b"", rule, False)
    return Edit(start, end, b"", rule, False)  # a lone keyword


def _replace_model_edit(data: bytes, starts: list[int], value: ast.expr, new_id: str, rule: str) -> Edit | None:
    if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
        return None
    start = _bpos(starts, value.lineno, value.col_offset)
    end = _bpos(starts, value.end_lineno, value.end_col_offset)
    if start is None or end is None or not (0 <= start < end <= len(data)):
        return None
    q = data[start:start + 1]
    if q not in (b"'", b'"'):
        return None  # f-string / concatenation / prefixed literal -> leave it
    return Edit(start, end, q + new_id.encode("utf-8") + q, rule, True)


def compute_edits(source: str, tree: ast.AST | None = None, include_unsafe: bool = False) -> list[Edit]:
    if tree is None:
        tree = ast.parse(source)
    data = source.encode("utf-8")
    starts = _byte_line_starts(data)
    lines = source.splitlines()
    edits: list[Edit] = []
    for call in find_llm_calls(tree, lines):
        model = (call.model or "").lower()
        if _rejects_sampling(model):
            for param in _SAMPLING_PARAMS:
                kw = _kw_node(call.node, param)
                if kw is not None:
                    e = _remove_kwarg_edit(data, starts, kw, "unsupported-params")
                    if e is not None:
                        edits.append(e)
        if include_unsafe and model:
            new_id = _RETIRED_MODELS.get(model) or _DEPRECATED_MODELS.get(model)
            kw = _kw_node(call.node, "model")
            if new_id and kw is not None:
                e = _replace_model_edit(data, starts, kw.value, new_id, "deprecated-model")
                if e is not None:
                    edits.append(e)
    return edits


def apply_edits(data: bytes, edits: list[Edit]) -> tuple[bytes, list[Edit]]:
    """Apply non-overlapping edits, highest offset first so earlier edits do not
    shift later ones. Overlapping edits (two removals sharing a comma) are skipped
    this pass rather than risk corruption."""
    chosen: list[Edit] = []
    boundary = len(data) + 1
    for e in sorted(edits, key=lambda x: x.start, reverse=True):
        if e.end <= boundary:
            chosen.append(e)
            boundary = e.start
    out = data
    for e in chosen:
        out = out[: e.start] + e.replacement + out[e.end:]
    return out, chosen


def fix_source(source: str, include_unsafe: bool = False) -> tuple[str, list[Edit]]:
    """Return (new_source, applied_edits). Unchanged source and [] when there is
    nothing safe to do or the result would not parse."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return source, []
    try:
        edits = compute_edits(source, tree, include_unsafe)
    except Exception:
        return source, []
    if not edits:
        return source, []
    data = source.encode("utf-8")
    new_data, applied = apply_edits(data, edits)
    if new_data == data:
        return source, []
    try:
        new = new_data.decode("utf-8")
        ast.parse(new)
    except (UnicodeDecodeError, SyntaxError, ValueError, RecursionError):
        return source, []  # never write something that no longer compiles
    return new, applied
