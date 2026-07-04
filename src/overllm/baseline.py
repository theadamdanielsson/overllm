"""Baseline / ratchet mode: adopt overllm on an existing codebase without
drowning in pre-existing findings.

Snapshot the current findings to a file (`--write-baseline`), commit it, then
gate CI on `--baseline` so only NEW findings are reported. The fingerprint is
content-based (rule + path + the whole call text + model) and deliberately
excludes the line number, so it survives edits elsewhere in the file. It counts
occurrences rather than netting them, so a new wasteful call still surfaces when
an old one is deleted -- UNLESS the new call is byte-identical to the deleted one
(two identical calls in a file are indistinguishable by content, so the count
can't tell them apart). That is the one inherent blind spot of a
line-independent baseline.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from .models import Finding

BASELINE_VERSION = 1
DEFAULT_BASELINE_FILE = "overllm-baseline.json"

_WS = re.compile(r"\s+")


def _norm_snippet(snippet: str) -> str:
    return _WS.sub(" ", snippet).strip()


def fingerprint(f: Finding) -> str:
    """Stable, location-fuzzy identity of a finding.

    Keys on rule + normalized path + normalized snippet + model. Excludes
    line/col (survives insertions/deletions elsewhere) but includes the model id
    (which the snippet's first line often omits), so two deprecated-model
    findings on different models never collide.
    """
    parts = [
        f.rule,
        Path(f.path).as_posix(),
        _norm_snippet(f.snippet),
        f.model or "",
    ]
    return hashlib.sha1("\0".join(parts).encode("utf-8")).hexdigest()[:12]


def build(findings: list[Finding]) -> dict:
    """Fingerprint -> {rule, path, snippet, count} for a set of findings."""
    counts = Counter(fingerprint(f) for f in findings)
    first: dict[str, Finding] = {}
    for f in findings:
        first.setdefault(fingerprint(f), f)
    entries = {}
    for fp, f in first.items():
        entries[fp] = {
            "rule": f.rule,
            "path": Path(f.path).as_posix(),
            "snippet": _norm_snippet(f.snippet),
            "count": counts[fp],
        }
    return entries


def to_document(findings: list[Finding], tool_version: str) -> dict:
    return {
        "version": BASELINE_VERSION,
        "tool": f"overllm {tool_version}",
        "findings": dict(sorted(build(findings).items())),  # sorted -> clean diffs
    }


def write(path: Path, findings: list[Finding], tool_version: str) -> int:
    doc = to_document(findings, tool_version)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return len(doc["findings"])


def load(path: Path) -> dict:
    """Return the {fingerprint: entry} map, or {} if the file is missing/bad."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    entries = doc.get("findings", {})
    return entries if isinstance(entries, dict) else {}


def filter_new(findings: list[Finding], baseline: dict) -> list[Finding]:
    """Drop findings recorded in the baseline; keep only new ones.

    Per-fingerprint counting: if the baseline allows N occurrences of a
    fingerprint, the first N current occurrences are suppressed and any excess is
    reported (a real regression). A new call at a different site fires as long as
    its full call text differs from the baselined ones; a byte-identical new call
    is the inherent blind spot noted in the module docstring.
    """
    seen: Counter = Counter()
    kept: list[Finding] = []
    for f in findings:
        fp = fingerprint(f)
        allowed = baseline.get(fp, {}).get("count", 0)
        if seen[fp] < allowed:
            seen[fp] += 1
            continue
        kept.append(f)
    return kept


def write_pruned(path: Path, findings: list[Finding], baseline: dict, tool_version: str) -> int:
    """Ratchet the baseline file in place and return the number of entries kept."""
    kept = prune(findings, baseline)
    doc = {"version": BASELINE_VERSION, "tool": f"overllm {tool_version}", "findings": kept}
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return len(kept)


def prune(findings: list[Finding], baseline: dict) -> dict:
    """Ratchet: keep only baseline entries that still fire, tightening the count
    to what is currently present. Never adds new fingerprints (new findings must
    surface, not be auto-absorbed). This is the `--update-baseline` behavior.
    """
    current = Counter(fingerprint(f) for f in findings)
    out: dict = {}
    for fp, entry in baseline.items():
        if fp in current:
            out[fp] = {**entry, "count": min(entry.get("count", 1), current[fp])}
    return dict(sorted(out.items()))
