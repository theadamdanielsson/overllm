"""Output formatters: human, json, sarif, markdown (for the PR comment)."""

from __future__ import annotations

import json
from collections import defaultdict

from . import __version__
from .models import Finding

_COLORS = {
    "reset": "\033[0m",
    "dim": "\033[2m",
    "bold": "\033[1m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "cyan": "\033[36m",
    "green": "\033[32m",
}

_SEV_COLOR = {"error": "red", "warning": "yellow", "info": "dim"}

_RULE_HELP = "https://github.com/theadamdanielsson/overllm#rules"


def _c(name: str, use_color: bool) -> str:
    return _COLORS[name] if use_color else ""


def format_human(findings: list[Finding], use_color: bool = True) -> str:
    if not findings:
        return f"{_c('green', use_color)}overllm: no needless LLM calls found.{_c('reset', use_color)}"
    lines: list[str] = []
    for f in findings:
        loc = f"{f.path}:{f.line}:{f.col + 1}"
        sev = _c(_SEV_COLOR.get(f.severity, "yellow"), use_color)
        lines.append(
            f"{_c('bold', use_color)}{loc}{_c('reset', use_color)} "
            f"{sev}{f.severity}{_c('reset', use_color)} "
            f"{_c('yellow', use_color)}{f.rule}{_c('reset', use_color)}  {f.message}"
        )
        if f.snippet:
            lines.append(f"    {_c('dim', use_color)}{f.snippet}{_c('reset', use_color)}")
        if f.suggestion:
            lines.append(f"    {_c('cyan', use_color)}-> {f.suggestion}{_c('reset', use_color)}")
        lines.append("")
    n = len(findings)
    files = len({f.path for f in findings})
    lines.append(
        f"{_c('bold', use_color)}{n} needless LLM call{'s' if n != 1 else ''} "
        f"in {files} file{'s' if files != 1 else ''}.{_c('reset', use_color)}"
    )
    return "\n".join(lines)


def format_json(findings: list[Finding]) -> str:
    payload = [
        {
            "path": f.path,
            "line": f.line,
            "col": f.col,
            "rule": f.rule,
            "message": f.message,
            "suggestion": f.suggestion,
            "severity": f.severity,
            "snippet": f.snippet,
        }
        for f in findings
    ]
    return json.dumps(payload, indent=2)


def format_sarif(findings: list[Finding]) -> str:
    rule_ids = sorted({f.rule for f in findings})
    rules = [
        {
            "id": rid,
            "name": rid,
            "shortDescription": {"text": f"overllm {rid}"},
            "helpUri": _RULE_HELP,
        }
        for rid in rule_ids
    ]
    results = [
        {
            "ruleId": f.rule,
            "level": {"error": "error", "warning": "warning", "info": "note"}.get(f.severity, "warning"),
            "message": {"text": f"{f.message}. {f.suggestion}".strip()},
            "locations": [
                {
                    "physicalLocation": {
                        "artifactLocation": {"uri": f.path},
                        "region": {"startLine": max(f.line, 1), "startColumn": f.col + 1},
                    }
                }
            ],
        }
        for f in findings
    ]
    sarif = {
        "version": "2.1.0",
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "overllm",
                        "version": __version__,
                        "informationUri": "https://github.com/theadamdanielsson/overllm",
                        "rules": rules,
                    }
                },
                "results": results,
            }
        ],
    }
    return json.dumps(sarif, indent=2)


_MARKER = "<!-- overllm-report -->"


def format_markdown(findings: list[Finding]) -> str:
    """One grounded PR comment. Humble, cites each line, silent posting handled upstream."""
    if not findings:
        return ""
    n = len(findings)
    by_file: dict[str, list[Finding]] = defaultdict(list)
    for f in findings:
        by_file[f.path].append(f)

    out = [
        _MARKER,
        f"### \U0001f9f9 overllm found {n} LLM call{'s' if n != 1 else ''} that plain code could handle",
        "",
        "Each of these calls an AI model to do something a library or a few lines of code do "
        "faster, cheaper, and deterministically. Worth a look, not gospel - "
        "add `# overllm: ignore` on a line to silence a false positive.",
        "",
    ]
    for path in sorted(by_file):
        out.append(f"**`{path}`**")
        for f in by_file[path]:
            out.append(f"- `L{f.line}` **{f.rule}** - {f.message}.")
            if f.suggestion:
                out.append(f"  -> {f.suggestion}")
        out.append("")
    out.append(
        "<sub>Flagged by [overllm](https://github.com/theadamdanielsson/overllm) - "
        "it only reports calls where deterministic code wins, and stays silent otherwise.</sub>"
    )
    return "\n".join(out)


def render(findings: list[Finding], fmt: str, use_color: bool = True) -> str:
    if fmt == "json":
        return format_json(findings)
    if fmt == "sarif":
        return format_sarif(findings)
    if fmt == "markdown":
        return format_markdown(findings)
    return format_human(findings, use_color=use_color)
