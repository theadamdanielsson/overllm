"""Walk files, parse them, run rules, and apply suppression + config."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from .config import DEFAULT_EXCLUDES, Config
from .detector import find_llm_calls
from .models import Finding
from .rules import run_rules

# `# overllm: ignore` or `# overllm: ignore=rule-a,rule-b`
_IGNORE_RE = re.compile(r"#\s*overllm:\s*ignore(?:=([\w\-,\s]+))?")
_IGNORE_FILE_RE = re.compile(r"#\s*overllm:\s*ignore-file\b")


def iter_python_files(paths: list[str], excludes: tuple[str, ...]) -> list[Path]:
    all_excludes = set(DEFAULT_EXCLUDES) | set(excludes)

    def excluded(p: Path) -> bool:
        parts = set(p.parts)
        if parts & set(DEFAULT_EXCLUDES):
            return True
        s = str(p)
        return any(ex and ex in s for ex in excludes)

    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_file():
            if p.suffix == ".py" and not excluded(p):
                out.append(p)
        elif p.is_dir():
            for f in sorted(p.rglob("*.py")):
                if not excluded(f):
                    out.append(f)
    # dedupe, keep order
    seen: set[str] = set()
    uniq: list[Path] = []
    for f in out:
        r = str(f.resolve())
        if r not in seen:
            seen.add(r)
            uniq.append(f)
    return uniq


def _line_directives(lines: list[str]) -> tuple[bool, dict[int, set[str] | None]]:
    """Return (ignore_whole_file, {line_no: None-for-all | set-of-rule-ids})."""
    per_line: dict[int, set[str] | None] = {}
    ignore_file = False
    for i, line in enumerate(lines, start=1):
        if _IGNORE_FILE_RE.search(line):
            ignore_file = True
        m = _IGNORE_RE.search(line)
        if m:
            raw = m.group(1)
            if raw:
                per_line[i] = {r.strip() for r in raw.split(",") if r.strip()}
            else:
                per_line[i] = None  # ignore all rules on this line
    return ignore_file, per_line


def _suppressed(finding: Finding, node_lines: range, per_line: dict[int, set[str] | None]) -> bool:
    # a directive on any physical line of the call (or the line above it) suppresses it
    candidate_lines = set(node_lines) | {finding.line, finding.line - 1}
    for ln in candidate_lines:
        if ln in per_line:
            rules = per_line[ln]
            if rules is None or finding.rule in rules:
                return True
    return False


def analyze_file(path: Path, config: Config) -> list[Finding]:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []  # not our job to report parse errors; stay quiet

    lines = source.splitlines()
    ignore_file, per_line = _line_directives(lines)
    if ignore_file:
        return []

    calls = find_llm_calls(tree, lines)
    findings: list[Finding] = []
    seen: set[tuple] = set()
    display_path = str(path)
    for call in calls:
        node_start = getattr(call.node, "lineno", call.line)
        node_end = getattr(call.node, "end_lineno", node_start) or node_start
        node_lines = range(node_start, node_end + 1)
        for f in run_rules(call, display_path):
            if not config.enabled(f.rule):
                continue
            if f.key in seen:
                continue
            if _suppressed(f, node_lines, per_line):
                continue
            seen.add(f.key)
            findings.append(f)
    return findings


def analyze_paths(paths: list[str], config: Config) -> list[Finding]:
    findings: list[Finding] = []
    for f in iter_python_files(paths, config.exclude):
        findings.extend(analyze_file(f, config))
    findings.sort(key=lambda x: (x.path, x.line, x.col, x.rule))
    return findings
