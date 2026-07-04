"""Output formatters that aren't covered elsewhere: the GitHub annotation format."""

from __future__ import annotations

from overllm.models import Finding
from overllm.report import format_github


def _f(rule="llm-mechanical", path="a.py", line=5, col=3, message="asks the model to sort",
       suggestion="use sorted()", severity="error"):
    return Finding(path=path, line=line, col=col, rule=rule, message=message,
                   suggestion=suggestion, severity=severity, snippet="x")


def test_github_workflow_command_shape_and_col_is_one_based():
    out = format_github([_f()])
    assert out.startswith("::error file=a.py,line=5,col=4,")
    assert "llm-mechanical: asks the model to sort -> use sorted()" in out


def test_github_severity_maps_info_to_notice():
    out = format_github([_f(severity="info"), _f(severity="warning")])
    assert "::notice " in out and "::warning " in out


def test_github_escapes_message_newlines():
    out = format_github([_f(message="line one\nline two", suggestion="")])
    assert "\n" not in out  # single finding: the newline must be escaped
    assert "%0A" in out


def test_github_empty_findings():
    assert format_github([]) == ""
