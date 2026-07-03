"""Core data types shared across overllm."""

from __future__ import annotations

from dataclasses import dataclass

SEVERITIES = ("error", "warning", "info")
SEVERITY_RANK = {"error": 3, "warning": 2, "info": 1}


@dataclass(frozen=True)
class Finding:
    """One flagged LLM call.

    A finding always names a concrete deterministic replacement in `suggestion`.
    The rule fires only on an observable code pattern, never on taste.
    """

    path: str
    line: int
    col: int
    rule: str
    message: str
    suggestion: str = ""
    severity: str = "warning"
    snippet: str = ""

    @property
    def key(self) -> tuple:
        return (self.path, self.line, self.col, self.rule)
