"""Configuration loading from pyproject.toml [tool.overllm] or .overllm.toml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

try:  # tomllib is stdlib on 3.11+; config files are simply ignored below it
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None

from .models import SEVERITY_RANK
from .rules import ALL_RULES

DEFAULT_EXCLUDES = (
    ".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "env",
    "node_modules", "build", "dist", ".mypy_cache", ".pytest_cache",
    ".tox", "site-packages", ".eggs",
)


@dataclass
class Config:
    select: tuple[str, ...] = ALL_RULES
    ignore: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    min_severity: str = "warning"
    llm_calls: tuple[str, ...] = ()  # user-declared LLM wrapper names, e.g. "myapp.llm.ask"

    def enabled(self, rule: str) -> bool:
        return rule in self.select and rule not in self.ignore

    def severe_enough(self, severity: str) -> bool:
        return SEVERITY_RANK.get(severity, 2) >= SEVERITY_RANK.get(self.min_severity, 2)


def _read_table(path: Path) -> dict:
    if tomllib is None:
        return {}
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    if path.name == "pyproject.toml":
        return data.get("tool", {}).get("overllm", {}) or {}
    return data


def load_config(start: Path | None = None, explicit: Path | None = None) -> Config:
    table: dict = {}
    if explicit is not None:
        table = _read_table(explicit)
    else:
        base = (start or Path.cwd()).resolve()
        for parent in [base, *base.parents]:
            for name in (".overllm.toml", "pyproject.toml"):
                p = parent / name
                if p.is_file():
                    t = _read_table(p)
                    if t or name == ".overllm.toml":
                        table = t
                        break
            if table:
                break

    select = tuple(table.get("select", ALL_RULES))
    ignore = tuple(table.get("ignore", ()))
    exclude = tuple(table.get("exclude", ()))
    # keep only known rule ids in select, so a typo does not silently disable everything
    select = tuple(r for r in select if r in ALL_RULES) or ALL_RULES
    min_severity = table.get("min_severity", "warning")
    if min_severity not in SEVERITY_RANK:
        min_severity = "warning"
    llm_calls = tuple(str(c) for c in table.get("llm_calls", ()) if str(c).strip())
    return Config(select=select, ignore=ignore, exclude=exclude,
                  min_severity=min_severity, llm_calls=llm_calls)
