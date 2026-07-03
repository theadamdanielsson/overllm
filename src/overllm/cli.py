"""Command-line entry point for overllm."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .analyze import analyze_paths
from .config import load_config
from .report import render
from .rules import ALL_RULES


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="overllm",
        description="Catch the LLM/AI calls you didn't need. Flags LLM API calls "
        "where deterministic code is simpler, cheaper, and more reliable.",
    )
    p.add_argument("paths", nargs="*", default=["."], help="files or directories to scan (default: .)")
    p.add_argument("--format", choices=["human", "json", "sarif", "markdown"], default="human")
    p.add_argument("--select", help="comma-separated rule ids to run (default: all)")
    p.add_argument("--ignore", help="comma-separated rule ids to skip")
    p.add_argument("--config", type=Path, help="path to a config file (pyproject.toml or .overllm.toml)")
    p.add_argument("--exit-zero", action="store_true", help="always exit 0, even when findings exist")
    p.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    p.add_argument("--version", action="version", version=f"overllm {__version__}")
    return p


def _split(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(v.strip() for v in value.split(",") if v.strip())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = args.paths or ["."]

    config = load_config(explicit=args.config)
    select = _split(args.select)
    if select:
        config.select = tuple(r for r in select if r in ALL_RULES) or ALL_RULES
    ignore = _split(args.ignore)
    if ignore:
        config.ignore = tuple(set(config.ignore) | set(ignore))

    findings = analyze_paths(paths, config)

    use_color = sys.stdout.isatty() and not args.no_color
    output = render(findings, args.format, use_color=use_color)
    if output:
        print(output)

    if args.exit_zero:
        return 0
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
