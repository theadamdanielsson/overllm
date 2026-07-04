"""Command-line entry point for overllm."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from . import baseline as bl
from .analyze import analyze_paths
from .baseline import DEFAULT_BASELINE_FILE
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
    p.add_argument("--format", choices=["human", "json", "sarif", "markdown", "github"], default="human")
    p.add_argument("--select", help="comma-separated rule ids to run (default: all)")
    p.add_argument("--ignore", help="comma-separated rule ids to skip")
    p.add_argument("--min-severity", choices=["error", "warning", "info"],
                   help="only report findings at this severity or above (default: warning)")
    p.add_argument("--all", action="store_true",
                   help="report everything, including info-level findings")
    p.add_argument("--config", type=Path, help="path to a config file (pyproject.toml or .overllm.toml)")
    p.add_argument("--exit-zero", action="store_true", help="always exit 0, even when findings exist")
    p.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    p.add_argument("--baseline", action="store_true",
                   help="report only findings not recorded in the baseline file")
    p.add_argument("--write-baseline", action="store_true",
                   help="snapshot current findings to the baseline file and exit 0")
    p.add_argument("--update-baseline", action="store_true",
                   help="like --baseline, but also prune fixed findings from the baseline (ratchet)")
    p.add_argument("--baseline-file", type=Path, default=Path(DEFAULT_BASELINE_FILE),
                   help=f"baseline file path (default: {DEFAULT_BASELINE_FILE})")
    p.add_argument("--fix", action="store_true",
                   help="apply safe autofixes (remove sampling params a model rejects), then report the rest")
    p.add_argument("--unsafe-fixes", action="store_true",
                   help="with --fix, also swap a retired/deprecated model id for its replacement")
    p.add_argument("--diff", action="store_true",
                   help="with --fix, print the patch instead of writing it")
    p.add_argument("--version", action="version", version=f"overllm {__version__}")
    return p


def _split(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(v.strip() for v in value.split(",") if v.strip())


def _run_fix(paths: list[str], config, include_unsafe: bool, diff_only: bool) -> int:
    """Apply (or, with diff_only, preview) safe autofixes across the Python files."""
    import difflib

    from . import fix as fx
    from .analyze import iter_python_files

    fixed_files = 0
    fixed_edits = 0
    for fpath in iter_python_files(paths, config.exclude):
        try:
            src = fpath.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        new, edits = fx.fix_source(src, include_unsafe=include_unsafe)
        if not edits or new == src:
            continue
        if diff_only:
            sys.stdout.writelines(
                difflib.unified_diff(
                    src.splitlines(keepends=True), new.splitlines(keepends=True),
                    fromfile=str(fpath), tofile=str(fpath),
                )
            )
        else:
            fpath.write_text(new, encoding="utf-8")
            fixed_files += 1
            fixed_edits += len(edits)
    if not diff_only and fixed_files:
        print(f"overllm: applied {fixed_edits} fix(es) across {fixed_files} file(s).")
    return 0


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
    if args.all:
        config.min_severity = "info"
    elif args.min_severity:
        config.min_severity = args.min_severity

    # Autofix runs first (it mutates files), then findings are computed on the
    # result so the report shows what remains.
    if args.fix or args.diff:
        rc = _run_fix(paths, config, include_unsafe=args.unsafe_fixes, diff_only=args.diff)
        if args.diff:
            return rc

    findings = analyze_paths(paths, config)

    # Baseline / ratchet: snapshot, or report only findings new since the snapshot.
    if args.write_baseline:
        n = bl.write(args.baseline_file, findings, __version__)
        print(f"overllm: wrote baseline of {n} finding group(s) to {args.baseline_file}")
        return 0
    if args.baseline or args.update_baseline:
        base = bl.load(args.baseline_file)
        if not base:
            print(f"overllm: no baseline at {args.baseline_file}; reporting all findings",
                  file=sys.stderr)
        if args.update_baseline:
            bl.write_pruned(args.baseline_file, findings, base, __version__)
        findings = bl.filter_new(findings, base)

    use_color = sys.stdout.isatty() and not args.no_color
    output = render(findings, args.format, use_color=use_color)
    if output:
        print(output)

    if args.exit_zero:
        return 0
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
