#!/usr/bin/env python3
"""Scan many real GitHub repos with overllm to measure precision and find real waste.

Finds repos that actually call LLM SDKs, shallow-clones a sample, and analyzes
every file with overllm as a library so it can report more than the CLI does:
the batchable-vs-necessary loop split, how many prompts take untrusted input, and
the actual rule findings. Two jobs: hardening (a low false-positive rate on real
code is the whole value) and the "how much are people wasting" data for a launch.

Usage:
    python scripts/scan_corpus.py --limit 80 --workdir /tmp/overllm-corpus

Requires: gh (authenticated), git, and overllm installed in the same environment.
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import subprocess
import sys
import warnings
from pathlib import Path

from overllm.detector import find_llm_calls
from overllm.rules import run_rules

warnings.filterwarnings("ignore", category=SyntaxWarning)  # from parsing others' code

# Queries that hit real call sites, weighted toward code that tends to over-call
# (per-row loops, scrapers, classifiers) so we surface batchable waste.
QUERIES = [
    "client.chat.completions.create language:python",
    "openai.ChatCompletion.create language:python",
    "client.messages.create anthropic language:python",
    "litellm.completion language:python",
    "ollama.chat language:python",
    "chat.completions.create iterrows language:python",
    "chat.completions.create scrape language:python",
    "chat.completions.create classify language:python",
    "chat.completions.create for row language:python",
]

SKIP = (
    "langchain", "transformers", "llama_index", "llama-index", "autogen",
    "litellm", "openai/openai", "vllm", "awesome", "ollama/ollama", "haystack",
)


def gh_repos(query: str, per: int) -> list[str]:
    try:
        out = subprocess.run(
            ["gh", "search", "code", query, "-L", str(per), "--json", "repository"],
            capture_output=True, text=True, timeout=60,
        ).stdout
        return [r["repository"]["nameWithOwner"] for r in json.loads(out or "[]")]
    except Exception:
        return []


def collect(limit: int) -> list[str]:
    seen: set[str] = set()
    repos: list[str] = []
    for q in QUERIES:
        for r in gh_repos(q, 40):
            rl = r.lower()
            if r in seen or any(s in rl for s in SKIP):
                continue
            seen.add(r)
            repos.append(r)
            if len(repos) >= limit:
                return repos
    return repos


def clone(repo: str, dest: Path) -> bool:
    if dest.exists():
        return True
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "-q", f"https://github.com/{repo}", str(dest)],
            timeout=90, capture_output=True,
        )
    except Exception:
        return False
    return dest.exists()


def analyze_file(path: Path) -> tuple[list, collections.Counter]:
    counts: collections.Counter = collections.Counter()
    findings: list = []
    try:
        src = path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(src)
    except (SyntaxError, ValueError, OSError, RecursionError):
        return findings, counts
    lines = src.splitlines()
    for c in find_llm_calls(tree, lines):
        counts["call_sites"] += 1
        if c.loop_kind:
            counts[f"loop_{c.loop_kind}"] += 1
        if c.tainted:
            counts["tainted_prompts"] += 1
        findings.extend(run_rules(c, str(path)))
    return findings, counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--workdir", default="/tmp/overllm-corpus")
    args = ap.parse_args()
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)

    repos = collect(args.limit)
    print(f"candidate repos: {len(repos)}", flush=True)
    cloned: list[tuple[str, Path]] = []
    for r in repos:
        dest = work / r.replace("/", "__")
        if clone(r, dest):
            cloned.append((r, dest))
    print(f"cloned: {len(cloned)}", flush=True)

    totals: collections.Counter = collections.Counter()
    by_rule: collections.Counter = collections.Counter()
    report: list[dict] = []
    py_files = 0
    for r, dest in cloned:
        for py in dest.rglob("*.py"):
            py_files += 1
            findings, counts = analyze_file(py)
            totals.update(counts)
            for f in findings:
                by_rule[f.rule] += 1
                report.append({
                    "repo": r, "path": str(py.relative_to(dest)), "line": f.line,
                    "rule": f.rule, "message": f.message, "snippet": f.snippet[:120],
                })

    print(f"\npython files scanned: {py_files}")
    print(f"LLM call sites found: {totals['call_sites']}")
    print("loop split:  batchable (savings) = "
          f"{totals['loop_batchable']}   necessary (agent/retry/while) = {totals['loop_necessary']}")
    print(f"prompts taking untrusted input: {totals['tainted_prompts']}")
    print(f"\nactionable rule findings: {sum(by_rule.values())}")
    print(f"by rule: {dict(by_rule)}")

    savings = [f for f in report if f["rule"] in ("llm-in-loop", "llm-extraction", "llm-mechanical", "static-prompt")]
    print("\nsavings candidates (spot-check):")
    for f in savings[:30]:
        print(f"  [{f['rule']}] {f['repo']} {f['path']}:{f['line']}  {f['snippet'][:70]}")

    (work / "_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nfull report: {work / '_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
