#!/usr/bin/env python3
"""Scan many real GitHub repos with overllm to measure precision and find real waste.

Uses `gh search code` to find repos that actually call LLM SDKs, shallow-clones a
sample, runs overllm on each, and prints an aggregate report. Two jobs at once:
hardening (a low false-positive rate on real code is the whole value), and the
"how much are people wasting" data for a launch post.

Usage:
    python scripts/scan_corpus.py --limit 50 --workdir /tmp/overllm-corpus

Requires: gh (authenticated), git, and overllm installed in the same environment.
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path

# Queries that hit real call sites across SDKs, not repos that merely mention them.
QUERIES = [
    "client.chat.completions.create language:python",
    "openai.ChatCompletion.create language:python",
    "client.messages.create anthropic language:python",
    "litellm.completion language:python",
    "ollama.chat language:python",
]

# Skip the big libraries themselves; we want application code that uses them.
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


def scan(path: Path) -> list[dict]:
    try:
        out = subprocess.run(
            [sys.executable, "-m", "overllm", str(path), "--format", "json", "--exit-zero"],
            capture_output=True, text=True, timeout=300,
        ).stdout
        return json.loads(out or "[]")
    except Exception:
        return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--workdir", default="/tmp/overllm-corpus")
    args = ap.parse_args()
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)

    repos = collect(args.limit)
    print(f"candidate repos: {len(repos)}")
    cloned: list[tuple[str, Path]] = []
    for r in repos:
        dest = work / r.replace("/", "__")
        if clone(r, dest):
            cloned.append((r, dest))
    print(f"cloned: {len(cloned)}")

    by_rule: collections.Counter = collections.Counter()
    findings: list[dict] = []
    py_files = 0
    for r, dest in cloned:
        py_files += sum(1 for _ in dest.rglob("*.py"))
        for f in scan(dest):
            f["repo"] = r
            findings.append(f)
            by_rule[f["rule"]] += 1

    repos_with = len({f["repo"] for f in findings})
    print(f"\npython files scanned: {py_files}")
    print(f"total findings: {len(findings)}  across {repos_with} repos")
    print(f"by rule: {dict(by_rule)}")
    print("\nsample (spot-check for false positives):")
    for f in findings[:30]:
        print(f"  [{f['rule']}] {f['repo']} L{f['line']}: {f['snippet'][:80]}")
    (work / "_report.json").write_text(json.dumps(findings, indent=2))
    print(f"\nfull report: {work / '_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
