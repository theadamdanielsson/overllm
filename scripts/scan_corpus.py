#!/usr/bin/env python3
"""Scan many real GitHub repos with overllm to measure precision and find real waste.

Finds repos that call LLM SDKs, shallow-clones a sample, and analyzes every file
with overllm as a library (Python via ast, JS/TS via tree-sitter) so it can report
more than the CLI: the batchable-vs-necessary loop split, prompts taking untrusted
input, and the rule findings. Two jobs: hardening (a low false-positive rate on
real code is the whole value) and the "how much are people wasting" data.

Usage:
    python scripts/scan_corpus.py --lang js --limit 80 --workdir /tmp/overllm-js

Requires: gh (authenticated), git, and overllm (plus overllm[js] for --lang js).
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

from overllm.config import DEFAULT_EXCLUDES
from overllm.detector import find_llm_calls
from overllm.jsdetector import find_llm_calls_js, lang_for
from overllm.rules import run_rules

warnings.filterwarnings("ignore", category=SyntaxWarning)

QUERIES_PY = [
    "client.chat.completions.create language:python",
    "openai.ChatCompletion.create language:python",
    "client.messages.create anthropic language:python",
    "litellm.completion language:python",
    "chat.completions.create iterrows language:python",
    "chat.completions.create for row language:python",
]
QUERIES_JS = [
    "generateText from ai language:typescript",
    "streamText from ai language:typescript",
    "chat.completions.create language:typescript",
    "messages.create anthropic language:typescript",
    "generateObject language:typescript",
    "generateText language:javascript",
]

SKIP = (
    "langchain", "transformers", "llama_index", "llama-index", "autogen",
    "litellm", "openai/openai", "vllm", "awesome", "ollama/ollama", "haystack",
    "vercel/ai", "vercel/ai-sdk", "anthropics/", "openai/openai-node",
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


def collect(queries: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    repos: list[str] = []
    for q in queries:
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
    except OSError:
        return findings, counts

    js = lang_for(str(path))
    try:
        if js is not None:
            calls = find_llm_calls_js(src, js)
        else:
            calls = find_llm_calls(ast.parse(src), src.splitlines())
    except (SyntaxError, ValueError, RecursionError, Exception):
        return findings, counts

    for c in calls:
        counts["call_sites"] += 1
        if c.loop_kind:
            counts[f"loop_{c.loop_kind}"] += 1
        if c.tainted:
            counts["tainted_prompts"] += 1
        findings.extend(run_rules(c, str(path)))
    return findings, counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["py", "js", "all"], default="py")
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--workdir", default="/tmp/overllm-corpus")
    args = ap.parse_args()
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)

    queries = {"py": QUERIES_PY, "js": QUERIES_JS, "all": QUERIES_PY + QUERIES_JS}[args.lang]
    repos = collect(queries, args.limit)
    print(f"candidate repos: {len(repos)}", flush=True)
    cloned: list[tuple[str, Path]] = []
    for r in repos:
        dest = work / r.replace("/", "__")
        if clone(r, dest):
            cloned.append((r, dest))
    print(f"cloned: {len(cloned)}", flush=True)

    exts = (".py", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts")
    totals: collections.Counter = collections.Counter()
    by_rule: collections.Counter = collections.Counter()
    report: list[dict] = []
    scanned = 0
    for r, dest in cloned:
        for f in dest.rglob("*"):
            if not f.is_file() or f.suffix not in exts:
                continue
            if any(part in DEFAULT_EXCLUDES for part in f.parts):
                continue
            scanned += 1
            findings, counts = analyze_file(f)
            totals.update(counts)
            for finding in findings:
                by_rule[finding.rule] += 1
                report.append({
                    "repo": r, "path": str(f.relative_to(dest)), "line": finding.line,
                    "rule": finding.rule, "snippet": finding.snippet[:110],
                })

    print(f"\nfiles scanned: {scanned}")
    print(f"LLM call sites found: {totals['call_sites']}")
    print("loop split:  batchable (savings) = "
          f"{totals['loop_batchable']}   necessary (agent/retry/while) = {totals['loop_necessary']}")
    print(f"prompts taking untrusted web input: {totals['tainted_prompts']}")
    print(f"\nactionable rule findings: {sum(by_rule.values())}")
    print(f"by rule: {dict(by_rule)}")
    print("\nsample (spot-check for false positives):")
    for f in report[:30]:
        print(f"  [{f['rule']}] {f['repo']} {f['path']}:{f['line']}  {f['snippet'][:65]}")

    (work / "_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nfull report: {work / '_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
