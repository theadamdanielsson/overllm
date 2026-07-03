"""End-to-end tests: file discovery, suppression, config, exit codes, formats."""

from __future__ import annotations

import json
from pathlib import Path

from overllm.analyze import analyze_paths
from overllm.cli import main
from overllm.config import Config
from overllm.report import render

STATIC_CALL = (
    'client.chat.completions.create(model="gpt-4o", '
    'messages=[{"role":"user","content":"Write a haiku about the sea."}])\n'
)


def write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def test_finds_across_directory(tmp_path):
    write(tmp_path, "a.py", STATIC_CALL)
    findings = analyze_paths([str(tmp_path)], Config())
    assert len(findings) == 1
    assert findings[0].rule == "static-prompt"


def test_inline_suppression(tmp_path):
    body = STATIC_CALL.rstrip("\n") + "  # overllm: ignore\n"
    write(tmp_path, "a.py", body)
    assert analyze_paths([str(tmp_path)], Config()) == []


def test_inline_suppression_specific_rule(tmp_path):
    body = STATIC_CALL.rstrip("\n") + "  # overllm: ignore=static-prompt\n"
    write(tmp_path, "a.py", body)
    assert analyze_paths([str(tmp_path)], Config()) == []


def test_ignore_file_directive(tmp_path):
    body = "# overllm: ignore-file\n" + STATIC_CALL
    write(tmp_path, "a.py", body)
    assert analyze_paths([str(tmp_path)], Config()) == []


def test_config_ignore_rule(tmp_path):
    write(tmp_path, "a.py", STATIC_CALL)
    cfg = Config(ignore=("static-prompt",))
    assert analyze_paths([str(tmp_path)], cfg) == []


def test_excluded_dirs_skipped(tmp_path):
    vendor = tmp_path / "node_modules"
    vendor.mkdir()
    write(vendor, "b.py", STATIC_CALL)
    assert analyze_paths([str(tmp_path)], Config()) == []


def test_syntax_error_file_is_silent(tmp_path):
    write(tmp_path, "bad.py", "def (:\n")
    assert analyze_paths([str(tmp_path)], Config()) == []


def test_cli_exit_code_nonzero_on_findings(tmp_path):
    write(tmp_path, "a.py", STATIC_CALL)
    assert main([str(tmp_path)]) == 1


def test_cli_exit_zero_flag(tmp_path):
    write(tmp_path, "a.py", STATIC_CALL)
    assert main([str(tmp_path), "--exit-zero"]) == 0


def test_cli_clean_repo_exits_zero(tmp_path):
    write(tmp_path, "a.py", "x = sorted(names)\n")
    assert main([str(tmp_path)]) == 0


def test_json_and_sarif_and_markdown_render(tmp_path):
    write(tmp_path, "a.py", STATIC_CALL)
    findings = analyze_paths([str(tmp_path)], Config())
    data = json.loads(render(findings, "json"))
    assert data[0]["rule"] == "static-prompt"
    sarif = json.loads(render(findings, "sarif", use_color=False))
    assert sarif["runs"][0]["results"][0]["ruleId"] == "static-prompt"
    md = render(findings, "markdown")
    assert "overllm" in md and "static-prompt" in md


def test_markdown_empty_when_clean():
    assert render([], "markdown") == ""
