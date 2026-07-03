"""End-to-end tests: file discovery, suppression, config, severity, exit codes, formats."""

from __future__ import annotations

import json
from pathlib import Path

from overllm.analyze import analyze_paths
from overllm.cli import main
from overllm.config import Config
from overllm.report import render

# error-level (llm-mechanical): shown and gates by default
MECH_CALL = (
    'client.chat.completions.create(model="gpt-4o", '
    'messages=[{"role":"user","content":"Sort these names alphabetically: a, b, c"}])\n'
)
# warning-level (llm-in-loop): shown by default
LOOP_CALL = (
    "for row in rows:\n"
    '    client.chat.completions.create(model="gpt-4o", '
    'messages=[{"role":"user","content": f"Tag {row}"}])\n'
)
# info-level (static-prompt): quiet by default
STATIC_CALL = (
    'client.chat.completions.create(model="gpt-4o", '
    'messages=[{"role":"user","content":"Write a haiku about the sea."}])\n'
)


def write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def test_finds_across_directory(tmp_path):
    write(tmp_path, "a.py", MECH_CALL)
    findings = analyze_paths([str(tmp_path)], Config())
    assert len(findings) == 1
    assert findings[0].rule == "llm-mechanical"
    assert findings[0].severity == "error"


def test_inline_suppression(tmp_path):
    body = MECH_CALL.rstrip("\n") + "  # overllm: ignore\n"
    write(tmp_path, "a.py", body)
    assert analyze_paths([str(tmp_path)], Config()) == []


def test_inline_suppression_specific_rule(tmp_path):
    body = MECH_CALL.rstrip("\n") + "  # overllm: ignore=llm-mechanical\n"
    write(tmp_path, "a.py", body)
    assert analyze_paths([str(tmp_path)], Config()) == []


def test_ignore_file_directive(tmp_path):
    body = "# overllm: ignore-file\n" + MECH_CALL
    write(tmp_path, "a.py", body)
    assert analyze_paths([str(tmp_path)], Config()) == []


def test_config_ignore_rule(tmp_path):
    write(tmp_path, "a.py", MECH_CALL)
    cfg = Config(ignore=("llm-mechanical",))
    assert analyze_paths([str(tmp_path)], cfg) == []


def test_excluded_dirs_skipped(tmp_path):
    vendor = tmp_path / "node_modules"
    vendor.mkdir()
    write(vendor, "b.py", MECH_CALL)
    assert analyze_paths([str(tmp_path)], Config()) == []


def test_syntax_error_file_is_silent(tmp_path):
    write(tmp_path, "bad.py", "def (:\n")
    assert analyze_paths([str(tmp_path)], Config()) == []


# --- severity thresholds -----------------------------------------------------

def test_static_prompt_is_info_and_hidden_by_default(tmp_path):
    write(tmp_path, "a.py", STATIC_CALL)
    assert analyze_paths([str(tmp_path)], Config()) == []


def test_static_prompt_shown_with_min_severity_info(tmp_path):
    write(tmp_path, "a.py", STATIC_CALL)
    findings = analyze_paths([str(tmp_path)], Config(min_severity="info"))
    assert len(findings) == 1
    assert findings[0].rule == "static-prompt"


def test_min_severity_error_hides_warning_loop(tmp_path):
    write(tmp_path, "a.py", LOOP_CALL)
    assert analyze_paths([str(tmp_path)], Config()) != []  # warning shown by default
    assert analyze_paths([str(tmp_path)], Config(min_severity="error")) == []


def test_cli_all_flag_shows_info(tmp_path):
    write(tmp_path, "a.py", STATIC_CALL)
    assert main([str(tmp_path)]) == 0            # hidden by default
    assert main([str(tmp_path), "--all"]) == 1   # shown with --all


# --- exit codes / formats ----------------------------------------------------

def test_cli_exit_code_nonzero_on_findings(tmp_path):
    write(tmp_path, "a.py", MECH_CALL)
    assert main([str(tmp_path)]) == 1


def test_cli_exit_zero_flag(tmp_path):
    write(tmp_path, "a.py", MECH_CALL)
    assert main([str(tmp_path), "--exit-zero"]) == 0


def test_cli_clean_repo_exits_zero(tmp_path):
    write(tmp_path, "a.py", "x = sorted(names)\n")
    assert main([str(tmp_path)]) == 0


def test_json_and_sarif_and_markdown_render(tmp_path):
    write(tmp_path, "a.py", MECH_CALL)
    findings = analyze_paths([str(tmp_path)], Config())
    data = json.loads(render(findings, "json"))
    assert data[0]["rule"] == "llm-mechanical"
    assert data[0]["severity"] == "error"
    sarif = json.loads(render(findings, "sarif", use_color=False))
    assert sarif["runs"][0]["results"][0]["level"] == "error"
    md = render(findings, "markdown")
    assert "overllm" in md and "llm-mechanical" in md


def test_markdown_empty_when_clean():
    assert render([], "markdown") == ""
