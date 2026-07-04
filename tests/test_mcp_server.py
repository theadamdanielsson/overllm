import asyncio
import os

import pytest

from overllm.mcp_server import scan_code, scan_path

EXAMPLE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "examples", "needless_llm_calls.py")
)


def test_scan_path_finds_needless_calls():
    findings = scan_path(EXAMPLE)
    assert len(findings) >= 1
    rules = {f["rule"] for f in findings}
    assert "llm-mechanical" in rules  # the example sorts with an LLM
    for f in findings:
        assert f["suggestion"]  # every finding names a replacement


def test_scan_code_scans_inline_and_relabels_path():
    with open(EXAMPLE, encoding="utf-8") as handle:
        code = handle.read()
    findings = scan_code(code, "python")
    assert len(findings) >= 1
    assert all(f["path"] == "<snippet>" for f in findings)


def test_scan_code_is_quiet_on_clean_code():
    assert scan_code("x = 1 + 1\nprint(x)\n", "python") == []


def test_unknown_path_returns_no_findings_not_a_crash():
    assert scan_path("/nope/does/not/exist") == []


def test_server_exposes_both_tools():
    pytest.importorskip("mcp")
    from overllm.mcp_server import build_server

    server = build_server()
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert {"scan_path", "scan_code"} <= names
