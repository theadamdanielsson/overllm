"""An MCP server that exposes overllm to agents.

Two tools: `scan_path` audits files or directories on disk, `scan_code` audits a
snippet passed inline. Both run the overllm CLI (`python -m overllm --format
json`) under the hood, so an agent gets the same deterministic findings a human
gets from the command line — where a model call could be replaced by plain code,
and the concrete replacement.

The scan functions below are import-free (they do not need the MCP SDK), so they
can be tested on their own. The SDK is imported only when the server is built.
"""

import json
import os
import subprocess
import sys
import tempfile

_EXT = {
    "python": ".py",
    "py": ".py",
    "javascript": ".js",
    "js": ".js",
    "typescript": ".ts",
    "ts": ".ts",
    "jsx": ".jsx",
    "tsx": ".tsx",
}


def _run_overllm(args):
    """Run the overllm CLI and return its findings as a list.

    Invoked through the same interpreter (`python -m overllm`) so it does not
    depend on `overllm` being on PATH. Always uses --exit-zero so the
    "findings exist" non-zero exit is not read as a crash.
    """
    cmd = [sys.executable, "-m", "overllm", *args, "--format", "json", "--exit-zero"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except OSError:
        return []
    out = (proc.stdout or "").strip()
    if not out:
        return []
    try:
        data = json.loads(out)
    except ValueError:
        return []
    return data if isinstance(data, list) else []


def scan_path(path, min_severity="warning", all_rules=False):
    """Findings for a file or directory already on disk."""
    args = [path, "--min-severity", min_severity]
    if all_rules:
        args.append("--all")
    return _run_overllm(args)


def scan_code(code, language="python", min_severity="warning", all_rules=False):
    """Findings for a snippet of source passed inline."""
    suffix = _EXT.get((language or "python").lower(), ".py")
    handle, tmp = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fileobj:
            fileobj.write(code or "")
        args = [tmp, "--min-severity", min_severity]
        if all_rules:
            args.append("--all")
        findings = _run_overllm(args)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    for finding in findings:
        finding["path"] = "<snippet>"
    return findings


def build_server():
    """Construct the FastMCP server. Imports the SDK lazily so the scan
    functions above stay usable without it installed."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("overllm")

    @mcp.tool(name="scan_path")
    def scan_path_tool(path: str = ".", min_severity: str = "warning") -> str:
        """Audit a file or directory on disk for needless LLM/AI calls — places
        where a model is used to do what plain code, a library, or a regex does
        cheaper and deterministically. Returns JSON findings, each with a file,
        line, rule, message, and the concrete replacement to use. `min_severity`
        is one of error, warning, info. Nothing is written or changed."""
        return json.dumps({"findings": scan_path(path, min_severity)}, indent=2)

    @mcp.tool(name="scan_code")
    def scan_code_tool(code: str, language: str = "python", min_severity: str = "warning") -> str:
        """Audit a snippet of source code (passed inline) for needless LLM/AI
        calls. Use this when you have code in context but not on disk.
        `language` is one of python, javascript, typescript. Returns JSON
        findings with the rule and the concrete replacement to use."""
        return json.dumps({"findings": scan_code(code, language, min_severity)}, indent=2)

    return mcp


def main():
    build_server().run()


if __name__ == "__main__":
    main()
