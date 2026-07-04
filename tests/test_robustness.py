"""Regression tests for the robustness bugs found by the 0.6.0 red-team:
- --fix byte-vs-char offset (silent corruption + crash) on non-ASCII lines
- detector unbounded recursion on deep concat prompts
- analyze must never let one file abort a scan
- baseline: distinct multi-line calls must not fingerprint-collide; paths must
  be stable across relative/absolute invocation.
"""

from __future__ import annotations

import ast

from overllm import baseline as bl
from overllm.analyze import analyze_paths
from overllm.config import Config
from overllm.detector import find_llm_calls
from overllm.fix import fix_source
from overllm.rules import run_rules

_NOSAMPLE = "claude-opus-4-8"  # rejects sampling params


# --- --fix offset correctness on non-ASCII (CRITICAL #1 / #2) -----------------

def test_fix_multibyte_before_kwarg_never_corrupts():
    src = (f'r = client.chat.completions.create(model="{_NOSAMPLE}", '
           'messages=[{"role":"user","content":"🎈🎈🎈"}], temperature=0.5, max_tokens=1000)\n')
    new, edits = fix_source(src)
    assert edits, "should remove the rejected sampling param"
    ast.parse(new)                       # must still compile
    assert "temperature" not in new
    assert "max_tokens=1000" in new      # NOT mangled into 'temperatuens=1000'
    assert "🎈🎈🎈" in new


def test_fix_multibyte_never_raises():
    long = "é" * 40
    safe = (f'r = client.chat.completions.create(model="{_NOSAMPLE}", '
            f'messages=[{{"role":"user","content":"{long}"}}], temperature=0.5)\n')
    new, _ = fix_source(safe)            # must not raise IndexError
    ast.parse(new)
    assert "temperature" not in new
    unsafe = (f'r = client.chat.completions.create(model="claude-2", '
              f'messages=[{{"role":"user","content":"{long}"}}])\n')
    new2, _ = fix_source(unsafe, include_unsafe=True)
    ast.parse(new2)
    assert "claude-sonnet-4-6" in new2


def test_fix_formfeed_in_earlier_line_no_corruption():
    src = ('x = "a\x0cb"\n'
           f'r = client.chat.completions.create(model="{_NOSAMPLE}", '
           'messages=[{"role":"user","content":"hi"}], temperature=0.5)\n')
    new, _ = fix_source(src)
    ast.parse(new)
    assert "temperature" not in new and 'x = "a\x0cb"' in new


# --- detector recursion + scan containment (CRITICAL #3) ----------------------

def test_detector_deep_concat_prompt_no_recursionerror():
    concat = "+".join(['"a"'] * 6000)
    src = f'x = client.chat.completions.create(model="gpt-4o", messages=[{{"role":"user","content":{concat}}}])'
    find_llm_calls(ast.parse(src), src.splitlines())  # must not raise


def test_analyze_one_bad_file_does_not_abort_scan(tmp_path):
    concat = "+".join(['"a"'] * 6000)
    (tmp_path / "aaa_bad.py").write_text(
        f'x = client.chat.completions.create(model="gpt-4o", messages=[{{"role":"user","content":{concat}}}])\n')
    (tmp_path / "zzz_good.py").write_text(
        'r = client.chat.completions.create(model="claude-2", messages=[{"role":"user","content":"hi"}])\n')
    findings = analyze_paths([str(tmp_path)], Config())      # must not raise
    assert any(f.rule == "deprecated-model" for f in findings), "good file must still be scanned"


# --- --fix safe matrix + overlap two-pass -------------------------------------

def test_fix_two_adjacent_sampling_params_two_pass():
    src = ('r = client.chat.completions.create(model="o3", '
           'messages=[{"role":"user","content":"hi there"}], temperature=0.5, top_p=0.9)\n')
    once, _ = fix_source(src)
    ast.parse(once)
    twice, _ = fix_source(once)
    ast.parse(twice)
    assert "temperature" not in twice and "top_p" not in twice


def test_fix_safe_matrix():
    base = 'r = client.chat.completions.create(model="o3", messages=[{{"role":"user","content":"hi there"}}], {tail}\n'
    for tail in ("temperature=0.5)  # keep me",
                 "temperature   =   0.5 )",
                 "temperature=(0.5))"):
        new, _ = fix_source(base.format(tail=tail))
        ast.parse(new)
        assert "temperature=" not in new
    assert "# keep me" in fix_source(base.format(tail="temperature=0.5)  # keep me"))[0]
    # CRLF multi-line call
    crlf = ('r = client.chat.completions.create(model="o3",\r\n'
            ' messages=[{"role":"user","content":"hi there"}],\r\n temperature=0.5)\n')
    ast.parse(fix_source(crlf)[0])
    assert "temperature=" not in fix_source(crlf)[0]
    # nested inner keyword of a non-LLM call must be untouched
    nested = ('r = client.chat.completions.create(model="o3", '
              'messages=[{"role":"user","content":"hi there"}], extra=helper(temperature=1))\n')
    nn, _ = fix_source(nested)
    assert "temperature=1" in nn
    ast.parse(nn)
    # prefixed / non-literal model must never be rewritten by the unsafe swap
    for m in ('f"claude-2"', 'r"claude-2"', 'b"claude-2"'):
        s = (f'r = client.chat.completions.create(model={m}, '
             'messages=[{"role":"user","content":"hi there"}])\n')
        out, _ = fix_source(s, include_unsafe=True)
        assert "claude-sonnet-4-6" not in out


# --- baseline correctness (HIGH #4 / #5) --------------------------------------

def _deprecated_findings(src):
    calls = find_llm_calls(ast.parse(src), src.splitlines())
    return [f for c in calls for f in run_rules(c, "a.py") if f.rule == "deprecated-model"]


def test_baseline_distinct_multiline_calls_dont_collide():
    # same rule, same model, IDENTICAL first line -> only the full-call snippet
    # keeps their fingerprints apart.
    src = ('client.chat.completions.create(\n'
           '    model="claude-2", messages=[{"role":"user","content":"alpha wolf"}])\n'
           'client.chat.completions.create(\n'
           '    model="claude-2", messages=[{"role":"user","content":"beta fish"}])\n')
    dep = _deprecated_findings(src)
    assert len(dep) == 2
    fps = {bl.fingerprint(f) for f in dep}
    assert len(fps) == 2, "distinct calls must fingerprint distinctly"
    # and a genuinely new one is not masked when an old one is baselined
    base = bl.build([dep[0]])
    kept = bl.filter_new([dep[0], dep[1]], base)
    assert any(bl.fingerprint(k) == bl.fingerprint(dep[1]) for k in kept)


def test_baseline_path_stable_relative_vs_absolute(tmp_path, monkeypatch):
    code = tmp_path / "app.py"
    code.write_text('r = client.chat.completions.create(model="claude-2", '
                    'messages=[{"role":"user","content":"hi"}])\n')
    monkeypatch.chdir(tmp_path)
    rel = analyze_paths(["app.py"], Config())
    absolute = analyze_paths([str(code)], Config())
    assert rel and absolute
    assert {bl.fingerprint(f) for f in rel} == {bl.fingerprint(f) for f in absolute}
