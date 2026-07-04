"""Baseline / ratchet mode: fingerprint stability, suppression, and ratcheting."""

from __future__ import annotations

from overllm import baseline as bl
from overllm.cli import main
from overllm.models import Finding


def F(rule="llm-mechanical", path="a.py", line=1, col=0, snippet="llm(...)", model=None):
    return Finding(path=path, line=line, col=col, rule=rule, message="m", snippet=snippet, model=model)


# --- fingerprint --------------------------------------------------------------

def test_fingerprint_ignores_line_number():
    assert bl.fingerprint(F(line=1)) == bl.fingerprint(F(line=999))


def test_fingerprint_ignores_snippet_whitespace():
    assert bl.fingerprint(F(snippet="a( x )")) == bl.fingerprint(F(snippet="a(   x )"))


def test_fingerprint_distinguishes_model_rule_snippet():
    assert bl.fingerprint(F(model="claude-2")) != bl.fingerprint(F(model="claude-3"))
    assert bl.fingerprint(F(rule="llm-in-loop")) != bl.fingerprint(F(rule="llm-mechanical"))
    assert bl.fingerprint(F(snippet="a")) != bl.fingerprint(F(snippet="b"))


# --- write / load / filter ----------------------------------------------------

def test_write_load_roundtrip(tmp_path):
    p = tmp_path / "bl.json"
    n = bl.write(p, [F(snippet="one"), F(snippet="two")], "0.6.0")
    assert n == 2
    base = bl.load(p)
    assert bl.fingerprint(F(snippet="one")) in base


def test_load_missing_file_is_empty():
    assert bl.load(bl.Path("does-not-exist-xyz.json")) == {}


def test_filter_new_suppresses_baselined_reports_new():
    base = bl.build([F(snippet="old")])
    kept = bl.filter_new([F(snippet="old", line=7), F(snippet="new", line=9)], base)
    assert [k.snippet for k in kept] == ["new"]


def test_filter_new_reports_extra_occurrence():
    base = bl.build([F(snippet="dup")])  # count 1
    kept = bl.filter_new([F(snippet="dup", line=1), F(snippet="dup", line=2)], base)
    assert len(kept) == 1  # one suppressed, the newly-introduced twin fires


# --- ratchet ------------------------------------------------------------------

def test_prune_drops_fixed_and_tightens_count():
    base = bl.build([F(snippet="a"), F(snippet="a", line=2), F(snippet="b")])  # a:2, b:1
    pruned = bl.prune([F(snippet="a")], base)  # only one 'a' remains, 'b' fixed
    fa = bl.fingerprint(F(snippet="a"))
    assert list(pruned) == [fa]
    assert pruned[fa]["count"] == 1


def test_prune_never_absorbs_new_findings():
    base = bl.build([F(snippet="a")])
    pruned = bl.prune([F(snippet="a"), F(snippet="brand-new")], base)
    assert bl.fingerprint(F(snippet="brand-new")) not in pruned


# --- CLI end-to-end -----------------------------------------------------------

_CALL = 'r{n} = client.chat.completions.create(model="gpt-4o", messages=[{{"role": "user", "content": "{p}"}}])\n'


def test_cli_baseline_end_to_end(tmp_path):
    code = tmp_path / "x.py"
    code.write_text(_CALL.format(n=1, p="sort these names: a, b"))
    bfile = tmp_path / "overllm-baseline.json"

    # snapshot -> exit 0, file written
    assert main([str(code), "--write-baseline", "--baseline-file", str(bfile)]) == 0
    assert bfile.exists()

    # everything baselined -> nothing new -> exit 0
    assert main([str(code), "--baseline", "--baseline-file", str(bfile)]) == 0

    # introduce a NEW wasteful call -> reported -> exit 1
    code.write_text(code.read_text() + _CALL.format(n=2, p="reverse this string: hi"))
    assert main([str(code), "--baseline", "--baseline-file", str(bfile)]) == 1
