"""Autofix: safe by construction. Removals never corrupt, model swaps are gated,
strings/comments are never touched, and the result always parses."""

from __future__ import annotations

import ast

from overllm.cli import main
from overllm.fix import fix_source
from overllm.rules import _RETIRED_MODELS

_MID = (
    'r = client.chat.completions.create('
    'model="o3", temperature=0.5, messages=[{"role": "user", "content": "hi there"}])'
)
_LAST = (
    'r = client.chat.completions.create('
    'model="o3", messages=[{"role": "user", "content": "hi there"}], temperature=0.5)'
)


def _parses(src: str) -> bool:
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False


# --- unsupported-params removal (SAFE) ---------------------------------------

def test_removes_sampling_param_in_middle():
    new, edits = fix_source(_MID)
    assert edits and "temperature" not in new and _parses(new)
    assert 'model="o3"' in new and "messages=" in new


def test_removes_sampling_param_when_last_arg_no_dangling_comma():
    new, _ = fix_source(_LAST)
    assert "temperature" not in new and _parses(new)
    assert ", )" not in new and ",)" not in new


def test_fix_is_idempotent():
    once, _ = fix_source(_MID)
    twice, edits = fix_source(once)
    assert twice == once and edits == []


def test_accepts_sampling_model_untouched():
    src = _MID.replace('"o3"', '"gpt-4o"')  # gpt-4o accepts temperature
    new, edits = fix_source(src)
    assert edits == [] and new == src


def test_trailing_comment_preserved():
    src = _MID + "  # keep this"
    new, _ = fix_source(src)
    assert "# keep this" in new and "temperature" not in new


def test_does_not_touch_string_containing_param_name():
    # no temperature KWARG; the phrase lives inside the prompt string.
    src = ('r = client.chat.completions.create('
           'model="o3", messages=[{"role": "user", "content": "set temperature=high"}])')
    new, edits = fix_source(src)
    assert edits == [] and "temperature=high" in new


# --- deprecated-model swap (UNSAFE, gated) -----------------------------------

def test_model_swap_requires_unsafe():
    src = ('r = client.chat.completions.create('
           'model="claude-2", messages=[{"role": "user", "content": "hi there"}])')
    safe, e1 = fix_source(src, include_unsafe=False)
    assert e1 == [] and safe == src
    unsafe, e2 = fix_source(src, include_unsafe=True)
    assert e2 and _RETIRED_MODELS["claude-2"] in unsafe and 'model="claude-2"' not in unsafe


def test_model_swap_preserves_quote_style():
    src = ("r = client.chat.completions.create("
           "model='claude-2', messages=[{'role': 'user', 'content': 'hi there'}])")
    new, _ = fix_source(src, include_unsafe=True)
    assert "model='claude-sonnet-4-6'" in new


def test_model_swap_skips_non_literal_model():
    # the rule resolves `m` for detection, but the call-site model= is a variable,
    # so there is no literal to rewrite -> fix leaves it alone.
    src = ('m = "claude-2"\n'
           'r = client.chat.completions.create(model=m, messages=[{"role": "user", "content": "hi there"}])')
    new, edits = fix_source(src, include_unsafe=True)
    assert edits == [] and new == src


# --- CLI wiring --------------------------------------------------------------

def test_cli_fix_writes(tmp_path):
    f = tmp_path / "a.py"
    f.write_text(_MID + "\n")
    main([str(f), "--fix"])
    assert "temperature" not in f.read_text()


def test_cli_diff_does_not_write(tmp_path):
    f = tmp_path / "a.py"
    original = _MID + "\n"
    f.write_text(original)
    rc = main([str(f), "--fix", "--diff"])
    assert rc == 0 and f.read_text() == original


def test_cli_model_swap_needs_unsafe(tmp_path):
    f = tmp_path / "a.py"
    f.write_text('r = client.chat.completions.create('
                 'model="claude-2", messages=[{"role": "user", "content": "hi there"}])\n')
    main([str(f), "--fix"])
    assert 'model="claude-2"' in f.read_text()  # safe pass leaves the model
    main([str(f), "--fix", "--unsafe-fixes"])
    assert "claude-sonnet-4-6" in f.read_text()
