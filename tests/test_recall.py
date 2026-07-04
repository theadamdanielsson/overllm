"""Recall additions: unambiguous-chain kwarg-splat detection, and embeddings."""

from __future__ import annotations

import ast

from overllm.detector import find_llm_calls
from overllm.rules import run_rules


def _n(src: str) -> int:
    return len(find_llm_calls(ast.parse(src), src.splitlines()))


def _rules(src: str) -> set[str]:
    return {f.rule for c in find_llm_calls(ast.parse(src), src.splitlines()) for f in run_rules(c, "<t>")}


# --- kwarg-splat on unambiguous chains (was a false negative) ------------------

def test_kwargs_splat_strong_chain_detected():
    assert _n("r = client.chat.completions.create(**params)") == 1
    assert _n("r = client.chat.completions.parse(**params)") == 1
    assert _n("r = client.responses.create(**cfg)") == 1


def test_ambiguous_messages_create_splat_still_guarded():
    # messages.create is not unambiguous (an ORM could expose it) -> keep the guard
    assert _n("x = db.messages.create(**kwargs)") == 0
    assert _n("x = queue.completions.create(**kwargs)") == 0


def test_splat_detected_in_loop_is_batchable():
    src = "for row in rows:\n    client.chat.completions.create(**build(row))"
    assert "llm-in-loop" in _rules(src)


# --- embeddings -> llm-in-loop ------------------------------------------------

def test_embeddings_in_loop_flagged():
    src = 'for doc in docs:\n    client.embeddings.create(model="text-embedding-3-small", input=doc)'
    assert "llm-in-loop" in _rules(src)


def test_embeddings_not_in_loop_is_silent():
    src = 'client.embeddings.create(model="text-embedding-3-small", input=docs)'
    assert _rules(src) == set()   # detected, but a single batched call is not waste


def test_embeddings_batched_loop_not_flagged():
    # already iterating batches -> not one-call-per-item
    src = 'for batch in batches:\n    client.embeddings.create(model="m", input=batch)'
    assert "llm-in-loop" not in _rules(src)


def test_embeddings_needs_input_or_model_guard():
    assert _n("x = obj.embeddings.create()") == 0


# --- module-level prompt resolution -------------------------------------------

def test_module_constant_prompt_resolved():
    src = ('EXTRACT = "extract the email address from the text"\n'
           'def handle(text):\n'
           '    return client.chat.completions.create(model="gpt-4o",\n'
           '        messages=[{"role":"user","content": EXTRACT}])\n')
    assert "llm-extraction" in _rules(src)


def test_module_constant_concat_static_part_read():
    src = ('SYS = "sort these names alphabetically:"\n'
           'def run(names):\n'
           '    return client.chat.completions.create(model="gpt-4o",\n'
           '        messages=[{"role":"user","content": SYS + names}])\n')
    assert "llm-mechanical" in _rules(src)


def test_no_cross_function_name_bleed():
    # b's param `p` must NOT resolve to a's local `p` (only module-level consts fall back)
    src = ('def a():\n'
           '    p = "extract the email address from the text"\n'
           '    return p\n'
           'def b(p):\n'
           '    return client.chat.completions.create(model="gpt-4o",\n'
           '        messages=[{"role":"user","content": p}])\n')
    assert "llm-extraction" not in _rules(src)
