"""JavaScript / TypeScript detection tests (require the [js] extra)."""

from __future__ import annotations

import pytest

from overllm.jsdetector import available, find_llm_calls_js
from overllm.rules import run_rules

pytestmark = pytest.mark.skipif(not available(), reason="tree-sitter not installed")


def rule_set(src: str, lang: str = "ts") -> set:
    out = set()
    for c in find_llm_calls_js(src, lang):
        for f in run_rules(c, "<t>"):
            out.add(f.rule)
    return out


def n_calls(src: str, lang: str = "ts") -> int:
    return len(find_llm_calls_js(src, lang))


def test_detects_vercel_generate_text():
    assert n_calls('const r = await generateText({ model, prompt: "hi there friend" });') == 1


def test_detects_openai_node():
    assert n_calls('const r = await client.chat.completions.create({ model, messages: m });') == 1


def test_detects_anthropic_node():
    assert n_calls('const r = await client.messages.create({ model, messages: m });') == 1


def test_ignores_unrelated_create():
    assert n_calls('const r = db.records.create({ data: 1 });') == 0


def test_vercel_extraction_flagged():
    src = 'const r = await generateText({ model, prompt: "Extract the email from " + x });'
    assert "llm-extraction" in rule_set(src)


def test_mechanical_template_flagged():
    src = 'const r = await client.chat.completions.create({ model, messages: [{ role: "user", content: `Sort these: ${xs}` }] });'
    assert "llm-mechanical" in rule_set(src)


def test_variable_held_prompt_resolved():
    src = "const p = `Extract the email from ${x}`;\nconst r = await generateText({ model, prompt: p });"
    assert "llm-extraction" in rule_set(src)


def test_static_prompt_flagged():
    assert "static-prompt" in rule_set('const r = await generateText({ model, prompt: "Write a haiku about the sea." });')


def test_batchable_for_of_loop_flagged():
    src = "for (const row of rows) { await generateText({ model, prompt: `tag ${row}` }); }"
    assert "llm-in-loop" in rule_set(src)


def test_while_loop_is_necessary():
    assert "llm-in-loop" not in rule_set("while (!done) { await generateText({ model, prompt: p }); }")


def test_streaming_for_await_not_a_loop():
    src = "for await (const chunk of streamText({ model, prompt: p })) { use(chunk); }"
    assert "llm-in-loop" not in rule_set(src)


def test_web_request_prompt_injection():
    src = "const q = req.query.q;\nconst r = await generateText({ model, prompt: `Answer: ${q}` });"
    assert "prompt-injection" in rule_set(src)


def test_normal_dynamic_is_silent():
    assert rule_set('const r = await generateText({ model, prompt: `Summarize: ${article}` });') == set()


def test_jsx_and_tsx_parse():
    assert n_calls('const r = await generateText({ model, prompt: "hello world here" });', "tsx") == 1
    assert n_calls('const r = await generateText({ model, prompt: "hello world here" });', "js") == 1
