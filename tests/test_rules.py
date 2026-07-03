"""Rule-level tests. Positive cases must fire; negative cases must stay silent."""

from __future__ import annotations

import ast

from overllm.detector import find_llm_calls
from overllm.rules import run_rules


def rules_for(source: str) -> list[str]:
    tree = ast.parse(source)
    lines = source.splitlines()
    out: list[str] = []
    for call in find_llm_calls(tree, lines):
        for f in run_rules(call, "<test>"):
            out.append(f.rule)
    return out


def rule_set(source: str) -> set[str]:
    return set(rules_for(source))


def n_calls(source: str) -> int:
    tree = ast.parse(source)
    return len(find_llm_calls(tree, source.splitlines()))


# --- detection ---------------------------------------------------------------

def test_detects_openai_chat_completions():
    src = 'r = client.chat.completions.create(model="gpt-4o", messages=[{"role":"user","content": x}])'
    assert n_calls(src) == 1


def test_detects_anthropic_messages():
    src = 'm = client.messages.create(model="claude-opus-4-8", messages=[{"role":"user","content": x}])'
    assert n_calls(src) == 1


def test_detects_langchain_invoke_on_tracked_var():
    src = "llm = ChatOpenAI(model='gpt-4o')\nout = llm.invoke(prompt)"
    assert n_calls(src) == 1


def test_detects_litellm_free_function():
    src = "from litellm import completion\nr = completion(model='gpt-4o', messages=msgs)"
    assert n_calls(src) == 1


def test_detects_raw_http_to_llm_host():
    src = 'r = requests.post("https://api.openai.com/v1/chat/completions", json=body)'
    assert n_calls(src) == 1


def test_ignores_lookalike_non_llm_call():
    # messages.create with no LLM-shaped kwarg is NOT an LLM call
    src = 'db.messages.create(text="hello", channel_id=3)'
    assert n_calls(src) == 0


def test_ignores_plain_stdlib():
    src = "y = sorted(names)\nz = json.loads(text)\nq = db.records.create(data=1)"
    assert n_calls(src) == 0


# --- R1 static-prompt --------------------------------------------------------

def test_static_prompt_flagged():
    src = 'client.chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":"Write a haiku about the sea."}])'
    assert "static-prompt" in rule_set(src)


def test_dynamic_user_prompt_not_static():
    src = 'client.chat.completions.create(model="gpt-4o", messages=[{"role":"user","content": f"Summarize: {article}"}])'
    assert "static-prompt" not in rule_set(src)


def test_static_system_but_dynamic_user_not_flagged():
    src = (
        'client.chat.completions.create(model="gpt-4o", messages=['
        '{"role":"system","content":"You are a helpful assistant."},'
        '{"role":"user","content": user_text}])'
    )
    assert "static-prompt" not in rule_set(src)


# --- R2 llm-extraction -------------------------------------------------------

def test_extraction_email_flagged():
    src = 'client.chat.completions.create(model="m", messages=[{"role":"user","content": f"Extract the email from: {text}"}])'
    assert "llm-extraction" in rule_set(src)


def test_json_output_request_not_flagged():
    # regression (found on a real repo): asking for JSON-shaped output is legitimate
    # structured output, and "extract info from a webpage" is real semantic work, not a regex
    src = 'client.chat.completions.create(model="m", messages=[{"role":"user","content": f"Extract all product info and format it as JSON: {content}"}])'
    assert rule_set(src) == set()


# --- R4 llm-mechanical -------------------------------------------------------

def test_mechanical_sort_flagged():
    src = "llm = ChatOpenAI()\nllm.invoke(f'Sort these names alphabetically: {names}')"
    assert "llm-mechanical" in rule_set(src)


def test_mechanical_reverse_flagged():
    src = "llm = ChatOpenAI()\nllm.invoke(f'Reverse the string: {s}')"
    assert "llm-mechanical" in rule_set(src)


def test_mechanical_arithmetic_flagged():
    src = 'client.chat.completions.create(model="m", messages=[{"role":"user","content":"What is 2384 * 91?"}])'
    assert "llm-mechanical" in rule_set(src)


# --- R3 llm-in-loop ----------------------------------------------------------

def test_llm_in_for_loop_flagged():
    src = (
        "for item in items:\n"
        '    client.chat.completions.create(model="m", messages=[{"role":"user","content": f"Summarize {item}"}])'
    )
    assert "llm-in-loop" in rule_set(src)


def test_llm_in_comprehension_flagged():
    src = 'results = [client.messages.create(model="m", messages=[{"role":"user","content": f"tag {x}"}]) for x in xs]'
    assert "llm-in-loop" in rule_set(src)


def test_llm_not_in_loop_not_flagged():
    src = 'client.chat.completions.create(model="m", messages=[{"role":"user","content": f"Summarize {article}"}])'
    assert "llm-in-loop" not in rule_set(src)


def test_streaming_async_for_not_flagged():
    # regression (found on a real repo): `async for chunk in create(stream=True)` is
    # ONE call whose chunks are iterated, not N calls
    src = (
        "async def go():\n"
        "    async for chunk in client.chat.completions.create("
        "model='m', messages=[{'role':'user','content': p}], stream=True):\n"
        "        print(chunk)"
    )
    assert "llm-in-loop" not in rule_set(src)


def test_call_as_for_iterable_not_flagged():
    src = "for x in client.chat.completions.create(model='m', messages=[{'role':'user','content': p}]):\n    print(x)"
    assert "llm-in-loop" not in rule_set(src)


def test_streaming_inside_outer_loop_still_flagged():
    # the call is the inner loop's iterable, but the outer loop makes it run N times
    src = (
        "async def go():\n"
        "    for item in items:\n"
        "        async for chunk in client.chat.completions.create("
        "model='m', messages=[{'role':'user','content': item}], stream=True):\n"
        "            print(chunk)"
    )
    assert "llm-in-loop" in rule_set(src)


# --- precision: a normal, justified LLM call produces zero findings ----------

def test_normal_dynamic_call_is_silent():
    src = (
        'client.chat.completions.create(model="gpt-4o", messages=['
        '{"role":"system","content":"You are a support agent."},'
        '{"role":"user","content": f"The customer wrote: {message}. Draft a warm reply."}])'
    )
    assert rule_set(src) == set()
