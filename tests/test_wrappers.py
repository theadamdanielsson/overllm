"""Wrapper call-graph recall + config allowlist.

Locks in both the recall wins (a project's own LLM wrapper is followed to its
call sites) and the precision boundaries (ambiguous names, `**kwargs` splat, and
unrelated same-named calls must NOT fire).
"""

from __future__ import annotations

import ast

from overllm.detector import find_llm_calls
from overllm.rules import run_rules


def _rules(source: str, allow: tuple = ()) -> set[str]:
    tree = ast.parse(source)
    lines = source.splitlines()
    out: set[str] = set()
    for call in find_llm_calls(tree, lines, allow):
        for f in run_rules(call, "<test>"):
            out.add(f.rule)
    return out


def _messages(source: str, rule: str, allow: tuple = ()) -> list[str]:
    tree = ast.parse(source)
    lines = source.splitlines()
    return [
        f.message
        for call in find_llm_calls(tree, lines, allow)
        for f in run_rules(call, "<test>")
        if f.rule == rule
    ]


# --- wrapper recall wins -----------------------------------------------------

def test_string_param_wrapper_callsite_flagged():
    src = '''
from openai import OpenAI
client = OpenAI()
def ask(prompt):
    return client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": prompt}])
x = ask("sort these names alphabetically: bob, al")
'''
    assert "llm-mechanical" in _rules(src)


def test_passthrough_messages_wrapper_flagged():
    src = '''
from openai import OpenAI
client = OpenAI()
def chat(messages):
    return client.chat.completions.create(model="gpt-4o", messages=messages)
out = chat([{"role": "user", "content": "reverse this string: hello"}])
'''
    assert "llm-mechanical" in _rules(src)


def test_method_wrapper_in_loop_is_batchable():
    src = '''
from openai import OpenAI
client = OpenAI()
class Bot:
    def ask(self, prompt):
        return client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": prompt}])
    def run(self, items):
        for it in items:
            self.ask("x " + it)
'''
    assert "llm-in-loop" in _rules(src)


# --- precision boundaries ----------------------------------------------------

def test_kwargs_splat_wrapper_not_detected():
    # messages flows through **kwargs -> no literal slot -> deliberately unhandled.
    src = '''
import litellm
def send(messages):
    kwargs = {"model": "gpt-4o", "messages": messages}
    return litellm.completion(**kwargs)
out = send([{"role": "user", "content": "sort these items: a, b"}])
'''
    assert _rules(src) == set()


def test_ambiguous_wrapper_name_not_detected():
    # two functions share a name -> ambiguous -> dropped, no misattribution.
    src = '''
from openai import OpenAI
client = OpenAI()
def ask(prompt):
    return client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": prompt}])
def ask(prompt, extra):
    return client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": prompt}])
y = ask("sort these names: a, b")
'''
    assert "llm-mechanical" not in _rules(src)


def test_unrelated_same_name_method_not_flagged():
    # a method wrapper is `self.ask`; an unrelated `other.ask(...)` must not match.
    src = '''
from openai import OpenAI
client = OpenAI()
class Bot:
    def ask(self, prompt):
        return client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": prompt}])
z = other.ask("sort these names: a, b")
'''
    assert "llm-mechanical" not in _rules(src)


# --- config allowlist --------------------------------------------------------

def test_allowlist_bare_name():
    src = 'r = ask("extract the email from the text")'
    assert "llm-extraction" in _rules(src, allow=("ask",))


def test_allowlist_dotted_call():
    src = 'r = myapp.llm.ask("sort these names: a, b")'
    assert "llm-mechanical" in _rules(src, allow=("myapp.llm.ask",))


def test_allowlist_imported_by_short_name():
    src = '''
from myapp.llm import ask
r = ask("reverse this string: hello")
'''
    assert "llm-mechanical" in _rules(src, allow=("myapp.llm.ask",))


def test_allowlist_call_in_loop_is_batchable():
    src = '''
for row in rows:
    ask("summarize: " + row)
'''
    assert "llm-in-loop" in _rules(src, allow=("ask",))


def test_allowlist_absent_stays_silent():
    src = 'r = ask("extract the email from the text")'
    assert _rules(src, allow=("other_fn",)) == set()


# --- async concurrency messaging ---------------------------------------------

def test_concurrent_gather_uses_cost_focused_message():
    src = '''
import asyncio
from openai import OpenAI
client = OpenAI()
async def extract(text):
    return await client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "count the words: " + text}])
async def main(docs):
    results = await asyncio.gather(*[extract(d) for d in docs])
'''
    msgs = _messages(src, "llm-in-loop")
    assert msgs and any("concurrent" in m for m in msgs)


def test_plain_loop_keeps_latency_message():
    src = '''
from openai import OpenAI
client = OpenAI()
def ask(prompt):
    return client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": prompt}])
for row in rows:
    ask("summarize: " + row)
'''
    msgs = _messages(src, "llm-in-loop")
    assert msgs and any("N latencies" in m for m in msgs)
