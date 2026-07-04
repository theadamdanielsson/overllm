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


def test_detects_bedrock_converse():
    src = 'brt.converse(modelId="anthropic.claude-3", messages=[{"role":"user","content":[{"text": x}]}])'
    assert n_calls(src) == 1


def test_detects_cohere_v2_chat():
    src = "co = cohere.ClientV2()\nco.chat(model='command-r', messages=[{'role':'user','content': q}])"
    assert n_calls(src) == 1


def test_detects_huggingface_chat_completion():
    src = "hf = InferenceClient()\nhf.chat_completion(messages=[{'role':'user','content': q}])"
    assert n_calls(src) == 1


def test_detects_replicate_run():
    src = 'replicate.run("meta/llama-3", input={"prompt": "hi"})'
    assert n_calls(src) == 1


def test_cohere_chat_extraction_flagged():
    # new SDK + variable prompt + rule all together
    src = (
        "def f(t):\n"
        "    co = cohere.ClientV2()\n"
        "    msg = f'Extract the email from: {t}'\n"
        "    return co.chat(model='command-r', messages=[{'role':'user','content': msg}])"
    )
    assert "llm-extraction" in rule_set(src)


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


def test_get_weather_on_a_date_not_extraction():
    # regression (found on a real repo): "get the weather ... on a specific date"
    # is not a date-extraction task; the verb and datum are unrelated
    src = 'client.chat.completions.create(model="m", messages=[{"role":"user","content": f"Get the weather for {city} on a specific date"}])'
    assert "llm-extraction" not in rule_set(src)


def test_prompt_held_in_variable_is_resolved():
    # the common real-world pattern: prompt in a variable, not inline
    src = (
        "def get_email(text):\n"
        "    p = f'Extract the email from: {text}'\n"
        "    return client.chat.completions.create(model='m', messages=[{'role':'user','content': p}])"
    )
    assert "llm-extraction" in rule_set(src)


def test_message_dict_held_in_variable_is_resolved():
    src = (
        "def ask(q):\n"
        "    user_msg = {'role':'user','content': f'Sort these alphabetically: {q}'}\n"
        "    return client.chat.completions.create(model='m', messages=[user_msg])"
    )
    assert "llm-mechanical" in rule_set(src)


def test_reassigned_variable_not_resolved():
    # safety: a reassigned variable is ambiguous, so we do not resolve through it
    src = (
        "def f(text):\n"
        "    p = 'Write a haiku about the sea.'\n"
        "    p = build_prompt(text)\n"
        "    return client.chat.completions.create(model='m', messages=[{'role':'user','content': p}])"
    )
    assert rule_set(src) == set()


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


def test_count_to_n_on_lines_not_mechanical():
    # regression (found on a real JS repo): "count from 1 to 5 on separate lines"
    # is output formatting, not counting lines
    src = 'client.chat.completions.create(model="m", messages=[{"role":"user","content":"Count from 1 to 5 on separate lines."}])'
    assert "llm-mechanical" not in rule_set(src)


def test_mechanical_deduplicate_flagged():
    src = 'client.chat.completions.create(model="m", messages=[{"role":"user","content":"Remove duplicates from this list of names."}])'
    assert "llm-mechanical" in rule_set(src)


def test_unique_adjective_not_dedup():
    # regression (found on real repos): "unique" as an adjective -- "unique
    # features", "a unique logo", "makes it unique" -- is not a dedup request.
    for prompt in ("Describe the unique features of this destination.",
                   "Create a unique, professional logo for the brand.",
                   "Explain what makes this place unique."):
        src = f'client.chat.completions.create(model="m", messages=[{{"role":"user","content":{prompt!r}}}])'
        assert "llm-mechanical" not in rule_set(src)


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


def test_retry_range_loop_is_necessary():
    # regression (found on a real repo, build-with-groq/g1): a range() retry loop
    # is not per-item batching
    src = (
        "for attempt in range(3):\n"
        "    client.chat.completions.create(model='m', messages=[{'role':'user','content': prompt}])"
    )
    assert "llm-in-loop" not in rule_set(src)


def test_while_loop_is_necessary():
    src = (
        "while not done:\n"
        "    client.chat.completions.create(model='m', messages=[{'role':'user','content': prompt}])"
    )
    assert "llm-in-loop" not in rule_set(src)


def test_conversation_feedback_loop_is_necessary():
    # a reasoning/agent loop that feeds prior answers back in is not batchable
    src = (
        "messages = []\n"
        "for turn in turns:\n"
        "    messages.append({'role':'user','content': turn})\n"
        "    client.chat.completions.create(model='m', messages=messages)"
    )
    assert "llm-in-loop" not in rule_set(src)


def test_already_batched_loop_is_necessary():
    # regression (found on a real repo, pavelhorak/Prospekt): a loop over batches
    # is already the batched solution; do not tell it to batch
    src = (
        "for batch in batches:\n"
        "    client.chat.completions.create(model='m', messages=[{'role':'user','content': str(batch)}])"
    )
    assert "llm-in-loop" not in rule_set(src)


def test_chunk_loop_is_necessary():
    # regression (found on a real repo, scooter7/carnegieseo): chunking a big
    # document to fit the context window is necessary, not per-item waste
    src = (
        "for chunk in html_chunks:\n"
        "    client.chat.completions.create(model='m', messages=[{'role':'user','content': chunk}])"
    )
    assert "llm-in-loop" not in rule_set(src)


# --- precision: a normal, justified LLM call produces zero findings ----------

def test_normal_dynamic_call_is_silent():
    src = (
        'client.chat.completions.create(model="gpt-4o", messages=['
        '{"role":"system","content":"You are a support agent."},'
        '{"role":"user","content": f"The customer wrote: {message}. Draft a warm reply."}])'
    )
    assert rule_set(src) == set()


# --- R5 prompt-injection -----------------------------------------------------

def test_prompt_injection_from_request_var_flagged():
    src = (
        "def handler():\n"
        "    q = request.args.get('q')\n"
        "    return client.chat.completions.create(model='m', messages=[{'role':'user','content': f'Answer this: {q}'}])"
    )
    assert "prompt-injection" in rule_set(src)


def test_prompt_injection_direct_request_flagged():
    src = (
        "def handler():\n"
        "    return client.chat.completions.create(model='m', messages=[{'role':'user','content': f\"Q: {request.json['q']}\"}])"
    )
    assert "prompt-injection" in rule_set(src)


def test_local_input_is_not_prompt_injection():
    # regression (found scanning 100 repos): local single-user input (input(),
    # CLI args, Streamlit text boxes) is not remote-untrusted; only web requests are
    src = (
        "q = input('ask: ')\n"
        "client.chat.completions.create(model='m', messages=[{'role':'user','content': f'User asked: {q}'}])"
    )
    assert "prompt-injection" not in rule_set(src)


def test_streamlit_input_is_not_prompt_injection():
    src = (
        "q = st.text_area('query')\n"
        "client.chat.completions.create(model='m', messages=[{'role':'user','content': f'Answer: {q}'}])"
    )
    assert "prompt-injection" not in rule_set(src)


def test_trusted_variable_is_not_prompt_injection():
    src = (
        "def summarize(article):\n"
        "    return client.chat.completions.create(model='m', messages=[{'role':'user','content': f'Summarize: {article}'}])"
    )
    assert "prompt-injection" not in rule_set(src)


# --- R6 deprecated-model -----------------------------------------------------

def test_retired_model_flagged_as_error():
    src = 'client.chat.completions.create(model="gpt-3.5-turbo-0301", messages=[])'
    assert "deprecated-model" in rule_set(src)


def test_deprecated_model_flagged():
    src = 'client.messages.create(model="claude-3-haiku-20240307", messages=[])'
    assert "deprecated-model" in rule_set(src)


def test_live_model_not_flagged():
    # exact-match on the id, so live aliases must never trip
    for m in ("gpt-4o", "gpt-4o-mini", "davinci-002", "claude-3-haiku", "claude-opus-4-8"):
        src = f'client.chat.completions.create(model="{m}", messages=[])'
        assert "deprecated-model" not in rule_set(src), m


# --- R7 unsupported-params ---------------------------------------------------

def test_temperature_on_reasoning_model_flagged():
    src = 'client.chat.completions.create(model="o1", messages=[], temperature=0)'
    assert "unsupported-params" in rule_set(src)


def test_sampling_param_on_new_anthropic_flagged():
    src = 'client.messages.create(model="claude-opus-4-8", messages=[], top_p=0.5)'
    assert "unsupported-params" in rule_set(src)


def test_temperature_on_sampling_model_not_flagged():
    # opus 4.6 and gpt-4o still accept sampling params
    for m in ("gpt-4o", "claude-opus-4-6"):
        src = f'client.chat.completions.create(model="{m}", messages=[], temperature=0.3)'
        assert "unsupported-params" not in rule_set(src), m


def test_reasoning_model_without_sampling_params_not_flagged():
    src = 'client.chat.completions.create(model="o1", messages=[])'
    assert "unsupported-params" not in rule_set(src)
