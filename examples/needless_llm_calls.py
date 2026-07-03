"""Examples of LLM calls that plain code should handle. overllm flags every one.

The last function is a legitimate call that overllm leaves alone.
"""

from openai import OpenAI
from litellm import completion
from langchain_openai import ChatOpenAI

client = OpenAI()


def greeting():
    # static-prompt: the input is constant, so the call buys nothing
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Write a short friendly greeting."}],
    )


def get_email(text):
    # llm-extraction: an email is a regex, not a model call
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"Extract the email address from: {text}"}],
    )


def sort_names(names):
    # llm-mechanical: this is sorted()
    llm = ChatOpenAI(model="gpt-4o")
    return llm.invoke(f"Sort these names alphabetically: {names}")


def tag_all(items):
    # llm-in-loop: one API round-trip per item
    tags = []
    for item in items:
        tags.append(
            completion(model="gpt-4o", messages=[{"role": "user", "content": f"Tag this: {item}"}])
        )
    return tags


def draft_reply(message):
    # legitimate: open-ended generation over real input. overllm stays silent.
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a warm, concise support agent."},
            {"role": "user", "content": f"The customer wrote: {message}. Draft a reply."},
        ],
    )
