"""Detect LLM/AI API calls in a Python AST and extract their prompt.

Deterministic and conservative: a call is only treated as an LLM call when it
matches a known SDK signature (or a raw HTTP request to a known LLM host). The
known-SDK surface is the precision anchor - we would rather miss an exotic
wrapper than false-positive on an unrelated `.create()`.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

# --- Known call surfaces -----------------------------------------------------

# Method-call suffixes (trailing attribute names). Matched against the tail of
# the call's attribute chain, so the receiver variable name does not matter:
# `client.chat.completions.create`, `self.oai.chat.completions.create`, etc.
LLM_METHOD_SUFFIXES = (
    ("chat", "completions", "create"),
    ("chat", "completions", "parse"),
    ("completions", "create"),
    ("responses", "create"),
    ("responses", "parse"),
    ("messages", "create"),
    ("messages", "stream"),
    ("chat", "create"),
    ("chat", "complete"),
    ("models", "generate_content"),
    ("generate_content",),
)

# Legacy module-level surfaces: openai.ChatCompletion.create, Completion.create.
LLM_LEGACY_SUFFIXES = (
    ("ChatCompletion", "create"),
    ("ChatCompletion", "acreate"),
    ("Completion", "create"),
    ("Completion", "acreate"),
)

# Fully-qualified free/dotted calls.
LLM_DOTTED_FQNS = {
    "litellm.completion",
    "litellm.acompletion",
    "ollama.chat",
    "ollama.generate",
}

# Bare free functions, only when imported from one of these modules.
LLM_FREE_FUNCS = {
    "completion": {"litellm"},
    "acompletion": {"litellm"},
}

# Constructors that produce an LLM/chat object (mostly LangChain + SDK clients).
LLM_CONSTRUCTORS = {
    "ChatOpenAI", "AzureChatOpenAI", "OpenAI", "AzureOpenAI", "AsyncOpenAI",
    "ChatAnthropic", "Anthropic", "AsyncAnthropic", "AnthropicVertex",
    "ChatGoogleGenerativeAI", "GenerativeModel", "ChatVertexAI",
    "ChatMistralAI", "MistralClient", "Mistral",
    "ChatCohere", "ChatGroq", "ChatOllama", "OllamaLLM", "ChatBedrock",
    "ChatLiteLLM", "LlamaCpp", "HuggingFaceHub", "LLMChain",
}

# Methods on a tracked LLM object that count as a call.
LLM_OBJ_METHODS = {
    "invoke", "ainvoke", "stream", "astream", "batch", "abatch",
    "predict", "apredict", "generate", "agenerate", "complete", "acomplete",
    "run", "call", "generate_content",
}

# Hosts that identify a raw-HTTP LLM request.
LLM_HOSTS = (
    "api.openai.com", "api.anthropic.com", "generativelanguage.googleapis.com",
    "api.mistral.ai", "api.cohere.ai", "api.cohere.com", "api.groq.com",
    "openrouter.ai", "api.together.xyz", "api.deepseek.com", "api.perplexity.ai",
    "api.x.ai", ":11434",  # ollama default port
)

# Kwargs that all LLM chat/completion calls carry - used as a precision guard so
# an unrelated `.messages.create()` (e.g. an ORM) is not mistaken for an LLM call.
LLM_KWARGS = {"model", "messages", "prompt", "input", "contents"}

LOOP_NODES = (ast.For, ast.AsyncFor, ast.comprehension)
FUNC_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)


@dataclass
class LLMCall:
    node: ast.Call
    line: int
    col: int
    api: str                      # openai_like | anthropic | google | positional | ollama_chat | ollama_generate | http
    prompt_text: str = ""         # lowercased visible literal text of the user prompt
    prompt_static: bool = False   # the user prompt is a compile-time constant
    prompt_resolved: bool = False # we could locate and read the user prompt
    in_loop: bool = False
    tainted: bool = False         # untrusted external input flows into the prompt
    snippet: str = ""


# --- AST helpers -------------------------------------------------------------

def _trailing_attrs(func: ast.expr) -> tuple[ast.expr, tuple[str, ...]]:
    attrs: list[str] = []
    node = func
    while isinstance(node, ast.Attribute):
        attrs.append(node.attr)
        node = node.value
    attrs.reverse()
    return node, tuple(attrs)


def _dotted(node: ast.expr) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        parts.reverse()
        return ".".join(parts)
    return None


def _ctor_name(call: ast.Call) -> str | None:
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _kw(call: ast.Call, name: str) -> ast.expr | None:
    for k in call.keywords:
        if k.arg == name:
            return k.value
    return None


def _has_llm_kwarg(call: ast.Call) -> bool:
    return any(k.arg in LLM_KWARGS for k in call.keywords)


def _const_str(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _dict_get(d: ast.Dict, key: str) -> ast.expr | None:
    for k, v in zip(d.keys, d.values):
        if _const_str(k) == key:
            return v
    return None


def _literal_text_and_static(node: ast.expr) -> tuple[str, bool]:
    """Collect the visible literal text of an expression and whether it is fully static.

    Handles string literals, f-strings, `+` concatenation, `.format()` and `%`
    templates, and lists/tuples of the above. Any variable, call, or interpolated
    value makes it non-static (but we still keep literal siblings for keyword
    matching).
    """
    texts: list[str] = []
    static = True

    def walk(n: ast.expr) -> None:
        nonlocal static
        if isinstance(n, ast.Constant):
            if isinstance(n.value, str):
                texts.append(n.value)
        elif isinstance(n, ast.JoinedStr):
            for v in n.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    texts.append(v.value)
                elif isinstance(v, ast.FormattedValue):
                    static = False
                else:
                    walk(v)
        elif isinstance(n, ast.BinOp):
            if isinstance(n.op, ast.Add):
                walk(n.left)
                walk(n.right)
            elif isinstance(n.op, ast.Mod):
                walk(n.left)  # the template string
                static = False
            else:
                static = False
        elif isinstance(n, (ast.List, ast.Tuple)):
            for e in n.elts:
                walk(e)
        elif isinstance(n, ast.Call):
            f = n.func
            if (
                isinstance(f, ast.Attribute)
                and f.attr in ("format", "join", "strip")
                and isinstance(f.value, ast.Constant)
                and isinstance(f.value.value, str)
            ):
                texts.append(f.value.value)
            static = False
        else:
            static = False

    walk(node)
    return " ".join(texts).lower().strip(), static


def _messages_user_prompt(msgs: ast.expr) -> tuple[list[ast.expr], bool]:
    """Extract the `content` of user-role messages from a `messages=[...]` list.

    Returns (content_nodes, resolved). resolved is False when the messages list
    is a variable or otherwise not a static list of dict literals with constant
    roles - in which case we stay silent rather than guess.
    """
    if not isinstance(msgs, ast.List):
        return [], False
    contents: list[ast.expr] = []
    for el in msgs.elts:
        if not isinstance(el, ast.Dict):
            return [], False
        role = _const_str(_dict_get(el, "role"))
        content = _dict_get(el, "content")
        if role is None:
            return [], False
        if role == "user" and content is not None:
            contents.append(content)
    return contents, True


# --- Prompt extraction per API shape ----------------------------------------

def _prompt_nodes(call: ast.Call, api: str) -> tuple[list[ast.expr], bool]:
    """Return (user-prompt expression nodes, resolved) for a detected LLM call."""
    nodes: list[ast.expr] = []
    resolved = True

    if api in ("openai_like", "anthropic", "ollama_chat"):
        messages = _kw(call, "messages")
        if messages is not None:
            nodes, resolved = _messages_user_prompt(messages)
        else:
            single = _kw(call, "prompt") or _kw(call, "input")
            if single is None and call.args:
                single = call.args[0]
            if single is not None:
                nodes = [single]
            else:
                resolved = False
    elif api == "ollama_generate":
        single = _kw(call, "prompt") or (call.args[0] if call.args else None)
        nodes, resolved = ([single], True) if single is not None else ([], False)
    elif api == "google":
        single = _kw(call, "contents") or (call.args[0] if call.args else None)
        nodes, resolved = ([single], True) if single is not None else ([], False)
    elif api == "positional":
        single = _kw(call, "input") or (call.args[0] if call.args else None)
        nodes, resolved = ([single], True) if single is not None else ([], False)
    else:  # http / generic - prompt not statically inspectable
        return [], False

    return nodes, resolved


def _extract_prompt(call: ast.Call, api: str) -> tuple[str, bool, bool]:
    """Return (prompt_text_lower, is_static, resolved) for a detected LLM call."""
    nodes, resolved = _prompt_nodes(call, api)
    if not resolved or not nodes:
        return "", False, resolved and bool(nodes)

    texts = []
    static = True
    for n in nodes:
        t, s = _literal_text_and_static(n)
        if t:
            texts.append(t)
        static = static and s
    return " ".join(texts).strip(), static, True


# --- Taint: untrusted external input flowing into a prompt --------------------

UNTRUSTED_ROOTS = ("request", "flask")  # web request objects
UNTRUSTED_INPUT_CALLS = {"input"}       # builtins input()
UNTRUSTED_WIDGET_METHODS = {"text_input", "text_area", "chat_input"}  # streamlit etc.


def _expr_is_untrusted(node: ast.expr) -> bool:
    """Does this expression read from a clearly untrusted external source?

    Conservative on purpose: web request objects, input(), sys.argv, and a few
    known input-widget calls. High precision beats broad coverage.
    """
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            f = sub.func
            if isinstance(f, ast.Name) and f.id in UNTRUSTED_INPUT_CALLS:
                return True
            if isinstance(f, ast.Attribute) and f.attr in UNTRUSTED_WIDGET_METHODS:
                return True
        base = None
        if isinstance(sub, ast.Attribute):
            base = _dotted(sub)
        elif isinstance(sub, ast.Subscript):
            base = _dotted(sub.value)
        if base:
            parts = base.split(".")
            if parts[0] in UNTRUSTED_ROOTS:
                return True
            if parts[0] == "sys" and "argv" in parts:
                return True
    return False


def _tainted_vars(scope: ast.AST) -> set[str]:
    tainted: set[str] = set()
    for n in ast.walk(scope):
        if isinstance(n, (ast.Assign, ast.AnnAssign)) and n.value is not None:
            if _expr_is_untrusted(n.value):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                for t in targets:
                    if isinstance(t, ast.Name):
                        tainted.add(t.id)
    return tainted


def _prompt_is_tainted(nodes: list[ast.expr], tainted: set[str]) -> bool:
    for pn in nodes:
        if _expr_is_untrusted(pn):
            return True
        for sub in ast.walk(pn):
            if isinstance(sub, ast.Name) and sub.id in tainted:
                return True
    return False


# --- Import / variable context ----------------------------------------------

class _Context:
    def __init__(self) -> None:
        self.free_llm_names: set[str] = set()   # bare names that are LLM free funcs
        self.llm_vars: set[str] = set()         # dotted names bound to an LLM object

    def scan(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    fn = alias.name
                    mods = LLM_FREE_FUNCS.get(fn)
                    if mods and node.module.split(".")[0] in mods:
                        self.free_llm_names.add(alias.asname or fn)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value
                if isinstance(value, ast.Call) and _ctor_name(value) in LLM_CONSTRUCTORS:
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for t in targets:
                        d = _dotted(t)
                        if d:
                            self.llm_vars.add(d)


# --- Classification ----------------------------------------------------------

_SUFFIX_API = {
    ("chat", "completions", "create"): "openai_like",
    ("chat", "completions", "parse"): "openai_like",
    ("completions", "create"): "openai_like",
    ("responses", "create"): "openai_like",
    ("responses", "parse"): "openai_like",
    ("messages", "create"): "anthropic",
    ("messages", "stream"): "anthropic",
    ("chat", "create"): "openai_like",
    ("chat", "complete"): "openai_like",
    ("models", "generate_content"): "google",
    ("generate_content",): "google",
}


def _classify(call: ast.Call, ctx: _Context) -> str | None:
    base, attrs = _trailing_attrs(call.func)

    # 1. Known method suffixes (need an LLM-shaped kwarg as a precision guard).
    for suf in LLM_METHOD_SUFFIXES:
        if len(attrs) >= len(suf) and attrs[-len(suf):] == suf:
            if suf == ("generate_content",) or _has_llm_kwarg(call) or call.args:
                return _SUFFIX_API[suf]

    # 2. Legacy module-level completion APIs.
    for suf in LLM_LEGACY_SUFFIXES:
        if len(attrs) >= len(suf) and attrs[-len(suf):] == suf:
            return "openai_like"

    # 3. LangChain / SDK object methods on a tracked LLM variable.
    if attrs and attrs[-1] in LLM_OBJ_METHODS and isinstance(call.func, ast.Attribute):
        recv = _dotted(call.func.value)
        if recv in ctx.llm_vars:
            return "positional"

    # 4. Bare free functions imported from a known LLM module.
    if isinstance(call.func, ast.Name) and call.func.id in ctx.free_llm_names:
        return "openai_like"

    # 5. Fully-qualified free/dotted calls.
    d = _dotted(call.func)
    if d in LLM_DOTTED_FQNS:
        if d == "ollama.generate":
            return "ollama_generate"
        if d == "ollama.chat":
            return "ollama_chat"
        return "openai_like"

    # 6. Raw HTTP to a known LLM host.
    if attrs and attrs[-1] == "post" and _has_llm_host_arg(call):
        return "http"

    return None


def _has_llm_host_arg(call: ast.Call) -> bool:
    strings: list[str] = []
    for a in call.args:
        s = _const_str(a)
        if s:
            strings.append(s)
    for k in call.keywords:
        s = _const_str(k.value)
        if s:
            strings.append(s)
    joined = " ".join(strings).lower()
    return any(host in joined for host in LLM_HOSTS)


# --- Loop ancestry -----------------------------------------------------------

def _build_parents(tree: ast.AST) -> dict[int, ast.AST]:
    parents: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    return parents


def _child_field(parent: ast.AST, child: ast.AST) -> str | None:
    for field, value in ast.iter_fields(parent):
        if value is child:
            return field
        if isinstance(value, list) and any(item is child for item in value):
            return field
    return None


def _in_loop(node: ast.AST, parents: dict[int, ast.AST]) -> bool:
    # True only if the call runs once per iteration. A call that is the loop's
    # iterable (e.g. `async for chunk in create(..., stream=True)`) runs once and
    # must NOT count -- that is streaming, not N calls.
    child = node
    while True:
        cur = parents.get(id(child))
        if cur is None or isinstance(cur, FUNC_NODES):
            return False
        if isinstance(cur, (ast.For, ast.AsyncFor)):
            if _child_field(cur, child) in ("body", "orelse"):
                return True
        elif isinstance(cur, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            if _child_field(cur, child) in ("elt", "key", "value"):
                return True
        elif isinstance(cur, ast.comprehension):
            if _child_field(cur, child) == "ifs":
                return True
        child = cur


def _enclosing_scope(node: ast.AST, parents: dict[int, ast.AST]) -> ast.AST | None:
    cur = parents.get(id(node))
    while cur is not None:
        if isinstance(cur, FUNC_NODES):
            return cur
        cur = parents.get(id(cur))
    return None


# --- Entry point -------------------------------------------------------------

def find_llm_calls(tree: ast.AST, source_lines: list[str]) -> list[LLMCall]:
    ctx = _Context()
    ctx.scan(tree)
    parents = _build_parents(tree)
    taint_cache: dict[int, set[str]] = {}

    calls: list[LLMCall] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        api = _classify(node, ctx)
        if api is None:
            continue
        text, static, resolved = _extract_prompt(node, api)

        scope = _enclosing_scope(node, parents) or tree
        sid = id(scope)
        if sid not in taint_cache:
            taint_cache[sid] = _tainted_vars(scope)
        p_nodes, _ = _prompt_nodes(node, api)
        tainted = _prompt_is_tainted(p_nodes, taint_cache[sid])

        line = getattr(node, "lineno", 0)
        snippet = source_lines[line - 1].strip() if 0 < line <= len(source_lines) else ""
        calls.append(
            LLMCall(
                node=node,
                line=line,
                col=getattr(node, "col_offset", 0),
                api=api,
                prompt_text=text,
                prompt_static=static,
                prompt_resolved=resolved,
                in_loop=_in_loop(node, parents),
                tainted=tainted,
                snippet=snippet,
            )
        )
    return calls
