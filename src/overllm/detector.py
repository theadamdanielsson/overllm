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
    ("converse",),           # AWS Bedrock converse API
    ("converse_stream",),
    ("embeddings", "create"),  # an embedding call: no NL prompt, but batchable in a loop
)

# Unambiguous chains: no non-LLM API exposes these, so they need no kwarg guard --
# this catches the common `client.chat.completions.create(**params)` splat form
# that hides the model/messages inside a dict.
_ALWAYS_MATCH = frozenset({
    ("chat", "completions", "create"), ("chat", "completions", "parse"),
    ("responses", "create"), ("responses", "parse"), ("generate_content",),
})

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
    "replicate.run",
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
    "InferenceClient", "ClientV2",  # HuggingFace, Cohere v2
}

# Methods on a tracked LLM object that count as a call.
LLM_OBJ_METHODS = {
    "invoke", "ainvoke", "stream", "astream", "batch", "abatch",
    "predict", "apredict", "generate", "agenerate", "complete", "acomplete",
    "run", "call", "generate_content",
    "chat", "chat_completion", "chat_stream", "text_generation",  # cohere / HF
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
    api: str                      # openai_like | anthropic | google | positional | ollama_chat | ollama_generate | http | wrapper
    prompt_text: str = ""         # lowercased visible literal text of the user prompt
    prompt_static: bool = False   # the user prompt is a compile-time constant
    prompt_resolved: bool = False # we could locate and read the user prompt
    loop_kind: str | None = None  # "batchable" | "necessary" | None (not in a loop)
    loop_concurrent: bool = False # a batchable loop dispatched via asyncio.gather/as_completed
    tainted: bool = False         # untrusted external input flows into the prompt
    snippet: str = ""
    model: str | None = None      # the model id, when it is a plain string literal
    params: frozenset = field(default_factory=frozenset)  # keyword arg names on the call


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


def _literal_text_and_static(node: ast.expr, resolve=None) -> tuple[str, bool]:
    """Collect the visible literal text of an expression and whether it is fully static.

    Handles string literals, f-strings, `+` concatenation, `.format()` and `%`
    templates, and lists/tuples of the above. Any variable, call, or interpolated
    value makes it non-static (but we still keep literal siblings for keyword
    matching).
    """
    texts: list[str] = []
    static = True

    def walk(n: ast.expr, depth: int = 0) -> None:
        nonlocal static
        if depth > 120:  # a pathological chain (thousands of `+` terms) -> bail, not crash
            static = False
            return
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
                    walk(v, depth + 1)
        elif isinstance(n, ast.BinOp):
            if isinstance(n.op, ast.Add):
                walk(n.left, depth + 1)
                walk(n.right, depth + 1)
            elif isinstance(n.op, ast.Mod):
                walk(n.left, depth + 1)  # the template string
                static = False
            else:
                static = False
        elif isinstance(n, (ast.List, ast.Tuple)):
            for e in n.elts:
                walk(e, depth + 1)
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
        elif isinstance(n, ast.Name) and resolve is not None:
            r = resolve(n)
            if isinstance(r, ast.Name):  # still unresolved -> a runtime value
                static = False
            else:
                walk(r, depth + 1)
        else:
            static = False

    walk(node)
    return " ".join(texts).lower().strip(), static


def _single_assignments(scope: ast.AST) -> dict[str, ast.expr]:
    """Names in a scope bound by exactly one plain `name = <expr>` assignment.

    Names that are reassigned, augmented, or used as loop targets are excluded, so
    resolving through them is safe. This lets us read prompts held in a variable
    (`p = f"..."; create(messages=[{"content": p}])`), which is how most real code
    is written, instead of only inline literals.
    """
    assigned: dict[str, ast.expr] = {}
    unsafe: set[str] = set()
    for n in ast.walk(scope):
        if isinstance(n, ast.Assign):
            if len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                name = n.targets[0].id
                if name in assigned:
                    unsafe.add(name)
                assigned[name] = n.value
            else:
                for t in n.targets:
                    for sub in ast.walk(t):
                        if isinstance(sub, ast.Name):
                            unsafe.add(sub.id)
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.value is not None:
            if n.target.id in assigned:
                unsafe.add(n.target.id)
            assigned[n.target.id] = n.value
        elif isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name):
            unsafe.add(n.target.id)
        elif isinstance(n, (ast.For, ast.AsyncFor)) and isinstance(n.target, ast.Name):
            unsafe.add(n.target.id)
    return {k: v for k, v in assigned.items() if k not in unsafe}


def _module_assignments(tree: ast.AST) -> dict[str, ast.expr]:
    """Top-level module constants only (`NAME = <expr>` directly in the module
    body). Deliberately NOT function-locals, so resolving a function's unresolved
    name against this can never pick up a different function's local of the same
    name -- only a genuine module-level constant (a `prompts.py` pattern)."""
    assigned: dict[str, ast.expr] = {}
    unsafe: set[str] = set()
    for n in getattr(tree, "body", []):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            name = n.targets[0].id
            if name in assigned:
                unsafe.add(name)
            assigned[name] = n.value
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.value is not None:
            if n.target.id in assigned:
                unsafe.add(n.target.id)
            assigned[n.target.id] = n.value
    return {k: v for k, v in assigned.items() if k not in unsafe}


def _resolve(node: ast.expr, assignments: dict[str, ast.expr]) -> ast.expr:
    seen: set[str] = set()
    while isinstance(node, ast.Name) and node.id in assignments and node.id not in seen:
        seen.add(node.id)
        node = assignments[node.id]
    return node


def _messages_user_prompt(msgs: ast.expr, resolve) -> tuple[list[ast.expr], bool]:
    """Extract the `content` of user-role messages from a `messages=[...]` list.

    Returns (content_nodes, resolved). resolved is False when the list (after
    variable resolution) is not a list of dict literals with constant roles - in
    which case we stay silent rather than guess.
    """
    msgs = resolve(msgs)
    if not isinstance(msgs, ast.List):
        return [], False
    contents: list[ast.expr] = []
    for el in msgs.elts:
        el = resolve(el)
        if not isinstance(el, ast.Dict):
            return [], False
        role = _const_str(_dict_get(el, "role"))
        content = _dict_get(el, "content")
        if role is None:
            return [], False
        if role == "user" and content is not None:
            contents.append(resolve(content))
    return contents, True


# --- Prompt extraction per API shape ----------------------------------------

def _prompt_nodes(call: ast.Call, api: str, resolve) -> tuple[list[ast.expr], bool]:
    """Return (user-prompt expression nodes, resolved) for a detected LLM call.

    `resolve` follows a variable one or more hops to its assignment, so a prompt
    held in a variable is read like an inline one.
    """
    nodes: list[ast.expr] = []
    resolved = True

    if api in ("openai_like", "anthropic", "ollama_chat"):
        messages = _kw(call, "messages")
        if messages is not None:
            nodes, resolved = _messages_user_prompt(messages, resolve)
        else:
            single = _kw(call, "prompt") or _kw(call, "input")
            if single is None and call.args:
                single = call.args[0]
            if single is not None:
                nodes = [resolve(single)]
            else:
                resolved = False
    elif api == "ollama_generate":
        single = _kw(call, "prompt") or (call.args[0] if call.args else None)
        nodes, resolved = ([resolve(single)], True) if single is not None else ([], False)
    elif api == "google":
        single = _kw(call, "contents") or (call.args[0] if call.args else None)
        nodes, resolved = ([resolve(single)], True) if single is not None else ([], False)
    elif api == "positional":
        single = _kw(call, "input") or (call.args[0] if call.args else None)
        nodes, resolved = ([resolve(single)], True) if single is not None else ([], False)
    else:  # http / generic - prompt not statically inspectable
        return [], False

    return nodes, resolved


def _extract_prompt(call: ast.Call, api: str, resolve) -> tuple[str, bool, bool]:
    """Return (prompt_text_lower, is_static, resolved) for a detected LLM call."""
    nodes, resolved = _prompt_nodes(call, api, resolve)
    if not resolved or not nodes:
        return "", False, resolved and bool(nodes)

    texts = []
    static = True
    for n in nodes:
        t, s = _literal_text_and_static(n, resolve)
        if t:
            texts.append(t)
        static = static and s
    return " ".join(texts).strip(), static, True


# --- Taint: untrusted external input flowing into a prompt --------------------

UNTRUSTED_ROOTS = ("request", "flask")  # flask / django / fastapi web request objects


def _expr_is_untrusted(node: ast.expr) -> bool:
    """Does this expression read from a remote, untrusted web request?

    Deliberately web-only (request.args / json / form / query_params, flask.request).
    A scan of 100 repos showed that counting local single-user input -- input(),
    sys.argv, Streamlit text boxes -- as untrusted fires on normal apps, because the
    operator is not attacking themselves. Prompt injection that matters is remote.
    """
    for sub in ast.walk(node):
        base = None
        if isinstance(sub, ast.Attribute):
            base = _dotted(sub)
        elif isinstance(sub, ast.Subscript):
            base = _dotted(sub.value)
        if base and base.split(".")[0] in UNTRUSTED_ROOTS:
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
    ("converse",): "anthropic",
    ("converse_stream",): "anthropic",
    ("embeddings", "create"): "embeddings",
}


def _classify(call: ast.Call, ctx: _Context) -> str | None:
    base, attrs = _trailing_attrs(call.func)

    # 1. Known method suffixes. Ambiguous ones need an LLM-shaped kwarg (or a
    #    positional arg) as a precision guard; the unambiguous chains in
    #    _ALWAYS_MATCH do not, so a `**kwargs` splat is still caught.
    for suf in LLM_METHOD_SUFFIXES:
        if len(attrs) >= len(suf) and attrs[-len(suf):] == suf:
            if suf in _ALWAYS_MATCH or _has_llm_kwarg(call) or call.args:
                return _SUFFIX_API[suf]

    # 2. Legacy module-level completion APIs.
    for suf in LLM_LEGACY_SUFFIXES:
        if len(attrs) >= len(suf) and attrs[-len(suf):] == suf:
            return "openai_like"

    # 3. LangChain / SDK object methods on a tracked LLM variable.
    if attrs and attrs[-1] in LLM_OBJ_METHODS and isinstance(call.func, ast.Attribute):
        recv = _dotted(call.func.value)
        if recv in ctx.llm_vars:
            return "openai_like" if _kw(call, "messages") is not None else "positional"

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
        if d == "replicate.run":
            return "generic"  # prompt is nested in input={...}; detect but do not extract
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


# --- Wrapper functions: a project's own LLM abstraction ----------------------
# Real code rarely calls the SDK inline; it wraps it (`def ask(prompt): ...
# client.chat.completions.create(...)`). Detect a function whose body makes a
# known LLM call fed by one of the function's own parameters, then treat calls
# to that function as LLM calls whose prompt is the mapped argument. Guards that
# keep precision: same file only, the name must be unambiguous, and a method
# wrapper matches `self.<name>(...)` alone -- so an unrelated `run()`/`ask()` is
# never mistaken for an LLM call.

@dataclass
class _Wrapper:
    name: str
    inner_call: ast.Call  # the SDK call inside the wrapper body
    api: str
    param_index: dict     # param_name -> call-site positional index (-1 = kw-only)
    is_method: bool


def _callable_param_index(fn: ast.AST) -> tuple[dict, bool]:
    """Map each param to the positional index it has AT A CALL SITE (self stripped)."""
    params = list(getattr(fn.args, "posonlyargs", [])) + list(fn.args.args)
    names = [a.arg for a in params]
    is_method = bool(names) and names[0] in ("self", "cls")
    if is_method:
        names = names[1:]
    index = {n: i for i, n in enumerate(names)}
    for a in fn.args.kwonlyargs:  # keyword-only params: reachable by name only
        index.setdefault(a.arg, -1)
    return index, is_method


def _slot_value_nodes(call: ast.Call, api: str) -> list:
    """Raw expression(s) sitting in the call's prompt slot, before extraction."""
    vals: list = []
    for name in ("messages", "prompt", "input", "contents"):
        v = _kw(call, name)
        if v is not None:
            vals.append(v)
    if not vals and call.args:
        vals.append(call.args[0])
    return vals


def _find_wrappers(tree: ast.AST, ctx: "_Context") -> dict:
    found: dict = {}
    ambiguous: set = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, FUNC_NODES):
            continue
        index, is_method = _callable_param_index(fn)
        if not index:
            continue
        for call in ast.walk(fn):
            if not isinstance(call, ast.Call):
                continue
            api = _classify(call, ctx)
            if api is None or api in ("http", "generic"):
                continue
            # register only if one of the function's params feeds the prompt slot
            # (a `**kwargs` splat has no literal slot, so those stay unhandled -- a
            # deliberate precision choice, not an oversight).
            dep = any(
                isinstance(sub, ast.Name) and sub.id in index
                for v in _slot_value_nodes(call, api)
                for sub in ast.walk(v)
            )
            if dep:
                if fn.name in found:
                    ambiguous.add(fn.name)
                found[fn.name] = _Wrapper(fn.name, call, api, index, is_method)
            break  # the function's first LLM call defines it as a wrapper
    for name in ambiguous:
        found.pop(name, None)
    return found


def _match_wrapper_call(call: ast.Call, wrappers: dict):
    f = call.func
    if isinstance(f, ast.Name):
        w = wrappers.get(f.id)
        return w if (w is not None and not w.is_method) else None
    if isinstance(f, ast.Attribute):
        w = wrappers.get(f.attr)
        if w is not None and w.is_method and isinstance(f.value, ast.Name) and f.value.id in ("self", "cls"):
            return w
    return None


# --- Config allowlist: user-declared LLM call surface ------------------------
# A team's own abstraction (`myapp.llm.ask(...)`, a provider method) is invisible
# to the signature whitelist, especially across files. `[tool.overllm] llm_calls`
# lets the user declare it by name; those calls are then treated as LLM calls and
# their arguments read for a prompt. Because the user opted in, matching by the
# trailing name is acceptable -- they own the precision trade-off.

def _callable_name(call: ast.Call) -> tuple[str | None, str | None]:
    f = call.func
    if isinstance(f, ast.Name):
        return f.id, None
    if isinstance(f, ast.Attribute):
        return f.attr, _dotted(f)
    return None, None


def _match_allowlist(call: ast.Call, allow: tuple) -> bool:
    if not allow:
        return False
    name, dotted = _callable_name(call)
    if name is None:
        return False
    for entry in allow:
        if "." in entry:
            if dotted == entry or (dotted and dotted.endswith("." + entry)):
                return True
            if isinstance(call.func, ast.Name) and name == entry.rsplit(".", 1)[-1]:
                return True  # imported into scope by its short name
        elif name == entry:
            return True
    return False


_CONCURRENT_FUNCS = {"gather", "as_completed", "wait"}


def _is_concurrent_dispatch(node: ast.AST, parents: dict) -> bool:
    """True when the batchable call sits in a comprehension handed to
    asyncio.gather / as_completed -- N calls, but latency is amortized."""
    child = node
    cur = parents.get(id(child))
    while cur is not None and not isinstance(cur, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
        if isinstance(cur, FUNC_NODES):
            return False
        child = cur
        cur = parents.get(id(child))
    if cur is None:
        return False
    child = cur  # the comprehension
    cur = parents.get(id(child))
    for _ in range(4):
        if cur is None:
            return False
        if isinstance(cur, ast.Call):
            return _callable_name(cur)[0] in _CONCURRENT_FUNCS
        if isinstance(cur, (ast.Starred, ast.List, ast.Tuple, ast.keyword)):
            child = cur
            cur = parents.get(id(child))
            continue
        return False
    return False


def _wrapper_extract(callsite: ast.Call, w: "_Wrapper", callsite_resolve) -> tuple[str, bool, bool, list]:
    """Inline the call-site arguments into the wrapper's inner call, then run the
    normal prompt extraction. This handles both a raw string/content param and a
    whole `messages=`/`prompt=` param passed straight through."""
    subst: dict = {}
    for name, idx in w.param_index.items():
        arg = _kw(callsite, name)
        if arg is None and 0 <= idx < len(callsite.args):
            arg = callsite.args[idx]
        if arg is not None:
            subst[name] = callsite_resolve(arg)

    def resolve(n):
        if isinstance(n, ast.Name) and n.id in subst:
            return subst[n.id]
        return callsite_resolve(n)

    text, static, resolved = _extract_prompt(w.inner_call, w.api, resolve)
    p_nodes, _ = _prompt_nodes(w.inner_call, w.api, resolve)
    return text, static, resolved, p_nodes


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


def _enclosing_loop(node: ast.AST, parents: dict[int, ast.AST]) -> ast.AST | None:
    # Innermost loop whose BODY contains the call (runs once per iteration). A call
    # that is the loop's iterable (`async for chunk in create(stream=True)`) runs
    # once and is not counted -- that is streaming, not N calls.
    child = node
    while True:
        cur = parents.get(id(child))
        if cur is None or isinstance(cur, FUNC_NODES):
            return None
        if isinstance(cur, (ast.For, ast.AsyncFor, ast.While)):
            if _child_field(cur, child) in ("body", "orelse"):
                return cur
        elif isinstance(cur, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            if _child_field(cur, child) in ("elt", "key", "value"):
                return cur
        child = cur


def _loop_has_feedback(loop: ast.AST, call: ast.Call) -> bool:
    # Conversation / agent loop: the messages list passed to the call is mutated
    # (appended to) inside the loop, so each call depends on the previous answer.
    msgs = _kw(call, "messages")
    if not isinstance(msgs, ast.Name):
        return False
    name = msgs.id
    for n in ast.walk(loop):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            if n.func.attr in ("append", "extend", "insert") and isinstance(n.func.value, ast.Name):
                if n.func.value.id == name:
                    return True
        if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name) and n.target.id == name:
            return True
    return False


def _looks_batched(loop: ast.AST) -> bool:
    # The loop already iterates over batches/chunks/groups, so it is not one
    # call per item -- flagging it would tell already-batched code to batch.
    names: list[str] = []
    for part in (loop.target, loop.iter):
        for n in ast.walk(part):
            if isinstance(n, ast.Name):
                names.append(n.id)
    low = " ".join(names).lower()
    return any(k in low for k in ("batch", "chunk", "group"))


def _loop_kind(node: ast.Call, parents: dict[int, ast.AST]) -> str | None:
    """Classify the enclosing loop.

    batchable -> a real per-item map (`for row in rows: llm(row)`); N calls that
                 could be batched, cached, or replaced. This is a savings finding.
    necessary -> a while loop, a `range(...)` retry/fixed-count loop, or a
                 conversation loop that feeds prior answers back in. Not waste.
    """
    loop = _enclosing_loop(node, parents)
    if loop is None:
        return None
    if isinstance(loop, ast.While):
        return "necessary"
    if isinstance(loop, (ast.For, ast.AsyncFor)):
        it = loop.iter
        if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range":
            return "necessary"
        if _looks_batched(loop) or _loop_has_feedback(loop, node):
            return "necessary"
        return "batchable"
    return "batchable"  # comprehension


def _enclosing_scope(node: ast.AST, parents: dict[int, ast.AST]) -> ast.AST | None:
    cur = parents.get(id(node))
    while cur is not None:
        if isinstance(cur, FUNC_NODES):
            return cur
        cur = parents.get(id(cur))
    return None


# --- Entry point -------------------------------------------------------------

def find_llm_calls(tree: ast.AST, source_lines: list[str], allow: tuple = ()) -> list[LLMCall]:
    ctx = _Context()
    ctx.scan(tree)
    wrappers = _find_wrappers(tree, ctx)
    module_assignments = _module_assignments(tree)
    parents = _build_parents(tree)
    taint_cache: dict[int, set[str]] = {}
    assign_cache: dict[int, dict[str, ast.expr]] = {}

    calls: list[LLMCall] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        api = _classify(node, ctx)
        wrapper = None
        if api is None:
            wrapper = _match_wrapper_call(node, wrappers)
            if wrapper is not None:
                api = "wrapper"
            elif _match_allowlist(node, allow):
                # user-declared surface: read the prompt from its own args like a
                # normal call (messages=/prompt=/contents=/first positional).
                api = "google" if _kw(node, "contents") is not None else "openai_like"
            else:
                continue

        scope = _enclosing_scope(node, parents) or tree
        sid = id(scope)
        if sid not in assign_cache:
            assign_cache[sid] = _single_assignments(scope)

        def resolve(n, _fn=assign_cache[sid], _mod=module_assignments):
            r = _resolve(n, _fn)
            if isinstance(r, ast.Name):  # unresolved in-function -> try module constants
                r = _resolve(r, _mod)
            return r

        if wrapper is not None:
            # A call to the project's own wrapper: inline the call-site args into
            # the wrapper's inner SDK call, resolved in the CALL SITE's scope.
            # Model/params live inside the wrapper (already covered at its def).
            text, static, resolved, p_nodes = _wrapper_extract(node, wrapper, resolve)
            model = None
            params: frozenset = frozenset()
        else:
            text, static, resolved = _extract_prompt(node, api, resolve)
            p_nodes, _ = _prompt_nodes(node, api, resolve)
            model_kw = _kw(node, "model")
            m = _const_str(resolve(model_kw)) if model_kw is not None else None
            model = m.strip() if m else None
            params = frozenset(k.arg for k in node.keywords if k.arg)

        if sid not in taint_cache:
            taint_cache[sid] = _tainted_vars(scope)
        tainted = _prompt_is_tainted(p_nodes, taint_cache[sid])

        line = getattr(node, "lineno", 0)
        # Capture the whole call span (collapsed to one line), not just its first
        # physical line, so two different multi-line calls don't share a snippet --
        # the baseline fingerprints on this, and identical first lines would collide.
        end_line = getattr(node, "end_lineno", line) or line
        if 0 < line <= len(source_lines):
            raw = " ".join(source_lines[line - 1:min(end_line, len(source_lines))])
            snippet = " ".join(raw.split())[:160]
        else:
            snippet = ""
        loop_kind = _loop_kind(node, parents)
        calls.append(
            LLMCall(
                node=node,
                line=line,
                col=getattr(node, "col_offset", 0),
                api=api,
                prompt_text=text,
                prompt_static=static,
                prompt_resolved=resolved,
                loop_kind=loop_kind,
                loop_concurrent=(loop_kind == "batchable" and _is_concurrent_dispatch(node, parents)),
                tainted=tainted,
                snippet=snippet,
                model=model,
                params=params,
            )
        )
    return calls
