"""Detect LLM/AI calls in JavaScript / TypeScript via tree-sitter.

Produces the same LLMCall objects the Python detector does, so the shared rules
in rules.py apply unchanged. Requires the optional `overllm[js]` dependencies;
if they are absent, `available()` returns False and JS files are skipped.
"""

from __future__ import annotations

import re

from .detector import LLMCall

try:  # optional extra: pip install overllm[js]
    import tree_sitter_javascript as _tsjs
    import tree_sitter_typescript as _tsts
    from tree_sitter import Language, Parser
    _AVAILABLE = True
except ImportError:  # pragma: no cover
    _AVAILABLE = False


def available() -> bool:
    return _AVAILABLE


# Vercel AI SDK top-level functions.
_VERCEL_FNS = {"generateText", "streamText", "generateObject", "streamObject"}
# openai-node / anthropic-node method-call suffixes.
_METHOD_SUFFIXES = (
    ("chat", "completions", "create"),
    ("chat", "completions", "parse"),
    ("messages", "create"),
    ("messages", "stream"),
    ("responses", "create"),
    ("completions", "create"),
)
_WEB_ROOTS = {"req", "request"}
_WEB_PROPS = {"query", "body", "params", "searchParams", "nextUrl"}
_LLM_KEYS = {"model", "messages", "prompt", "system", "input"}

_LOOP_TYPES = {"for_in_statement", "for_statement", "while_statement", "do_statement"}
_FUNC_TYPES = {
    "function_declaration", "function_expression", "arrow_function",
    "method_definition", "generator_function_declaration", "generator_function",
}

_parsers: dict = {}


def _parser(lang: str):
    if lang not in _parsers:
        if lang == "ts":
            language = Language(_tsts.language_typescript())
        elif lang == "tsx":
            language = Language(_tsts.language_tsx())
        else:
            language = Language(_tsjs.language())
        _parsers[lang] = Parser(language)
    return _parsers[lang]


def lang_for(path: str) -> str | None:
    if path.endswith(".ts") or path.endswith(".mts") or path.endswith(".cts"):
        return "ts"
    if path.endswith(".tsx"):
        return "tsx"
    if path.endswith((".js", ".jsx", ".mjs", ".cjs")):
        return "js"
    return None


def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", "ignore")


def _member_props(func) -> tuple[str | None, tuple[str, ...]]:
    """For a member_expression, return (root_identifier, trailing property names)."""
    props: list[str] = []
    node = func
    while node is not None and node.type == "member_expression":
        prop = node.child_by_field_name("property")
        if prop is not None:
            props.append(prop.text.decode("utf-8", "ignore"))
        node = node.child_by_field_name("object")
    props.reverse()
    root = node.text.decode("utf-8", "ignore") if node is not None and node.type == "identifier" else None
    return root, tuple(props)


def _object_arg(call):
    args = call.child_by_field_name("arguments")
    if args is None:
        return None
    for c in args.named_children:
        if c.type == "object":
            return c
    return None


def _pairs(obj) -> dict:
    out = {}
    for c in obj.named_children:
        if c.type == "pair":
            key = c.child_by_field_name("key")
            val = c.child_by_field_name("value")
            if key is not None and val is not None:
                name = key.text.decode("utf-8", "ignore").strip("\"'`")
                out[name] = val
    return out


# --- variable resolution (one hop, single declaration) ----------------------

def _declarations(root, src: bytes) -> dict:
    decls: dict = {}
    seen_twice: set = set()

    def walk(n):
        if n.type == "variable_declarator":
            name = n.child_by_field_name("name")
            val = n.child_by_field_name("value")
            if name is not None and name.type == "identifier" and val is not None:
                nm = name.text.decode("utf-8", "ignore")
                if nm in decls:
                    seen_twice.add(nm)
                decls[nm] = val
        for c in n.children:
            walk(c)

    walk(root)
    return {k: v for k, v in decls.items() if k not in seen_twice}


def _resolve(node, decls: dict):
    seen: set = set()
    while node is not None and node.type == "identifier":
        nm = node.text.decode("utf-8", "ignore")
        if nm in decls and nm not in seen:
            seen.add(nm)
            node = decls[nm]
        else:
            break
    return node


# --- text / static ----------------------------------------------------------

def _literal_text_and_static(node, src: bytes, decls: dict) -> tuple[str, bool]:
    texts: list[str] = []
    static = [True]

    def walk(n):
        n = _resolve(n, decls)
        t = n.type
        if t == "string":
            for c in n.children:
                if c.type == "string_fragment":
                    texts.append(_text(c, src))
        elif t == "template_string":
            for c in n.children:
                if c.type == "string_fragment":
                    texts.append(_text(c, src))
                elif c.type == "template_substitution":
                    static[0] = False
        elif t == "binary_expression":
            left = n.child_by_field_name("left")
            right = n.child_by_field_name("right")
            if left is not None:
                walk(left)
            if right is not None:
                walk(right)
        elif t in ("number", "true", "false", "null", "undefined"):
            pass
        else:
            static[0] = False

    walk(node)
    return " ".join(texts).lower().strip(), static[0]


def _messages_user(node, src: bytes, decls: dict) -> tuple[list, bool]:
    node = _resolve(node, decls)
    if node.type != "array":
        return [], False
    contents = []
    for el in node.named_children:
        el = _resolve(el, decls)
        if el.type != "object":
            return [], False
        p = _pairs(el)
        role_node = p.get("role")
        role = None
        if role_node is not None:
            rt, _ = _literal_text_and_static(role_node, src, decls)
            role = rt
        if role is None:
            return [], False
        if role == "user" and "content" in p:
            contents.append(p["content"])
    return contents, True


def _prompt_nodes(obj, src: bytes, decls: dict) -> tuple[list, bool]:
    p = _pairs(obj)
    if "messages" in p:
        return _messages_user(p["messages"], src, decls)
    for key in ("prompt", "input"):
        if key in p:
            return [p[key]], True
    return [], False


# --- taint (remote web-request input) ---------------------------------------

def _expr_is_untrusted(node, src: bytes, decls: dict) -> bool:
    def check(n):
        if n.type == "member_expression":
            root, props = _member_props(n)
            if root in _WEB_ROOTS and any(pr in _WEB_PROPS for pr in props):
                return True
        for c in n.children:
            if check(c):
                return True
        return False

    return check(_resolve(node, decls)) or check(node)


def _tainted_names(root, src: bytes, decls: dict) -> set:
    tainted: set = set()
    for nm, val in decls.items():
        if _expr_is_untrusted(val, src, decls):
            tainted.add(nm)
    return tainted


def _prompt_tainted(nodes, src: bytes, decls: dict, tainted: set) -> bool:
    for n in nodes:
        if _expr_is_untrusted(n, src, decls):
            return True
        for m in _walk(n):
            if m.type == "identifier" and m.text.decode("utf-8", "ignore") in tainted:
                return True
    return False


def _walk(n):
    yield n
    for c in n.children:
        yield from _walk(c)


# --- loops -------------------------------------------------------------------

def _loop_kind(call, src: bytes) -> str | None:
    n = call.parent
    while n is not None:
        if n.type in _FUNC_TYPES:
            return None
        if n.type in _LOOP_TYPES:
            body = n.child_by_field_name("body")
            if body is not None and body.start_byte <= call.start_byte < body.end_byte:
                if n.type == "for_in_statement":
                    header = _text(n, src).split("{")[0].lower()
                    if any(k in header for k in ("batch", "chunk", "group")):
                        return "necessary"
                    return "batchable"
                return "necessary"  # while / do / C-style for
        n = n.parent
    return None


# --- classification ----------------------------------------------------------

def _ai_imports(root, src: bytes) -> set:
    """Names imported from the `ai` package or an `@ai-sdk/*` provider.

    Gating the bare Vercel functions on a real import avoids matching a
    user-defined helper that happens to be named `generateText`.
    """
    names: set = set()
    for n in _walk(root):
        if n.type == "import_statement":
            source = n.child_by_field_name("source")
            if source is None:
                continue
            mod = _text(source, src).strip("\"'`")
            if mod == "ai" or mod.startswith("@ai-sdk"):
                for m in _walk(n):
                    if m.type == "identifier":
                        names.add(_text(m, src))
    return names


def _classify(call, ai_imports: set) -> bool:
    func = call.child_by_field_name("function")
    if func is None:
        return False
    if func.type == "identifier":
        name = func.text.decode("utf-8", "ignore")
        return name in _VERCEL_FNS and name in ai_imports
    if func.type == "member_expression":
        _, props = _member_props(func)
        for suf in _METHOD_SUFFIXES:
            if len(props) >= len(suf) and tuple(props[-len(suf):]) == suf:
                return True
    return False


def find_llm_calls_js(source: str, lang: str) -> list[LLMCall]:
    if not _AVAILABLE:
        return []
    src = source.encode("utf-8")
    tree = _parser(lang).parse(src)
    root = tree.root_node
    decls = _declarations(root, src)
    ai_imports = _ai_imports(root, src)
    tainted = _tainted_names(root, src, decls)
    lines = source.splitlines()

    calls: list[LLMCall] = []
    for node in _walk(root):
        if node.type != "call_expression" or not _classify(node, ai_imports):
            continue
        obj = _object_arg(node)
        text, static, resolved = "", False, False
        p_nodes: list = []
        if obj is not None:
            p_nodes, resolved = _prompt_nodes(obj, src, decls)
            if resolved and p_nodes:
                parts = []
                st = True
                for pn in p_nodes:
                    t, s = _literal_text_and_static(pn, src, decls)
                    if t:
                        parts.append(t)
                    st = st and s
                text, static = " ".join(parts).strip(), st
        line = node.start_point[0] + 1
        col = node.start_point[1]
        snippet = lines[line - 1].strip() if 0 < line <= len(lines) else ""
        calls.append(
            LLMCall(
                node=node,
                line=line,
                col=col,
                api="js",
                prompt_text=text,
                prompt_static=static,
                prompt_resolved=resolved,
                loop_kind=_loop_kind(node, src),
                tainted=_prompt_tainted(p_nodes, src, decls, tainted) if p_nodes else False,
                snippet=snippet,
            )
        )
    return calls
