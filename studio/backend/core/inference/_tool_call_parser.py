# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Shared tool-call XML / JSON parser for the Unsloth Studio inference
backends.

Originally lived as :meth:`LlamaCppBackend._parse_tool_calls_from_text`
at ``core/inference/llama_cpp.py:2016`` and was lifted into this module
during Phase-5 (Chunk C) so the MLX backend can share the same parsing
logic without taking a dependency on the GGUF backend.

Three wire-level dialects are recognised. All are emitted by chat
templates in the wild, and the same Qwen3-family model can emit either
one depending on how the base template was customised:

1. **JSON-inside-``<tool_call>``** (Qwen3 / Bonsai / most open-weights
   templates). The assistant turn is wrapped as::

       <tool_call>{"name": "...", "arguments": {...}}</tool_call>

   The JSON body may be pretty-printed, may include escaped quotes, and
   ``</tool_call>`` is frequently omitted at the end of the stream.

2. **XML-function tags** (Claude-style / TexteGen-family / some Mistral
   fine-tunes). The call is rendered as nested tags::

       <function=tool_name><parameter=field_name>value</parameter></function>

   where ``</parameter>`` / ``</function>`` are optional — models
   routinely drop the closing tags when the argument value contains
   ``</function>`` or similar substrings.

3. **Gemma-4 ``<|tool_call>``** (Gemma-4 E4B / 31B-it / 26B-a4b MoE).
   The call is rendered with Gemma-specific delimiter tokens::

       <|tool_call>call:tool_name{key:<|"|>value<|"|>,...}<tool_call|>

   where ``<|"|>`` is Gemma's escaped-quote token pair, keys are bare
   identifiers (no surrounding quotes), and the closing marker is
   ``<tool_call|>``. The body is JSON-ish: bools/numbers/null are
   literal, nested objects/arrays use standard ``{}`` / ``[]``. We
   normalise this to regular JSON (Gemma-quotes → ``"``, bare keys
   get quoted) and parse.

The parser returns OpenAI-compatible ``tool_calls`` dicts (the same
shape llama-server already synthesises on its end), so every caller can
feed the output straight into ``tools.execute_tool`` or an OpenAI
``assistant.tool_calls`` message without translation.

Module-level regexes are pre-compiled for the hot path:
``parse_tool_calls_from_text`` is invoked once per generation turn
inside the agentic loop; avoiding ``re.compile`` per call matters when
the same backend instance handles hundreds of requests.

:func:`strip_tool_markup` is a second public entry used by the GGUF
backend's speculative buffer to clean XML fragments out of the text
stream before it yields to the consumer. It is dialect-agnostic — the
regexes here are the canonical definition, the backend no longer
maintains its own copies.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union


# ── Pre-compiled patterns for tool-call XML parsing ──────────────────
# Keep the regex names aligned with the ``_TC_*`` names the GGUF
# backend used before extraction, so grep hits on either still land on
# the same patterns.

#: Start-tag that signals a JSON-body tool call. The ``\s*\{`` tail
#: ensures we only match when the body actually opens a JSON object —
#: a bare ``<tool_call>`` with no body is not a tool call.
_TC_JSON_START_RE = re.compile(r"<tool_call>\s*\{")

#: Start-tag of an XML-function call. The function name is captured.
_TC_FUNC_START_RE = re.compile(r"<function=(\w+)>\s*")

#: End-tag for the ``<tool_call>`` wrapper. Used as a hard boundary when
#: scanning body contents. Optional in practice.
_TC_END_TAG_RE = re.compile(r"</tool_call>")

#: End-tag for a function body. Only stripped at the *tail* of a body
#: because the literal ``</function>`` can appear inside code parameter
#: values.
_TC_FUNC_CLOSE_RE = re.compile(r"\s*</function>\s*$")

#: Start-tag of a parameter inside a function body.
_TC_PARAM_START_RE = re.compile(r"<parameter=(\w+)>\s*")

#: End-tag of a parameter. Only stripped from the tail of a value for
#: the same reason ``</function>`` is: closing tags may appear inside
#: code/text values and we don't want to truncate at the first literal
#: occurrence.
_TC_PARAM_CLOSE_RE = re.compile(r"\s*</parameter>\s*$")

#: Gemma-4 tool-call start: ``<|tool_call>call:NAME{``. Captures the
#: function name and aligns with the opening brace so the balanced-
#: brace walker can pick up from there.
_TC_GEMMA_START_RE = re.compile(r"<\|tool_call>call:(\w+)\s*\{")

#: Gemma's escaped-quote token pair — used to delimit string values in
#: the tool-call body. Normalised to ``"`` before JSON parsing.
_TC_GEMMA_QUOTE = '<|"|>'

#: Unquoted-key detector for Gemma bodies. Only fires at a position
#: following ``{`` or ``,`` (with optional whitespace) — matches the
#: natural positions of object keys without touching identifiers
#: elsewhere.
_TC_GEMMA_KEY_RE = re.compile(r'([{,]\s*)(\w+)\s*:')

#: GLM-4/4.6/4.7 tool-call start: ``<tool_call>{name}\n`` where the
#: first line after the opening tag is the function name (no JSON, no
#: colon, just the bare identifier). The ``(?!\s*\{)`` guard ensures
#: we don't steal matches from the JSON-body dialect — a body that
#: opens with ``{`` is JSON, not GLM.
_TC_GLM_START_RE = re.compile(
    r"<tool_call>\s*(?!\{)([\w.\-]+)\s*\n", re.IGNORECASE
)

#: GLM argument pair: ``<arg_key>K</arg_key>\s*<arg_value>V</arg_value>``.
#: Non-greedy so adjacent pairs don't collapse. Values may span
#: multiple lines (code blocks etc.), so DOTALL.
_TC_GLM_ARG_RE = re.compile(
    r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
    re.DOTALL,
)

#: GLM tool-call close tag.
_TC_GLM_END_RE = re.compile(r"</tool_call>")


# ── Auto-heal / stripping patterns ───────────────────────────────────
# These are the regexes GGUF uses to *remove* tool-call XML from a
# content stream after the parser has lifted out the structured calls.
# The two tiers map to the two moments when stripping runs:
#
#   • ``CLOSED``: only complete, well-formed tool-call blocks are
#     removed. Used while we're still streaming — we don't want to cut
#     off a partial block mid-token because the model might still be
#     emitting the closing tag.
#   • ``ALL`` = ``CLOSED`` + greedy patterns that also catch *unclosed*
#     tool-call blocks. Applied at the end of the turn, when we know no
#     further tokens are coming.

#: Fully-closed patterns safe to strip mid-stream.
TOOL_CLOSED_PATS = [
    re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL),
    re.compile(r"<function=\w+>.*?</function>", re.DOTALL),
    re.compile(r"<\|tool_call>.*?<tool_call\|>", re.DOTALL),
    # Gemma-4 thinking-channel block — emitted as ``<|channel>thought\n
    # ...<channel|>`` before the tool call. Not a tool call itself, but
    # the re-render path in ``mlx_vlm`` routes channel content back to
    # ``reasoning_content`` where the template puts it in the right
    # place. Strip it from the visible content stream so it doesn't
    # leak into the UI as raw markup.
    re.compile(r"<\|channel>.*?<channel\|>", re.DOTALL),
]

#: Final-flush patterns: also strip dangling unclosed blocks.
TOOL_ALL_PATS = TOOL_CLOSED_PATS + [
    re.compile(r"<tool_call>.*$", re.DOTALL),
    re.compile(r"<function=\w+>.*$", re.DOTALL),
    re.compile(r"<\|tool_call>.*$", re.DOTALL),
    re.compile(r"<\|channel>.*$", re.DOTALL),
    # Orphan closing fragments. When a tool-call body is stripped by
    # the closed-pattern above but the stream also contains a stray
    # trailing close marker (from a malformed / continuation turn —
    # e.g. Gemma-4 resuming mid-tool_call generates ``<|"|>}<tool_call|>``
    # with no opening), those fragments would otherwise leak to the UI.
    # Safe to strip at final-flush time.
    re.compile(r"<tool_call\|>", re.DOTALL),
    re.compile(r"<tool_response\|>", re.DOTALL),
    re.compile(r"<channel\|>", re.DOTALL),
    re.compile(r"<turn\|>", re.DOTALL),
    re.compile(r'<\|"\|>', re.DOTALL),
]

#: Prefixes that the speculative buffer watches for. If the assistant
#: stream starts with any of these, the buffer holds back emission
#: until the shape resolves (either into a complete tool call, which
#: gets drained, or into plain content, which gets flushed).
TOOL_XML_SIGNALS = (
    "<tool_call>",
    "<function=",
    "<|tool_call>",
    "<|channel>",
)


# ── Public dataclass result (optional) ───────────────────────────────
# Primary callers still consume the OpenAI-shaped dict returned by
# :func:`parse_tool_calls_from_text` for minimal churn. The dataclass
# is a thin convenience layer for tests and future callers that want a
# typed surface.


@dataclass
class ParsedToolCall:
    """Typed view of a parsed tool call.

    Mirrors the OpenAI ``tool_calls`` entry shape. Use
    :meth:`to_openai_dict` to round-trip back to the plain dict format
    the rest of Studio consumes.
    """

    name: str
    #: Always a JSON-encoded string, matching the OpenAI wire format
    #: where ``function.arguments`` is a string even when the upstream
    #: model emitted a structured object.
    arguments: str
    #: Unique OpenAI-style call id (``call_<uuid>``). Synthesised when
    #: the source markup did not carry one.
    id: str
    #: Raw markup slice that produced this call, for logging.
    raw: str = ""
    #: Which dialect the markup matched: ``"json"`` or ``"xml"``.
    dialect: str = "json"

    def to_openai_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        }


# ── Public API ───────────────────────────────────────────────────────


def parse_tool_calls_from_text(
    content: str,
    *,
    model_family: str = "auto",
) -> List[Dict[str, Any]]:
    """Parse tool calls from text markup.

    Returns a list of OpenAI-compatible ``tool_calls`` dicts (same
    shape as OpenAI's ``assistant.tool_calls`` / llama-server's
    ``delta.tool_calls``). Each dict has::

        {
            "id": "call_<n>",
            "type": "function",
            "function": {
                "name": "<tool>",
                "arguments": "<json-encoded-string>",
            },
        }

    Args:
        content: Raw assistant text including any tool-call markup.
        model_family: Dialect hint. ``"auto"`` (default) tries the
            JSON-body dialect first and falls back to the XML-function
            dialect when no calls are found. ``"qwen"`` / ``"bonsai"``
            / ``"hermes"`` force the JSON-body dialect. ``"claude"`` /
            ``"xml"`` force the XML-function dialect. Unknown values
            behave like ``"auto"``.

    The function is non-destructive — it does not mutate ``content``.
    Callers that want the text with tool-call markup stripped should
    additionally call :func:`strip_tool_markup`.
    """
    if not content:
        return []

    family = (model_family or "auto").lower()
    tool_calls: List[Dict[str, Any]] = []

    try_json = family in ("auto", "qwen", "bonsai", "hermes", "json")
    try_xml = family in ("auto", "claude", "xml", "text-gen", "mistral")
    try_gemma = family in ("auto", "gemma", "gemma4", "gemma-4")
    try_glm = family in ("auto", "glm", "glm4", "glm-4", "glm4v")

    # ── Dialect 1: JSON inside <tool_call> tags ─────────────────
    # Use balanced-brace extraction that skips braces inside JSON
    # strings. This is critical because the common case has the whole
    # arguments object pretty-printed with newlines inside the
    # <tool_call> wrapper.
    if try_json:
        for m in _TC_JSON_START_RE.finditer(content):
            brace_start = m.end() - 1  # position of the opening {
            depth, i = 0, brace_start
            in_string = False
            while i < len(content):
                ch = content[i]
                if in_string:
                    if ch == "\\" and i + 1 < len(content):
                        i += 2  # skip escaped character
                        continue
                    if ch == '"':
                        in_string = False
                elif ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            if depth == 0:
                json_str = content[brace_start : i + 1]
                try:
                    obj = json.loads(json_str)
                    name = obj.get("name", "") or ""
                    args = obj.get("arguments", {})
                    # OpenAI-compat: arguments is always a JSON string
                    if isinstance(args, dict):
                        args_str = json.dumps(args)
                    elif isinstance(args, str):
                        args_str = args
                    else:
                        args_str = json.dumps(args)
                    tool_calls.append(
                        {
                            "id": f"call_{len(tool_calls)}",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": args_str,
                            },
                        }
                    )
                except (json.JSONDecodeError, ValueError):
                    # Malformed JSON body — skip this match. Other
                    # dialects may still produce valid calls.
                    pass

    # ── Dialect 2: XML <function=name><parameter=key>value</parameter> ──
    # Only try this when the JSON pass produced nothing, or when the
    # caller forced the XML dialect. This keeps the behaviour
    # identical to what GGUF had before the extraction: a model that
    # emits mixed markup is parsed by its primary dialect only.
    if try_xml and (not tool_calls or family in ("claude", "xml", "text-gen", "mistral")):
        xml_calls = _parse_xml_function_dialect(content)
        # Re-number to keep ids unique across dialects.
        for call in xml_calls:
            call["id"] = f"call_{len(tool_calls)}"
            tool_calls.append(call)

    # ── Dialect 3: Gemma-4 <|tool_call>call:NAME{...}<tool_call|> ──
    # Tried last in auto mode because the opening marker is a distinct
    # token pair that doesn't collide with the other two dialects, and
    # models that emit Gemma-style calls never emit JSON-in-<tool_call>
    # simultaneously. Forced when caller passes model_family="gemma".
    if try_gemma and (not tool_calls or family in ("gemma", "gemma4", "gemma-4")):
        gemma_calls = _parse_gemma_dialect(content)
        for call in gemma_calls:
            call["id"] = f"call_{len(tool_calls)}"
            tool_calls.append(call)

    # ── Dialect 5: GLM-4/4.6/4.7 <tool_call>name\n<arg_key>...</arg_key>
    # <arg_value>...</arg_value>...</tool_call> ───────────────────────
    # Shares the ``<tool_call>`` opener with Dialect 1 but has a bare
    # identifier (not JSON) after the tag — the JSON regex requires
    # ``{`` immediately after the opener and therefore never matches a
    # GLM body. We try GLM on the ``not tool_calls`` branch in auto
    # mode so it fires when Dialect 1 produced nothing.
    if try_glm and (not tool_calls or family in ("glm", "glm4", "glm-4", "glm4v")):
        glm_calls = _parse_glm_dialect(content)
        for call in glm_calls:
            call["id"] = f"call_{len(tool_calls)}"
            tool_calls.append(call)

    # ── Dialect 4: loose top-level JSON envelope ──────────────────
    # Some models (notably Gemma-4 E4B under certain prompts, also a few
    # Llama / Mistral fine-tunes) emit a bare JSON object *without* any
    # wrapper tag when they decide to call a tool. Key name conventions
    # vary — observed in the wild:
    #   {"tool_name": "X", "params": {...}}
    #   {"name": "X", "arguments": {...}}
    #   {"tool": "X", "input": {...}}
    #   {"function": "X", "parameters": {...}}
    # Try these as a last resort: only when every other dialect found
    # nothing AND the content is a single top-level JSON object (or
    # ends with one) that has one of the name/args key pairs. This is
    # strictly narrower than "any JSON content is a tool call" — a
    # legitimate JSON response without tool-call keys stays inert.
    if not tool_calls and try_json:
        loose_calls = _parse_loose_json_envelope(content)
        for call in loose_calls:
            call["id"] = f"call_{len(tool_calls)}"
            tool_calls.append(call)

    return tool_calls


def strip_tool_markup(
    text: str,
    *,
    final: bool = False,
) -> str:
    """Remove tool-call markup from a content stream.

    Args:
        text: Content text possibly containing tool-call XML.
        final: When False (default), only strip *closed* tool-call
            blocks — safe to call mid-stream because a partial block
            might still be completing. When True, also strip any
            trailing unclosed blocks (``<tool_call>.*$``), suitable
            for the final flush at end-of-turn.

    The return value is ``text.strip()`` when ``final=True`` and
    unmodified apart from the regex substitutions when ``final=False``
    — matching the behaviour GGUF relies on in its streaming auto-heal
    path.
    """
    if not text:
        return text
    patterns = TOOL_ALL_PATS if final else TOOL_CLOSED_PATS
    for pat in patterns:
        text = pat.sub("", text)
    return text.strip() if final else text


#: Gemma-4 thinking-channel block. The opening tag is
#: ``<|channel>thought`` (or similar channel-name suffix); the closing
#: tag is ``<channel|>``. ``strip_thinking`` in the Gemma chat template
#: removes these from content at render time, so to avoid double-
#: emitting (once as raw markup in content, once as a re-rendered
#: ``reasoning_content`` block) we extract them out of the raw turn
#: text before storing the assistant message.
_CHANNEL_BLOCK_RE = re.compile(
    r"<\|channel>(?:thought\s*\n?)?(.*?)<channel\|>",
    re.DOTALL,
)


def extract_channel_thought(text: str) -> Tuple[Optional[str], str]:
    """Split Gemma-4's ``<|channel>thought\\n...<channel|>`` block.

    Returns ``(reasoning, remaining_content)``.

    - ``reasoning`` is ``None`` when no channel block is present.
      Otherwise it is the concatenation of all channel bodies (trimmed,
      newline-joined). The Gemma template re-renders ``reasoning`` /
      ``reasoning_content`` back into ``<|channel>thought\\n...<channel|>``
      at the correct position (before tool_calls), so round-tripping
      is lossless.
    - ``remaining_content`` is the original text with all channel
      blocks removed, preserving surrounding prose verbatim (no
      ``.strip()`` — callers decide whether to trim).

    Multiple channel blocks in the same text concatenate into a single
    reasoning string separated by ``\\n\\n``. This matches the UX
    expectation of showing one "thoughts" section per assistant turn.
    """
    if not text or "<|channel>" not in text:
        return None, text
    thoughts: List[str] = []

    def _capture(m: "re.Match[str]") -> str:
        body = m.group(1).strip()
        if body:
            thoughts.append(body)
        return ""

    remaining = _CHANNEL_BLOCK_RE.sub(_capture, text)
    reasoning = "\n\n".join(thoughts) if thoughts else None
    return reasoning, remaining


# ── Internal helpers ─────────────────────────────────────────────────


def _parse_xml_function_dialect(content: str) -> List[Dict[str, Any]]:
    """Parse the ``<function=name><parameter=key>value</parameter>`` dialect.

    Handles the case where closing tags (``</parameter>``, ``</function>``,
    ``</tool_call>``) are optional. The body boundary rules match the
    behaviour GGUF's backend had before extraction:

    - A function body ends at the next ``<function=`` start-tag, or at a
      ``</tool_call>``, or at end-of-content — whichever comes first.
    - A trailing ``</function>`` is stripped off the body *after*
      extraction, so a closing tag that arrived in time is honoured.
    - Inside a body, parameter boundaries are determined by the next
      ``<parameter=`` start-tag when there are multiple parameters.
      For a single parameter, the value extends to end-of-body; a
      trailing ``</parameter>`` is stripped. This is critical for
      code/terminal tools where the value contains ``</parameter>``-
      looking substrings.
    """
    tool_calls: List[Dict[str, Any]] = []
    func_starts = list(_TC_FUNC_START_RE.finditer(content))
    for idx, fm in enumerate(func_starts):
        func_name = fm.group(1)
        body_start = fm.end()
        next_func = (
            func_starts[idx + 1].start()
            if idx + 1 < len(func_starts)
            else len(content)
        )
        end_tag = _TC_END_TAG_RE.search(content[body_start:])
        if end_tag:
            body_end = body_start + end_tag.start()
        else:
            body_end = len(content)
        body_end = min(body_end, next_func)
        body = content[body_start:body_end]
        # Trim trailing </function> if present.
        body = _TC_FUNC_CLOSE_RE.sub("", body)

        arguments: Dict[str, str] = {}
        param_starts = list(_TC_PARAM_START_RE.finditer(body))
        if len(param_starts) == 1:
            pm = param_starts[0]
            val = body[pm.end() :]
            val = _TC_PARAM_CLOSE_RE.sub("", val)
            arguments[pm.group(1)] = val.strip()
        else:
            for pidx, pm in enumerate(param_starts):
                param_name = pm.group(1)
                val_start = pm.end()
                next_param = (
                    param_starts[pidx + 1].start()
                    if pidx + 1 < len(param_starts)
                    else len(body)
                )
                val = body[val_start:next_param]
                val = _TC_PARAM_CLOSE_RE.sub("", val)
                arguments[param_name] = val.strip()

        tool_calls.append(
            {
                "id": f"call_{len(tool_calls)}",
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": json.dumps(arguments),
                },
            }
        )
    return tool_calls


def _parse_gemma_dialect(content: str) -> List[Dict[str, Any]]:
    """Parse Gemma-4's ``<|tool_call>call:NAME{...}<tool_call|>`` dialect.

    Gemma emits tool calls with bare (unquoted) keys and string values
    wrapped in the ``<|"|>`` token pair rather than standard ``"``.
    Booleans / numbers / ``null`` are literal; nested objects and
    arrays use standard ``{}`` / ``[]``.

    The parser walks the body with balanced-brace counting that
    respects Gemma's quote tokens, then normalises the extracted body
    to standard JSON in two steps (Gemma-quotes → ``"`` and bare-key
    quoting) before ``json.loads``. Pathological string values that
    themselves contain a ``{key:`` shape at depth zero can trip the
    key-quoting regex; those calls fail ``json.loads`` and are
    skipped rather than mis-parsed.

    The returned shape matches :func:`_parse_xml_function_dialect` so
    the caller can re-number ids uniformly across dialects.
    """
    tool_calls: List[Dict[str, Any]] = []
    qlen = len(_TC_GEMMA_QUOTE)
    for m in _TC_GEMMA_START_RE.finditer(content):
        func_name = m.group(1)
        # ``m.end()`` lands one past the ``{``. Back up so body_start
        # points AT the opening brace so the depth counter starts
        # from 1 after we consume it below.
        body_start = m.end() - 1
        depth = 0
        in_quote = False
        i = body_start
        while i < len(content):
            if content[i : i + qlen] == _TC_GEMMA_QUOTE:
                in_quote = not in_quote
                i += qlen
                continue
            if not in_quote:
                ch = content[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        break
            i += 1
        if depth != 0:
            # Unclosed body; skip this call.
            continue
        body = content[body_start : i + 1]  # includes { and }

        # Normalise to JSON: Gemma-quotes to standard quotes, then
        # quote the bare keys. The order matters — if we quote keys
        # first, any Gemma-quoted string that happens to contain a
        # ``,word:`` substring would get spuriously re-quoted.
        json_text = body.replace(_TC_GEMMA_QUOTE, '"')
        json_text = _TC_GEMMA_KEY_RE.sub(r'\1"\2":', json_text)
        try:
            obj = json.loads(json_text)
        except (json.JSONDecodeError, ValueError):
            # Malformed body (or pathological string-content collision
            # with the key-quoting regex). Skip this call.
            continue
        if not isinstance(obj, dict):
            continue
        tool_calls.append(
            {
                "id": f"call_{len(tool_calls)}",
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": json.dumps(obj),
                },
            }
        )
    return tool_calls


def _parse_glm_dialect(content: str) -> List[Dict[str, Any]]:
    """Parse GLM-4/4.6/4.7's tool-call dialect.

    GLM's chat template instructs the model to emit::

        <tool_call>{function-name}
        <arg_key>{arg-key-1}</arg_key>
        <arg_value>{arg-value-1}</arg_value>
        <arg_key>{arg-key-2}</arg_key>
        <arg_value>{arg-value-2}</arg_value>
        ...
        </tool_call>

    So the first line after the ``<tool_call>`` opener is the function
    name (bare identifier — no JSON, no colon, no wrapping tag), and
    the body is a flat list of ``<arg_key>``/``<arg_value>`` pairs.

    The JSON-body dialect's ``_TC_JSON_START_RE`` requires a ``{``
    after the opener, so a GLM body never matches that dialect — the
    two are disjoint and can be tried in any order. We still try GLM
    after JSON in the auto chain so that a model emitting proper
    JSON-in-``<tool_call>`` isn't slowed down.

    Value coercion policy: the GLM template renders values via
    ``{{ v | tojson(ensure_ascii=False) if v is not string else v }}``
    — strings pass through verbatim, everything else is JSON-encoded.
    When parsing incoming text we:
      1. Try ``json.loads`` on the raw value (handles ``42``, ``true``,
         ``null``, nested objects / arrays, JSON-quoted strings).
      2. On failure, treat as a plain string (handles unquoted names
         like ``Ternary Bonsai``, Python ``None`` / ``True`` literals
         the model sometimes emits, natural-language paragraphs).

    Either way the assembled argument dict is JSON-encoded into the
    OpenAI ``arguments`` string so downstream tool execution sees a
    consistent shape.
    """
    tool_calls: List[Dict[str, Any]] = []
    for start_match in _TC_GLM_START_RE.finditer(content):
        name = start_match.group(1)
        body_start = start_match.end()
        # Scope the body to the next </tool_call> or the next
        # <tool_call> opener — whichever comes first — matching the
        # behaviour of the other dialects that bound cleanly.
        end_match = _TC_GLM_END_RE.search(content, body_start)
        next_start = _TC_GLM_START_RE.search(content, body_start)
        body_end = len(content)
        if end_match:
            body_end = min(body_end, end_match.start())
        if next_start:
            body_end = min(body_end, next_start.start())
        body = content[body_start:body_end]

        arguments: Dict[str, Any] = {}
        for arg_match in _TC_GLM_ARG_RE.finditer(body):
            key = arg_match.group(1).strip()
            raw_value = arg_match.group(2)
            # Don't trim multi-line values (code/text blobs) — only
            # the surrounding whitespace a template would introduce.
            value: Any = raw_value.strip()
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                # Keep as string. Some models emit Python literals
                # (None / True / False) that aren't JSON; we still
                # preserve them as strings — tool executors can
                # normalise or reject as appropriate. Common values
                # like the string "None" for an absent URL are fine
                # to pass through verbatim.
                pass
            arguments[key] = value

        # Skip empty bodies — a malformed ``<tool_call>name\n</tool_call>``
        # with no args isn't executable and shouldn't be surfaced as a
        # call.
        if not arguments:
            continue

        tool_calls.append(
            {
                "id": f"call_{len(tool_calls)}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments),
                },
            }
        )
    return tool_calls


_LOOSE_NAME_KEYS = ("name", "tool_name", "tool", "function")
_LOOSE_ARGS_KEYS = ("arguments", "params", "input", "parameters")


def _parse_loose_json_envelope(content: str) -> List[Dict[str, Any]]:
    """Parse a bare JSON envelope as a tool call.

    Accepts top-level objects shaped like any of:
        {"tool_name": "X", "params": {...}}
        {"name": "X", "arguments": {...}}
        {"tool": "X", "input": {...}}
        {"function": "X", "parameters": {...}}

    Also handles an outer ``{"tool_calls": [...]}`` wrapper where each
    list entry is one of the above shapes.

    Scans the content's tail for the last balanced ``{...}`` object
    (the model may precede it with prose). Returns [] if nothing
    parses, if required keys are missing, or if the matched object
    looks like regular JSON content (e.g. only one of the keys is
    present and it's a plain string).
    """
    if not content or "{" not in content:
        return []

    # Find the OUTERMOST balanced object. Scan from the first '{'; if
    # it fails, try subsequent '{' positions (model may have prose
    # that looks like "Here is the call: {...}").
    for start in _iter_brace_starts(content):
        body, end = _take_balanced_object(content, start)
        if body is None:
            continue
        try:
            parsed = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(parsed, dict):
            continue
        # Outer "tool_calls" wrapper: recurse into each entry.
        tc_list = parsed.get("tool_calls")
        if isinstance(tc_list, list) and tc_list:
            out: List[Dict[str, Any]] = []
            for entry in tc_list:
                if isinstance(entry, dict):
                    call = _coerce_loose_to_tool_call(entry)
                    if call is not None:
                        call["id"] = f"call_{len(out)}"
                        out.append(call)
            if out:
                return out
        # Top-level tool-call shape.
        call = _coerce_loose_to_tool_call(parsed)
        if call is not None:
            return [call]
    return []


def _coerce_loose_to_tool_call(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalise one loose-JSON object into the OpenAI-compat shape.

    Requires BOTH a name-like key AND an args-like key (narrowly
    typed) to fire. ``{"name": "Alice"}`` is not a tool call even
    though it has a ``name``; ``{"tool_name": "web_search", "params":
    {...}}`` is.
    """
    name = None
    args_found = False
    args: Any = None
    for k in _LOOSE_NAME_KEYS:
        v = obj.get(k)
        if isinstance(v, str) and v:
            name = v
            break
        # Handle nested ``"function": {"name": ..., "arguments": ...}``.
        if k == "function" and isinstance(v, dict):
            inner_name = v.get("name")
            if isinstance(inner_name, str) and inner_name:
                name = inner_name
                if "arguments" in v:
                    args = v["arguments"]
                    args_found = True
                break
    if not name:
        return None
    if not args_found:
        for k in _LOOSE_ARGS_KEYS:
            if k in obj:
                args = obj[k]
                args_found = True
                break
    if not args_found:
        # Required key pair missing — treat as incidental JSON, not a
        # tool call. Prevents false positives on e.g. ``{"name": "X"}``.
        return None
    # Args should be a dict or string; anything else isn't a real call.
    if not isinstance(args, (dict, str, list)):
        return None
    if isinstance(args, dict):
        args_str = json.dumps(args)
    elif isinstance(args, str):
        args_str = args
    else:
        args_str = json.dumps(args)
    return {
        "id": "call_0",
        "type": "function",
        "function": {"name": name, "arguments": args_str},
    }


def _iter_brace_starts(content: str):
    """Yield indices of every unescaped '{' in ``content``."""
    in_string = False
    i = 0
    while i < len(content):
        ch = content[i]
        if in_string:
            if ch == "\\" and i + 1 < len(content):
                i += 2
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            yield i
        i += 1


def _take_balanced_object(content: str, start: int) -> Tuple[Optional[str], int]:
    """Extract content[start:end+1] where end closes the object at
    content[start] == '{'. Returns (substring, end) on success, or
    (None, start) on mismatch. Respects JSON string escaping so braces
    inside strings don't unbalance the counter.
    """
    if start >= len(content) or content[start] != "{":
        return None, start
    depth = 0
    in_string = False
    i = start
    while i < len(content):
        ch = content[i]
        if in_string:
            if ch == "\\" and i + 1 < len(content):
                i += 2
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return content[start : i + 1], i
        i += 1
    return None, start


__all__ = [
    "ParsedToolCall",
    "parse_tool_calls_from_text",
    "strip_tool_markup",
    "extract_channel_thought",
    "TOOL_CLOSED_PATS",
    "TOOL_ALL_PATS",
    "TOOL_XML_SIGNALS",
]
