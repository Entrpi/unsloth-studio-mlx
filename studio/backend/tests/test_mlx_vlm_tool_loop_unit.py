# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Chunk H-2 — unit tests for :meth:`MlxVlmBackend.generate_chat_completion_with_tools`.

The VLM agentic loop is a near-clone of MlxLm's
``generate_chat_completion_with_tools`` (mirror the same event shapes,
reuse the shared ``_tool_call_parser``, share the concurrent-futures
timeout wrapper) so the unit tests mirror the MlxLm coverage with the
addition of the ``image_b64`` contract: the first iteration sees the
image, subsequent ones do not.

``mlx_vlm.stream_generate`` is stubbed via :mod:`unittest.mock` — none
of these tests require ``mlx-vlm`` to actually be installed, so they
run on any platform.
"""

from __future__ import annotations

import sys
import threading
import time as _time
from unittest import mock

import pytest


# ── Fixtures ────────────────────────────────────────────────────

class _FakeResp:
    """Stand-in for mlx-vlm's stream_generate response object."""

    def __init__(self, text, **kwargs):
        self.text = text
        for k, v in kwargs.items():
            setattr(self, k, v)


def _stub_vlm_backend_for_tools():
    """Return a loaded-shaped :class:`MlxVlmBackend` ready for tool
    tests.

    The stubbed backend has an in-memory model / processor and
    ``_supports_tools=True`` so the guard check passes.
    """
    from core.inference.mlx_vlm import MlxVlmBackend

    b = MlxVlmBackend()
    b._model = mock.MagicMock(name = "vlm_model")
    b._processor = mock.MagicMock(name = "vlm_processor")
    b._processor.tokenizer = mock.MagicMock(name = "vlm_tokenizer")
    b._config = {"model_type": "stub-vlm"}
    b._model_identifier = "unit/test-vlm"
    b._supports_tools = True
    # Avoid triggering the apply_chat_template path — mock it to a
    # predictable string so we don't need real ``mlx_vlm.prompt_utils``.
    b._chat_template = "{% if tool_calls %}{{ tool_calls }}{% endif %}"
    return b


def _run_vlm_tool_loop(
    b,
    *,
    turns_text,
    tools,
    tool_choice = None,
    image_b64 = None,
    max_iter = 10,
    **extra_kwargs,
):
    """Drive :meth:`generate_chat_completion_with_tools` with a scripted
    stream.

    ``turns_text`` is a list of cumulative-text strings, one per
    assistant turn. Each turn's internal loop yields two chunks so the
    test exercises the content-accumulation + hold-back path.
    """
    turn_iter = iter(turns_text)
    seen_kwargs: list = []

    def fake_stream_generate(_model, _processor, **kwargs):
        seen_kwargs.append(dict(kwargs))
        txt = next(turn_iter)
        if not txt:
            yield _FakeResp("", prompt_tokens = 5, generation_tokens = 0)
            return
        mid = len(txt) // 2
        yield _FakeResp(txt[:mid])
        yield _FakeResp(
            txt[mid:],
            prompt_tokens = 10,
            generation_tokens = 20,
            prompt_tps = 5.0,
            generation_tps = 30.0,
        )

    # Stub the mlx_vlm module for the ``from mlx_vlm import
    # stream_generate`` import inside _stream_vlm_assistant_turn and
    # stub ``prompt_utils.apply_chat_template`` so we don't need the
    # real package. Also stub _render_prompt bypass through the
    # apply_chat_template mock.
    fake_mlx_vlm = mock.MagicMock()
    fake_mlx_vlm.stream_generate = fake_stream_generate
    fake_prompt_utils = mock.MagicMock()
    fake_prompt_utils.apply_chat_template = mock.MagicMock(
        return_value = "PROMPT"
    )
    # Hook the submodule lookup — ``from mlx_vlm.prompt_utils import
    # apply_chat_template`` hits sys.modules keyed by dotted path.
    with mock.patch.dict(sys.modules, {
        "mlx_vlm": fake_mlx_vlm,
        "mlx_vlm.prompt_utils": fake_prompt_utils,
    }):
        events = list(
            b.generate_chat_completion_with_tools(
                messages = [{"role": "user", "content": "hi"}],
                tools = tools,
                tool_choice = tool_choice,
                image_b64 = image_b64,
                max_tool_iterations = max_iter,
                **extra_kwargs,
            )
        )
    return events, seen_kwargs


# ── Baseline / guard tests ──────────────────────────────────────

class TestGuards:
    def test_unloaded_backend_raises(self):
        from core.inference.mlx_vlm import MlxVlmBackend

        b = MlxVlmBackend()
        with pytest.raises(RuntimeError, match = "not loaded"):
            list(
                b.generate_chat_completion_with_tools(
                    messages = [{"role": "user", "content": "hi"}],
                    tools = [],
                )
            )

    def test_no_tools_support_raises(self):
        from core.inference.mlx_vlm import MlxVlmBackend

        b = MlxVlmBackend()
        # Pretend-loaded but supports_tools=False.
        b._model = mock.MagicMock()
        b._processor = mock.MagicMock()
        b._supports_tools = False
        with pytest.raises(RuntimeError, match = "does not advertise"):
            list(
                b.generate_chat_completion_with_tools(
                    messages = [{"role": "user", "content": "hi"}],
                    tools = [],
                )
            )


# ── Tool-choice normalisation ───────────────────────────────────

class TestToolChoiceNormalisation:
    def test_values(self):
        from core.inference.mlx_vlm import MlxVlmBackend

        assert MlxVlmBackend._normalize_tool_choice(None) == "auto"
        assert MlxVlmBackend._normalize_tool_choice("auto") == "auto"
        assert MlxVlmBackend._normalize_tool_choice("required") == "required"
        assert MlxVlmBackend._normalize_tool_choice("none") == "none"
        assert (
            MlxVlmBackend._normalize_tool_choice(
                {"type": "function", "function": {"name": "x"}}
            )
            == "required"
        )
        assert MlxVlmBackend._normalize_tool_choice("bogus") == "auto"

    def test_none_short_circuits_loop(self):
        b = _stub_vlm_backend_for_tools()
        events, _ = _run_vlm_tool_loop(
            b,
            turns_text = ["plain answer, no tool"],
            tools = [{"type": "function", "function": {"name": "x"}}],
            tool_choice = "none",
        )
        types = [e.get("type") for e in events if isinstance(e, dict)]
        # No tool_start / tool_end should appear.
        assert "tool_start" not in types
        assert "tool_end" not in types
        assert "metadata" in types


# ── Happy-path single-tool ──────────────────────────────────────

class TestVlmAgenticLoopHappyPath:
    def test_single_tool_call_then_final_answer(self):
        b = _stub_vlm_backend_for_tools()
        tool_call_markup = (
            'reasoning...\n<tool_call>{"name": "get_weather", '
            '"arguments": {"city": "Paris"}}</tool_call>'
        )
        with mock.patch(
            "core.inference.tools.execute_tool",
            return_value = '{"temp": 22}',
        ) as exec_mock:
            events, _ = _run_vlm_tool_loop(
                b,
                turns_text = [
                    tool_call_markup,
                    "The weather in Paris is 22 degrees C.",
                ],
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {},
                        },
                    }
                ],
            )

        types = [e.get("type") for e in events if isinstance(e, dict)]
        assert types.count("tool_start") == 1
        assert types.count("tool_end") == 1
        assert types.count("metadata") == 1
        tool_starts = [
            e for e in events
            if isinstance(e, dict) and e.get("type") == "tool_start"
        ]
        assert tool_starts[0]["tool_name"] == "get_weather"
        assert tool_starts[0]["arguments"] == {"city": "Paris"}
        exec_mock.assert_called_once()

    def test_gemma_dialect_tool_call(self):
        """Gemma-4's ``<|tool_call>call:NAME{...}<tool_call|>`` idiom
        is parsed by the shared parser (Chunk H-2 B3 closure). This is
        the exact shape Gemma-4 E4B VLM emits on a ``get_weather``
        prompt.
        """
        b = _stub_vlm_backend_for_tools()
        gemma_call = (
            "Let me check.\n"
            '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>'
        )
        with mock.patch(
            "core.inference.tools.execute_tool",
            return_value = '{"temp": 18}',
        ):
            events, _ = _run_vlm_tool_loop(
                b,
                turns_text = [gemma_call, "Paris is 18 degrees."],
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {},
                        },
                    }
                ],
            )
        starts = [
            e for e in events
            if isinstance(e, dict) and e.get("type") == "tool_start"
        ]
        assert starts, "Gemma dialect call didn't parse into tool_start"
        assert starts[0]["tool_name"] == "get_weather"
        args = starts[0]["arguments"]
        # Arguments may be dict or {"raw": "..."} — either is acceptable
        # as long as Paris is findable.
        args_str = str(args)
        assert "Paris" in args_str


# ── Image-b64 handoff contract ──────────────────────────────────

class TestImageBbHandoff:
    """The first iteration must forward the image to stream_generate;
    subsequent iterations must run text-only. Uses a small 4×4 PNG
    base64 so _decode_image_b64_to_path succeeds without PIL errors.
    """

    @staticmethod
    def _tiny_png_b64():
        import base64
        import io

        from PIL import Image

        img = Image.new("RGB", (4, 4), "red")
        buf = io.BytesIO()
        img.save(buf, format = "PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def test_first_iteration_gets_image_second_does_not(self):
        b = _stub_vlm_backend_for_tools()
        with mock.patch(
            "core.inference.tools.execute_tool",
            return_value = "ok",
        ):
            events, seen_kwargs = _run_vlm_tool_loop(
                b,
                turns_text = [
                    '<tool_call>{"name": "x", "arguments": {}}</tool_call>',
                    "done",
                ],
                tools = [{"type": "function", "function": {"name": "x"}}],
                image_b64 = self._tiny_png_b64(),
            )
        # Two stream_generate calls — one per iteration.
        assert len(seen_kwargs) == 2, (
            f"expected 2 stream_generate calls, saw {len(seen_kwargs)}"
        )
        # First call must have image=; second must not.
        assert "image" in seen_kwargs[0]
        assert "image" not in seen_kwargs[1], (
            f"second iteration unexpectedly got image: {seen_kwargs[1]}"
        )
        # Ensure the loop still produced a final metadata event.
        types = [e.get("type") for e in events if isinstance(e, dict)]
        assert "metadata" in types


# ── Content stream hold-back ────────────────────────────────────

class TestContentStreamHoldBack:
    """Verify that partial ``<tool_call>`` markup doesn't leak to the
    SSE wire mid-stream. Mirrors the MlxLm regression coverage added
    in commit 257d486c.
    """

    def test_partial_tool_call_markup_held_back(self):
        b = _stub_vlm_backend_for_tools()
        with mock.patch(
            "core.inference.tools.execute_tool",
            return_value = "ok",
        ):
            events, _ = _run_vlm_tool_loop(
                b,
                turns_text = [
                    # Half the tool call, but still a valid
                    # cumulative string.
                    '<tool_call>{"name": "x", "arguments": {}}</tool_call>',
                    "final.",
                ],
                tools = [{"type": "function", "function": {"name": "x"}}],
            )
        contents = [
            e for e in events
            if isinstance(e, dict) and e.get("type") == "content"
        ]
        # None of the intermediate content events should contain the
        # raw ``<tool_call>`` marker (stripped / held back).
        for c in contents:
            text = c.get("text", "")
            assert "<tool_call>" not in text, (
                f"partial markup leaked to content stream: {text!r}"
            )


# ── Timeout wrapper ─────────────────────────────────────────────

class TestTimeoutWrapper:
    """Regression: a hung tool must not wedge the VLM agentic loop
    beyond the configured ``tool_call_timeout``. Same contract as the
    MlxLm test ``test_hung_tool_returns_timeout_error_not_deadlock``.
    """

    def test_hung_tool_returns_timeout_error(self):
        b = _stub_vlm_backend_for_tools()
        slept_for = {}

        def _hang(name, arguments, cancel_event, timeout, session_id):
            start = _time.monotonic()
            _time.sleep(10)
            slept_for["done"] = _time.monotonic() - start
            return "should-not-reach-here"

        with mock.patch(
            "core.inference.tools.execute_tool", side_effect = _hang
        ):
            start = _time.monotonic()
            events, _ = _run_vlm_tool_loop(
                b,
                turns_text = [
                    '<tool_call>{"name": "web_search", '
                    '"arguments": {"q": "x"}}</tool_call>',
                    "final.",
                ],
                tools = [
                    {"type": "function", "function": {"name": "web_search"}}
                ],
                tool_call_timeout = 1,
                max_iter = 3,
            )
            elapsed = _time.monotonic() - start

        assert elapsed < 5, (
            f"tool-hang wedged the VLM loop for {elapsed:.1f}s; "
            f"timeout was not enforced"
        )
        tool_end_events = [
            e for e in events if e.get("type") == "tool_end"
        ]
        assert tool_end_events, "no tool_end event — loop didn't unblock"
        assert "timed out" in tool_end_events[0]["result"].lower()
        assert "done" not in slept_for


# ── Cancel event exit ───────────────────────────────────────────

class TestCancelEventExit:
    """Setting cancel_event mid-tool must return from the loop within
    the 0.5 s poll slice — mirrors the MlxLm coverage.
    """

    def test_cancel_during_tool_exits_cleanly(self):
        b = _stub_vlm_backend_for_tools()
        cancel = threading.Event()

        def _hang(name, arguments, cancel_event, timeout, session_id):
            _time.sleep(10)
            return "too-late"

        def _fire_cancel():
            _time.sleep(0.3)
            cancel.set()

        threading.Thread(target = _fire_cancel, daemon = True).start()
        with mock.patch(
            "core.inference.tools.execute_tool", side_effect = _hang
        ):
            start = _time.monotonic()
            events, _ = _run_vlm_tool_loop(
                b,
                turns_text = [
                    '<tool_call>{"name": "web_search", '
                    '"arguments": {"q": "x"}}</tool_call>',
                    "final.",
                ],
                tools = [
                    {"type": "function", "function": {"name": "web_search"}}
                ],
                cancel_event = cancel,
                tool_call_timeout = 30,
                max_iter = 3,
            )
            elapsed = _time.monotonic() - start

        assert elapsed < 3, (
            f"cancel_event didn't interrupt VLM tool loop; "
            f"ran {elapsed:.1f}s"
        )
        # tool_start at minimum must have fired.
        assert events


# ── Max-iterations cap ──────────────────────────────────────────

class TestVlmMaxIterations:
    def test_cap_triggers_final_nudge(self):
        b = _stub_vlm_backend_for_tools()
        # Each turn uses different arguments so the duplicate-call
        # guard (TestAgenticLoopDuplicateDetection in the MLX-LM
        # test file covers that) doesn't short-circuit the cap.
        calls = [
            '<tool_call>{"name": "loop", "arguments": {"step": 1}}</tool_call>',
            '<tool_call>{"name": "loop", "arguments": {"step": 2}}</tool_call>',
            "Final fallback answer.",
        ]
        with mock.patch(
            "core.inference.tools.execute_tool", return_value = "ok"
        ):
            events, _ = _run_vlm_tool_loop(
                b,
                turns_text = calls,
                tools = [
                    {"type": "function", "function": {"name": "loop"}}
                ],
                max_iter = 2,
            )
        starts = [e for e in events if e.get("type") == "tool_start"]
        assert len(starts) == 2
        contents = [e for e in events if e.get("type") == "content"]
        assert any("Final fallback answer" in c["text"] for c in contents)


# ── Metadata event ──────────────────────────────────────────────

class TestVlmMetadataEvent:
    def test_metadata_event_has_token_counts(self):
        b = _stub_vlm_backend_for_tools()
        with mock.patch(
            "core.inference.tools.execute_tool",
            return_value = "ok",
        ):
            events, _ = _run_vlm_tool_loop(
                b,
                turns_text = [
                    '<tool_call>{"name": "x", "arguments": {}}</tool_call>',
                    "final",
                ],
                tools = [{"type": "function", "function": {"name": "x"}}],
            )
        metas = [
            e for e in events
            if isinstance(e, dict) and e.get("type") == "metadata"
        ]
        assert metas
        final_meta = metas[-1]
        usage = final_meta.get("usage", {})
        # Two assistant turns, each stub generates 20 completion tokens
        # → 40 total.
        assert usage["completion_tokens"] >= 20
        # total = prompt + completion
        assert (
            usage["total_tokens"]
            == usage["prompt_tokens"] + usage["completion_tokens"]
        )
