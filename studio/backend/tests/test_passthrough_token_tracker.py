# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for :class:`routes.inference._PassthroughTokenTracker`.

The GGUF passthrough path proxies llama-server's SSE byte stream
verbatim, so the route can't emit token telemetry inline like the MLX
backend does — the tracker re-derives counts from the wire format.

These tests pin the contract:
    - ``delta.content`` chunks count as 1 token each (llama-server
      emits one chunk per token), and ``len(content)`` is accumulated.
    - ``delta.tool_calls`` chunks also count (model is generating the
      function-call JSON; those are output tokens).
    - The trailing ``stream_options.include_usage`` chunk overrides
      the interpolated count with ``usage.completion_tokens`` and
      force-emits.
    - Emissions are rate-limited to ``emit_interval`` for live ticks
      but NOT for the final reconciliation.
    - Without a session_id or a broadcaster the tracker is inert.
    - Malformed JSON / missing fields don't raise.

No app boot, no httpx — the tracker is a pure function over SSE
payload strings.
"""

import asyncio
import json
import os
import sys

import pytest

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from core.telemetry.broadcaster import TelemetryBroadcaster, SubscriptionSpec
from routes.inference import _PassthroughTokenTracker


# ── Helpers ─────────────────────────────────────────────────────────


def _content_chunk(text: str) -> str:
    """Build the JSON payload for a content-bearing SSE chunk."""
    return json.dumps(
        {
            "id": "chatcmpl-test",
            "choices": [{"index": 0, "delta": {"content": text}}],
        }
    )


def _tool_call_chunk(arg_fragment: str) -> str:
    """Build the JSON payload for a tool_calls SSE chunk."""
    return json.dumps(
        {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": arg_fragment},
                            }
                        ]
                    },
                }
            ],
        }
    )


def _usage_chunk(completion_tokens: int) -> str:
    """Build the trailing usage chunk llama-server emits when
    ``stream_options.include_usage`` is set."""
    return json.dumps(
        {
            "id": "chatcmpl-test",
            "choices": [],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": completion_tokens,
                "total_tokens": 11 + completion_tokens,
            },
        }
    )


class _StubClock:
    """Monotonic-ish clock that advances only when explicitly bumped.

    Lets the tests deterministically cross / not cross the
    ``emit_interval`` rate-limit boundary.
    """

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _drain(handle, n: int, timeout: float = 1.0):
    """Pull exactly ``n`` events off a broadcaster handle."""

    async def run():
        ai = handle.__aiter__()
        out = []
        for _ in range(n):
            out.append(await asyncio.wait_for(ai.__anext__(), timeout=timeout))
        return out

    return asyncio.run(run())


# ── Tests ───────────────────────────────────────────────────────────


def test_inert_without_broadcaster():
    """No broadcaster → tracker is a no-op (no exceptions, no state changes
    from emit calls)."""
    t = _PassthroughTokenTracker(broadcaster=None, session_id="s1")
    t.ingest(_content_chunk("hello"))
    t.ingest(_usage_chunk(5))
    assert t.tok_count == 0
    assert t.final_completion_tokens is None


def test_inert_without_session_id():
    """Even with a broadcaster, an unset session_id disables emission so
    we don't pollute every other open chip."""

    async def run():
        b = TelemetryBroadcaster()
        handle = await b.subscribe(
            SubscriptionSpec(event_types={"tokens"}, session_id=None)
        )
        t = _PassthroughTokenTracker(broadcaster=b, session_id=None)
        t.ingest(_content_chunk("hello"))
        t.ingest(_usage_chunk(5))
        ai = handle.__aiter__()
        try:
            await asyncio.wait_for(ai.__anext__(), timeout=0.2)
            return "delivered"
        except asyncio.TimeoutError:
            return "timeout"
        finally:
            handle.close()

    assert asyncio.run(run()) == "timeout"


def test_content_chunk_increments_count_and_emits():
    """A single content chunk → tok_count = 1, chars_streamed = len(content),
    one ``tokens`` event with iteration=0."""

    async def run():
        b = TelemetryBroadcaster()
        handle = await b.subscribe(
            SubscriptionSpec(event_types={"tokens"}, session_id="s1")
        )
        clk = _StubClock()
        t = _PassthroughTokenTracker(
            broadcaster=b, session_id="s1", emit_interval=0.25, now_fn=clk
        )
        t.ingest(_content_chunk("Hello"))
        ai = handle.__aiter__()
        ev = await asyncio.wait_for(ai.__anext__(), timeout=1.0)
        handle.close()
        return ev, t

    ev, t = asyncio.run(run())
    assert ev["type"] == "tokens"
    assert ev["session_id"] == "s1"
    assert ev["iteration"] == 0
    assert ev["pre_filter_tokens"] == 1
    assert ev["post_filter_tokens"] == 1
    # First chunk → first_content_ts == now → elapsed == 0 → tps None.
    assert ev["tps"] is None
    assert ev["prompt_tps"] is None
    assert t.tok_count == 1
    assert t.chars_streamed == len("Hello")


def test_tool_call_chunk_counts_as_token():
    """``delta.tool_calls`` chunks (model is generating the function-call
    JSON) are output tokens too — must increment the count."""

    async def run():
        b = TelemetryBroadcaster()
        handle = await b.subscribe(
            SubscriptionSpec(event_types={"tokens"}, session_id="s1")
        )
        clk = _StubClock()
        t = _PassthroughTokenTracker(
            broadcaster=b, session_id="s1", emit_interval=0.25, now_fn=clk
        )
        t.ingest(_tool_call_chunk('{"city":'))
        # Need to drain to make sure event was emitted.
        ai = handle.__aiter__()
        ev = await asyncio.wait_for(ai.__anext__(), timeout=1.0)
        handle.close()
        return ev, t

    ev, t = asyncio.run(run())
    assert t.tok_count == 1
    # Tool-call deltas don't carry ``content``, so chars_streamed stays 0.
    assert t.chars_streamed == 0
    assert ev["pre_filter_tokens"] == 1


def test_emissions_are_rate_limited():
    """Multiple content chunks within ``emit_interval`` produce ONE event;
    advancing the clock past the interval triggers a second."""

    async def run():
        b = TelemetryBroadcaster()
        handle = await b.subscribe(
            SubscriptionSpec(event_types={"tokens"}, session_id="s1")
        )
        clk = _StubClock()
        t = _PassthroughTokenTracker(
            broadcaster=b, session_id="s1", emit_interval=0.25, now_fn=clk
        )
        # Three chunks with no clock advance — should emit only once.
        t.ingest(_content_chunk("a"))
        t.ingest(_content_chunk("b"))
        t.ingest(_content_chunk("c"))
        # Advance past the rate-limit window and ingest one more.
        clk.advance(0.30)
        t.ingest(_content_chunk("d"))
        # Drain whatever has accumulated.
        ai = handle.__aiter__()
        out = []
        try:
            while True:
                out.append(await asyncio.wait_for(ai.__anext__(), timeout=0.2))
        except asyncio.TimeoutError:
            pass
        handle.close()
        return out, t

    out, t = asyncio.run(run())
    assert t.tok_count == 4
    # Exactly two emissions: one for the first chunk (last_emit_ts was 0,
    # so the rate-limit gate is open), one for the chunk after the
    # clock advance.
    assert len(out) == 2
    assert out[0]["pre_filter_tokens"] == 1
    assert out[1]["pre_filter_tokens"] == 4


def test_usage_chunk_reconciles_and_force_emits():
    """The trailing ``usage`` chunk must override the interpolated count
    AND force-emit, even within the rate-limit window."""

    async def run():
        b = TelemetryBroadcaster()
        handle = await b.subscribe(
            SubscriptionSpec(event_types={"tokens"}, session_id="s1")
        )
        clk = _StubClock()
        t = _PassthroughTokenTracker(
            broadcaster=b, session_id="s1", emit_interval=0.25, now_fn=clk
        )
        # Stream three chunks (one emit fires from chunk #1).
        t.ingest(_content_chunk("a"))
        t.ingest(_content_chunk("b"))
        t.ingest(_content_chunk("c"))
        # Final usage chunk arrives within the rate-limit window and
        # reports a count higher than the interpolation (prompt-eval
        # plus completion all rolled up; or interpolation undercounted
        # because llama-server batched a chunk with multiple tokens).
        t.ingest(_usage_chunk(7))
        ai = handle.__aiter__()
        out = []
        try:
            while True:
                out.append(await asyncio.wait_for(ai.__anext__(), timeout=0.2))
        except asyncio.TimeoutError:
            pass
        handle.close()
        return out, t

    out, t = asyncio.run(run())
    assert t.final_completion_tokens == 7
    # Two emissions: chunk #1 (pre=1) + forced reconciliation (pre=7).
    assert len(out) == 2
    assert out[0]["pre_filter_tokens"] == 1
    assert out[1]["pre_filter_tokens"] == 7
    assert out[1]["post_filter_tokens"] == 7


def test_tps_uses_first_content_timestamp():
    """``tps`` = pre_filter_tokens / (now - first_content_ts). Driven by
    the stub clock so we can assert exact values."""

    async def run():
        b = TelemetryBroadcaster()
        handle = await b.subscribe(
            SubscriptionSpec(event_types={"tokens"}, session_id="s1")
        )
        clk = _StubClock()
        t = _PassthroughTokenTracker(
            broadcaster=b, session_id="s1", emit_interval=0.0, now_fn=clk
        )
        # First content sets first_content_ts. Tps still None on chunk #1
        # (elapsed == 0).
        t.ingest(_content_chunk("a"))
        # Advance 1 second, ingest 9 more chunks. Last chunk's emit
        # should report tps = 10 / 1.0.
        for _ in range(9):
            clk.advance(0.1)
            t.ingest(_content_chunk("x"))
        ai = handle.__aiter__()
        last = None
        try:
            while True:
                last = await asyncio.wait_for(ai.__anext__(), timeout=0.2)
        except asyncio.TimeoutError:
            pass
        handle.close()
        return last, t

    last, t = asyncio.run(run())
    assert t.tok_count == 10
    assert last["pre_filter_tokens"] == 10
    # 10 tokens over ~0.9s of advance after the first chunk = ~11.1 t/s.
    # First chunk set first_content_ts at t=1000.0, last advance puts
    # clock at t=1000.9 → elapsed 0.9 → tps 10/0.9.
    assert last["tps"] is not None
    assert abs(last["tps"] - (10 / 0.9)) < 1e-6


def test_done_marker_is_ignored():
    """``[DONE]`` must not blow up or be parsed as JSON."""
    b = TelemetryBroadcaster()
    t = _PassthroughTokenTracker(broadcaster=b, session_id="s1")
    # No exception, no count change.
    t.ingest("[DONE]")
    t.ingest("")
    assert t.tok_count == 0


def test_malformed_json_swallowed():
    """A garbled chunk must not raise — telemetry is best-effort."""
    b = TelemetryBroadcaster()
    t = _PassthroughTokenTracker(broadcaster=b, session_id="s1")
    t.ingest("{not valid json")
    t.ingest("null")  # Valid JSON but not a dict.
    t.ingest(json.dumps({"choices": "wrong-type"}))  # choices not a list.
    assert t.tok_count == 0


def test_empty_choices_doesnt_count():
    """A chunk with empty ``choices`` (possible during model warm-up
    or finish_reason-only chunks) must not increment the count."""
    b = TelemetryBroadcaster()
    t = _PassthroughTokenTracker(broadcaster=b, session_id="s1")
    # finish_reason-only chunk: choices present but delta is empty.
    t.ingest(
        json.dumps(
            {"id": "x", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        )
    )
    assert t.tok_count == 0


def test_session_id_filter_isolates_streams():
    """Two concurrent sessions: tracker for s1 must not deliver events to
    a subscriber filtering on s2."""

    async def run():
        b = TelemetryBroadcaster()
        handle_s2 = await b.subscribe(
            SubscriptionSpec(event_types={"tokens"}, session_id="s2")
        )
        clk = _StubClock()
        t = _PassthroughTokenTracker(
            broadcaster=b, session_id="s1", emit_interval=0.0, now_fn=clk
        )
        t.ingest(_content_chunk("hello"))
        ai = handle_s2.__aiter__()
        try:
            await asyncio.wait_for(ai.__anext__(), timeout=0.2)
            return "leaked"
        except asyncio.TimeoutError:
            return "isolated"
        finally:
            handle_s2.close()

    assert asyncio.run(run()) == "isolated"
