# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for :mod:`core.telemetry.broadcaster`.

Covers:
    - fan-out to matching subscribers only
    - session_id filter semantics (global events fan out to all,
      session-bound events only to matching subscribers)
    - queue overflow drops OLDEST and keeps NEWEST events
    - ``emit`` is safe from a non-asyncio worker thread

The project doesn't run pytest-asyncio, so tests drive their own
event loop via ``asyncio.run``; each coroutine is fully self-contained.
"""

import asyncio
import threading
import time

import pytest

from core.telemetry.broadcaster import (
    TelemetryBroadcaster,
    SubscriptionSpec,
    _SUBSCRIBER_QUEUE_MAXSIZE,
)


def test_emit_fans_out_to_subscribers_of_matching_type():
    async def run():
        b = TelemetryBroadcaster()
        handle = await b.subscribe(SubscriptionSpec(event_types={"gpu"}))
        b.emit("gpu", {"util_pct": 42.0})
        ai = handle.__aiter__()
        event = await asyncio.wait_for(ai.__anext__(), timeout=1.0)
        handle.close()
        return event

    event = asyncio.run(run())
    assert event["type"] == "gpu"
    assert event["util_pct"] == 42.0


def test_mismatched_type_is_not_delivered():
    async def run():
        b = TelemetryBroadcaster()
        handle = await b.subscribe(SubscriptionSpec(event_types={"gpu"}))
        b.emit("tokens", {"iteration": 0})
        ai = handle.__aiter__()
        try:
            await asyncio.wait_for(ai.__anext__(), timeout=0.2)
            return "delivered"
        except asyncio.TimeoutError:
            return "timeout"
        finally:
            handle.close()

    assert asyncio.run(run()) == "timeout"


def test_session_filter_matches():
    async def run():
        b = TelemetryBroadcaster()
        sub_a = await b.subscribe(SubscriptionSpec(event_types={"tokens"}, session_id="A"))
        sub_b = await b.subscribe(SubscriptionSpec(event_types={"tokens"}, session_id="B"))
        b.emit("tokens", {"iteration": 1}, session_id="A")
        ai_a = sub_a.__aiter__()
        ai_b = sub_b.__aiter__()
        evt_a = await asyncio.wait_for(ai_a.__anext__(), timeout=1.0)
        try:
            await asyncio.wait_for(ai_b.__anext__(), timeout=0.2)
            b_got = True
        except asyncio.TimeoutError:
            b_got = False
        sub_a.close()
        sub_b.close()
        return evt_a, b_got

    evt_a, b_got = asyncio.run(run())
    assert evt_a["session_id"] == "A"
    assert b_got is False


def test_global_event_fans_out_regardless_of_session():
    async def run():
        b = TelemetryBroadcaster()
        sub_a = await b.subscribe(SubscriptionSpec(event_types={"gpu"}, session_id="A"))
        sub_b = await b.subscribe(SubscriptionSpec(event_types={"gpu"}, session_id="B"))
        b.emit("gpu", {"util_pct": 10})
        evt_a = await asyncio.wait_for(sub_a.__aiter__().__anext__(), timeout=1.0)
        evt_b = await asyncio.wait_for(sub_b.__aiter__().__anext__(), timeout=1.0)
        sub_a.close()
        sub_b.close()
        return evt_a, evt_b

    evt_a, evt_b = asyncio.run(run())
    assert evt_a["util_pct"] == 10
    assert evt_b["util_pct"] == 10


def test_no_session_filter_receives_everything():
    async def run():
        b = TelemetryBroadcaster()
        sub = await b.subscribe(SubscriptionSpec(event_types={"tokens"}))
        b.emit("tokens", {"i": 0}, session_id="A")
        b.emit("tokens", {"i": 1}, session_id="B")
        ai = sub.__aiter__()
        e1 = await asyncio.wait_for(ai.__anext__(), timeout=1.0)
        e2 = await asyncio.wait_for(ai.__anext__(), timeout=1.0)
        sub.close()
        return e1, e2

    e1, e2 = asyncio.run(run())
    assert {e1["session_id"], e2["session_id"]} == {"A", "B"}


def test_queue_overflow_drops_oldest_and_keeps_newest():
    async def run():
        b = TelemetryBroadcaster()
        sub = await b.subscribe(SubscriptionSpec(event_types={"tokens"}))
        for i in range(_SUBSCRIBER_QUEUE_MAXSIZE + 5):
            b.emit("tokens", {"i": i})
        await asyncio.sleep(0.05)

        received = []
        ai = sub.__aiter__()
        try:
            while True:
                evt = await asyncio.wait_for(ai.__anext__(), timeout=0.1)
                received.append(evt["i"])
        except asyncio.TimeoutError:
            pass
        sub.close()
        return received

    received = asyncio.run(run())
    assert len(received) <= _SUBSCRIBER_QUEUE_MAXSIZE
    assert max(received) == _SUBSCRIBER_QUEUE_MAXSIZE + 4


def test_emit_from_worker_thread_is_safe():
    async def run():
        b = TelemetryBroadcaster()
        sub = await b.subscribe(SubscriptionSpec(event_types={"tokens"}))

        def worker():
            for i in range(10):
                b.emit("tokens", {"i": i})
                time.sleep(0.005)

        t = threading.Thread(target=worker)
        t.start()
        received = []
        ai = sub.__aiter__()
        try:
            for _ in range(10):
                evt = await asyncio.wait_for(ai.__anext__(), timeout=1.0)
                received.append(evt["i"])
        finally:
            t.join()
            sub.close()
        return received

    assert asyncio.run(run()) == list(range(10))


def test_unsubscribe_stops_delivery():
    async def run():
        b = TelemetryBroadcaster()
        sub = await b.subscribe(SubscriptionSpec(event_types={"gpu"}))
        assert b.subscriber_count == 1
        sub.close()
        assert b.subscriber_count == 0
        # Emit after unsubscribe must not raise.
        b.emit("gpu", {"util_pct": 1})
        return True

    assert asyncio.run(run()) is True
