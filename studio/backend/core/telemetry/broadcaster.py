# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""In-process pub/sub for Phase 3 live telemetry.

``emit`` is synchronous and safe to call from any thread (the MLX
streaming loop runs in a worker thread; ``asyncio.to_thread`` hops
back to the event loop). Each subscriber owns a bounded
``asyncio.Queue``; on overflow we drop the OLDEST event and log a
warning once per subscriber — losing telemetry is acceptable, blocking
the inference path is not.

Subscription filter:

    {
        "event_types": {"gpu", "session", "tokens"},
        "session_id": "abc123" | None,
    }

Events with no ``session_id`` (e.g. the global GPU sparkline) go to
every subscriber that subscribed to that type regardless of their
session_id filter.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Optional, Set

from loggers import get_logger

logger = get_logger(__name__)


# Bounded per-subscriber queue. 256 events is ~64 seconds of GPU
# samples at 4 Hz plus the occasional token/session burst — plenty of
# slack before we start dropping.
_SUBSCRIBER_QUEUE_MAXSIZE = 256


@dataclass
class SubscriptionSpec:
    """Filter for which events a subscriber cares about."""

    event_types: Set[str] = field(default_factory = set)
    session_id: Optional[str] = None


@dataclass
class _Subscriber:
    sub_id: str
    spec: SubscriptionSpec
    queue: asyncio.Queue
    loop: asyncio.AbstractEventLoop
    dropped_warning_logged: bool = False


class TelemetryBroadcaster:
    """Thread-safe pub/sub over ``asyncio.Queue``s.

    ``emit`` may be called from any thread — it hops the enqueue
    operation onto the subscriber's event loop via
    ``loop.call_soon_threadsafe`` when we're not already in it.
    """

    def __init__(self) -> None:
        self._subscribers: Dict[str, _Subscriber] = {}
        self._lock = threading.Lock()

    # ── Public API ───────────────────────────────────────────────────

    def emit(
        self,
        event_type: str,
        payload: Dict[str, Any],
        *,
        session_id: Optional[str] = None,
    ) -> None:
        """Emit an event to every matching subscriber.

        Safe to call from any thread. Does NOT block — overflowing
        queues drop their oldest event.
        """
        event = {"type": event_type, **payload}
        if session_id is not None and "session_id" not in event:
            event["session_id"] = session_id

        with self._lock:
            # Snapshot so we don't hold the lock through fan-out.
            targets = [s for s in self._subscribers.values() if _matches(s.spec, event_type, session_id)]

        for sub in targets:
            try:
                sub.loop.call_soon_threadsafe(_enqueue_or_drop, sub, event)
            except RuntimeError:
                # Event loop already closed — subscriber is a zombie,
                # skip.  The WS handler's ``unsubscribe`` cleans up on
                # disconnect; this is a belt-and-braces guard.
                continue

    async def subscribe(self, spec: SubscriptionSpec) -> "_SubscriptionHandle":
        """Register a subscriber; returns a handle for iteration."""
        loop = asyncio.get_running_loop()
        sub = _Subscriber(
            sub_id = uuid.uuid4().hex,
            spec = spec,
            queue = asyncio.Queue(maxsize = _SUBSCRIBER_QUEUE_MAXSIZE),
            loop = loop,
        )
        with self._lock:
            self._subscribers[sub.sub_id] = sub
        logger.info(
            "Telemetry subscribe id=%s types=%s session=%s",
            sub.sub_id, sorted(spec.event_types), spec.session_id,
        )
        return _SubscriptionHandle(self, sub)

    def unsubscribe(self, sub_id: str) -> None:
        with self._lock:
            self._subscribers.pop(sub_id, None)
        logger.info("Telemetry unsubscribe id=%s", sub_id)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)


def _matches(spec: SubscriptionSpec, event_type: str, session_id: Optional[str]) -> bool:
    if event_type not in spec.event_types:
        return False
    # Global events (no session_id) fan out to everyone.
    if session_id is None:
        return True
    # Subscriber with no session filter receives everything.
    if spec.session_id is None:
        return True
    return spec.session_id == session_id


def _enqueue_or_drop(sub: _Subscriber, event: Dict[str, Any]) -> None:
    """Put an event on a subscriber's queue; drop oldest on overflow."""
    try:
        sub.queue.put_nowait(event)
    except asyncio.QueueFull:
        try:
            sub.queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            sub.queue.put_nowait(event)
        except asyncio.QueueFull:
            pass  # still full — give up on this event
        if not sub.dropped_warning_logged:
            sub.dropped_warning_logged = True
            logger.warning(
                "Telemetry subscriber %s queue overflowed — dropping "
                "oldest events (further overflows suppressed)",
                sub.sub_id,
            )


class _SubscriptionHandle:
    """Yielded by ``subscribe``; iterate via ``async for event in handle``."""

    def __init__(self, broadcaster: TelemetryBroadcaster, sub: _Subscriber) -> None:
        self._broadcaster = broadcaster
        self._sub = sub

    @property
    def sub_id(self) -> str:
        return self._sub.sub_id

    def __aiter__(self) -> AsyncIterator[Dict[str, Any]]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[Dict[str, Any]]:
        try:
            while True:
                event = await self._sub.queue.get()
                yield event
        finally:
            self._broadcaster.unsubscribe(self._sub.sub_id)

    def close(self) -> None:
        self._broadcaster.unsubscribe(self._sub.sub_id)


# Module-level singleton. Import the instance, not the class, from
# hot paths so emit() is a cheap attribute lookup.
broadcaster: TelemetryBroadcaster = TelemetryBroadcaster()
