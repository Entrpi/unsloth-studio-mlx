# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Phase 3 — ``/ws/telemetry`` WebSocket endpoint.

Protocol:

    1. Client connects with the JWT / API key as a ``?token=`` query
       parameter (browser WebSocket APIs can't set ``Authorization``
       headers, which is why we mirror the existing Bearer flow into
       the query string).
    2. Client sends a subscription frame as its first message:
       ``{"subscribe": ["gpu", "session", "tokens"], "session_id":
       "abc123" | null}``.
    3. Server streams JSON events until disconnect. Client may send
       ``{"unsubscribe": true}`` or simply close the socket.

Feature-flag via ``STUDIO_ENABLE_TELEMETRY_WS`` env var (default on,
``0`` / ``false`` / ``no`` / ``off`` to disable). Mirrors the Phase
1 / Phase 2 knob style.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.security import HTTPAuthorizationCredentials
from starlette.websockets import WebSocket, WebSocketDisconnect

from auth.authentication import _get_current_subject
from core.telemetry import broadcaster
from core.telemetry.broadcaster import SubscriptionSpec
from loggers import get_logger

logger = get_logger(__name__)

router = APIRouter()


def _telemetry_ws_enabled() -> bool:
    raw = os.environ.get("STUDIO_ENABLE_TELEMETRY_WS")
    if raw is None:
        return True
    return raw.strip() not in {"0", "false", "False", "no", "off"}


async def _authenticate_ws(token: Optional[str]) -> Optional[str]:
    """Reuse the REST ``get_current_subject`` dependency's guts.

    We can't depend on it directly because WebSockets don't flow
    through FastAPI's dependency-injection for headers the same way
    HTTP routes do. Reading the token from ``?token=`` and calling
    the private impl gives us identical JWT / API-key semantics.
    """
    if not token:
        return None
    credentials = HTTPAuthorizationCredentials(scheme = "Bearer", credentials = token)
    try:
        return await _get_current_subject(credentials, allow_password_change = False)
    except Exception as exc:
        logger.info("Telemetry WS auth failed: %s", exc)
        return None


@router.websocket("/ws/telemetry")
async def telemetry_ws(
    websocket: WebSocket,
    token: Optional[str] = Query(default = None),
) -> None:
    if not _telemetry_ws_enabled():
        await websocket.close(code = 1008, reason = "telemetry disabled")
        return

    subject = await _authenticate_ws(token)
    if subject is None:
        await websocket.close(code = 1008, reason = "unauthorized")
        return

    await websocket.accept()
    logger.info("Telemetry WS connected subject=%s", subject)

    handle = None
    forward_task: Optional[asyncio.Task] = None
    try:
        # First frame: subscription. Bounded wait so a silent client
        # doesn't pin a subscriber forever.
        try:
            raw = await asyncio.wait_for(websocket.receive_text(), timeout = 10.0)
        except asyncio.TimeoutError:
            await websocket.close(code = 1008, reason = "subscribe timeout")
            return

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await websocket.close(code = 1003, reason = "invalid json")
            return

        subscribe = msg.get("subscribe") if isinstance(msg, dict) else None
        if not isinstance(subscribe, list):
            await websocket.close(code = 1003, reason = "missing subscribe list")
            return
        event_types = {str(t) for t in subscribe if isinstance(t, str)}
        session_id = msg.get("session_id") if isinstance(msg, dict) else None
        if session_id is not None and not isinstance(session_id, str):
            session_id = None

        spec = SubscriptionSpec(event_types = event_types, session_id = session_id)
        handle = await broadcaster.subscribe(spec)

        async def _forward() -> None:
            async for event in handle:
                await websocket.send_text(json.dumps(event))

        forward_task = asyncio.create_task(_forward())

        # Read loop — accept unsubscribe / re-subscribe frames and
        # detect client disconnect.
        while True:
            recv = await websocket.receive_text()
            try:
                m = json.loads(recv)
            except json.JSONDecodeError:
                continue
            if isinstance(m, dict) and m.get("unsubscribe"):
                break
            if isinstance(m, dict) and isinstance(m.get("subscribe"), list):
                # Re-subscribe with updated filter.
                new_types = {str(t) for t in m["subscribe"] if isinstance(t, str)}
                new_session = m.get("session_id") if isinstance(m.get("session_id"), str) else None
                if handle is not None:
                    handle.close()
                if forward_task is not None:
                    forward_task.cancel()
                spec = SubscriptionSpec(event_types = new_types, session_id = new_session)
                handle = await broadcaster.subscribe(spec)

                async def _forward2() -> None:
                    async for event in handle:
                        await websocket.send_text(json.dumps(event))

                forward_task = asyncio.create_task(_forward2())
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("Telemetry WS error subject=%s: %s", subject, exc)
    finally:
        if forward_task is not None:
            forward_task.cancel()
        if handle is not None:
            handle.close()
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info("Telemetry WS disconnected subject=%s", subject)
