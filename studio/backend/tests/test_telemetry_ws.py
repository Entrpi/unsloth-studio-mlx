# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Integration tests for the ``/ws/telemetry`` WebSocket endpoint.

Uses Starlette's ``TestClient`` (via FastAPI) which supports WebSocket
routes. We build a minimal app that mounts only the telemetry router
so the test doesn't boot the full Studio backend (which has heavy
side-effects during lifespan).
"""

import asyncio
import json
import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.telemetry import broadcaster


def _build_app():
    """Minimal app with only the telemetry router mounted.

    The route's ``_authenticate_ws`` gets monkeypatched per-test so we
    don't need a real JWT flow.
    """
    from routes.telemetry import router

    app = FastAPI()
    app.include_router(router)
    return app


def test_ws_rejects_missing_token():
    app = _build_app()
    client = TestClient(app)
    with pytest.raises(Exception):
        with client.websocket_connect("/ws/telemetry"):
            pass


def test_ws_accepts_valid_token_and_subscription(monkeypatch):
    from routes import telemetry as tel_mod

    async def fake_auth(token):
        return "test-user" if token == "ok" else None

    monkeypatch.setattr(tel_mod, "_authenticate_ws", fake_auth)

    app = _build_app()
    client = TestClient(app)
    with client.websocket_connect("/ws/telemetry?token=ok") as ws:
        ws.send_text(json.dumps({"subscribe": ["gpu"]}))

        # Emit from a background thread so the subscribe message has
        # time to register on the event loop BEFORE the emit fires.
        # A sleep here would race; a small gate pattern is reliable.
        def _emit_after_delay():
            time.sleep(0.1)
            broadcaster.emit("gpu", {"util_pct": 55.0})

        threading.Thread(target=_emit_after_delay, daemon=True).start()

        data = ws.receive_text()
        event = json.loads(data)
        assert event["type"] == "gpu"
        assert event["util_pct"] == 55.0


def test_ws_session_filter(monkeypatch):
    from routes import telemetry as tel_mod

    async def fake_auth(token):
        return "test-user"

    monkeypatch.setattr(tel_mod, "_authenticate_ws", fake_auth)

    app = _build_app()
    client = TestClient(app)
    with client.websocket_connect("/ws/telemetry?token=ok") as ws:
        ws.send_text(json.dumps({
            "subscribe": ["tokens"],
            "session_id": "want",
        }))

        def _emit():
            time.sleep(0.1)
            # Wrong session — should be filtered.
            broadcaster.emit("tokens", {"pre_filter_tokens": 1}, session_id="other")
            # Right session — should arrive.
            broadcaster.emit("tokens", {"pre_filter_tokens": 2}, session_id="want")

        threading.Thread(target=_emit, daemon=True).start()

        data = ws.receive_text()
        event = json.loads(data)
        assert event["session_id"] == "want"
        assert event["pre_filter_tokens"] == 2


def test_ws_feature_flag_disabled(monkeypatch):
    from routes import telemetry as tel_mod

    monkeypatch.setenv("STUDIO_ENABLE_TELEMETRY_WS", "0")

    async def fake_auth(token):
        return "test-user"

    monkeypatch.setattr(tel_mod, "_authenticate_ws", fake_auth)

    app = _build_app()
    client = TestClient(app)
    with pytest.raises(Exception):
        with client.websocket_connect("/ws/telemetry?token=ok") as ws:
            # Server should close immediately with code 1008.
            ws.receive_text()
