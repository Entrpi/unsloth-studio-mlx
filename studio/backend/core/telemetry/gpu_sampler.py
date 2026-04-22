# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Phase 3 — Apple-Silicon GPU sampler.

Fallback chain (tried in order, first-works wins):

    1. **iokit**    — pyobjc ``IOReport``/``IOAccelerator`` for
                      device-util %. No sudo required. Skipped when
                      pyobjc is not installed (the default on a bare
                      Studio environment).
    2. **mlx_mem**  — ``mlx.core.metal.get_active_memory()`` +
                      ``device_info()``. Always works on a box with
                      MLX installed, which is the Studio-supported
                      hardware baseline. Util % is synthesised from
                      the "is any session actively generating" flag:
                      100 while generating, 0 otherwise. Crude but
                      actionable.
    3. **powermetrics** — subprocess with sudo probe. Skipped unless
                      ``STUDIO_TELEMETRY_ALLOW_POWERMETRICS=1`` and
                      passwordless sudo is configured, because
                      prompting for a password would hang the app.
    4. **unavailable** — emits one ``{"source": "unavailable"}``
                      event and stops sampling.

Samples every 500ms (2 Hz); the emit rate is conservative so the
WebSocket doesn't get spammed and the sparkline stays smooth at a
human-perceivable cadence.
"""

from __future__ import annotations

import os
import threading
import time
from typing import TYPE_CHECKING, Optional

from loggers import get_logger

if TYPE_CHECKING:
    from core.telemetry.broadcaster import TelemetryBroadcaster

logger = get_logger(__name__)


_SAMPLE_INTERVAL_S = 0.5


class GpuSampler:
    """Daemon thread that polls GPU util and emits ``gpu`` events.

    Single global instance; started once at FastAPI startup.
    ``broadcaster`` is passed in at ``start()`` time so the sampler
    module doesn't directly import the broadcaster (avoids circular
    imports and keeps this module unit-testable in isolation).
    """

    def __init__(self) -> None:
        self._broadcaster: Optional["TelemetryBroadcaster"] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._source: str = "unknown"
        self._active_sessions: set = set()
        self._active_sessions_lock = threading.Lock()

    # ── Public API ───────────────────────────────────────────────────

    def start(self, broadcaster: "TelemetryBroadcaster") -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._broadcaster = broadcaster
        self._stop_event.clear()
        self._thread = threading.Thread(
            target = self._run,
            name = "gpu-sampler",
            daemon = True,
        )
        self._thread.start()
        logger.info("GPU sampler started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout = 2.0)
        self._thread = None
        self._broadcaster = None
        logger.info("GPU sampler stopped")

    @property
    def is_active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def source(self) -> str:
        return self._source

    def notify_session_active(self, session_id: Optional[str], active: bool) -> None:
        """Called by the inference hooks when a session starts/finishes
        generating. Used by the ``mlx_mem`` fallback to synthesise a
        binary util signal.
        """
        key = session_id or "__default"
        with self._active_sessions_lock:
            if active:
                self._active_sessions.add(key)
            else:
                self._active_sessions.discard(key)

    def _any_session_active(self) -> bool:
        with self._active_sessions_lock:
            return bool(self._active_sessions)

    # ── Backend probes ───────────────────────────────────────────────

    def _probe_iokit(self) -> bool:
        """Best-effort pyobjc IOKit probe. Returns True on success."""
        try:
            import Foundation  # noqa: F401
            import IOKit  # noqa: F401
            # Just importing isn't enough — would need a real probe to
            # confirm IORegistryEntry data is queryable. Left as a
            # scaffold; fall through to mlx_mem on bare environments.
            return False
        except ImportError:
            return False

    def _probe_mlx_mem(self) -> bool:
        try:
            import mlx.core as mx  # noqa: F401
            from mlx.core import metal as _metal  # noqa: F401
            return True
        except ImportError:
            return False

    def _probe_powermetrics(self) -> bool:
        if os.environ.get("STUDIO_TELEMETRY_ALLOW_POWERMETRICS", "").strip() not in (
            "1", "true", "True", "yes", "on",
        ):
            return False
        # Non-interactive probe: `sudo -n true` fails cleanly if
        # passwordless sudo isn't configured, without prompting.
        try:
            import subprocess
            rc = subprocess.run(
                ["sudo", "-n", "true"],
                capture_output = True,
                timeout = 2.0,
            ).returncode
            return rc == 0
        except (OSError, subprocess.SubprocessError):
            return False

    def _pick_source(self) -> str:
        if self._probe_iokit():
            return "iokit"
        if self._probe_mlx_mem():
            return "mlx_mem"
        if self._probe_powermetrics():
            return "powermetrics"
        return "unavailable"

    # ── Sampling ─────────────────────────────────────────────────────

    def _read_mlx_mem(self) -> dict:
        """Read GPU memory via MLX. util is a binary 100/0 proxy based
        on whether any session is actively generating.
        """
        try:
            import mlx.core as mx
            # Prefer the non-deprecated top-level ``mx.get_active_memory``
            # when available; fall back to ``mx.metal.get_active_memory``
            # for older mlx builds.
            try:
                active_bytes = int(mx.get_active_memory())
            except AttributeError:
                active_bytes = int(mx.metal.get_active_memory())
            try:
                info = mx.device_info()
            except Exception:
                info = mx.metal.device_info()  # deprecated fallback
            total_bytes = int(info.get("memory_size", 0)) or 0
            util = 100.0 if self._any_session_active() else 0.0
            return {
                "util_pct": util,
                "mem_used_gb": round(active_bytes / 1e9, 3),
                "mem_total_gb": round(total_bytes / 1e9, 3) if total_bytes else None,
                "source": "mlx_mem",
            }
        except Exception as exc:
            logger.debug("mlx_mem sample failed: %s", exc)
            return {
                "util_pct": None,
                "mem_used_gb": None,
                "mem_total_gb": None,
                "source": "mlx_mem",
            }

    def _run(self) -> None:
        assert self._broadcaster is not None
        self._source = self._pick_source()
        logger.info("GPU sampler source=%s", self._source)

        if self._source == "unavailable":
            # Emit once so clients can render "GPU unavailable".
            self._broadcaster.emit(
                "gpu",
                {
                    "ts": time.monotonic(),
                    "util_pct": None,
                    "mem_used_gb": None,
                    "mem_total_gb": None,
                    "source": "unavailable",
                },
            )
            return

        # Steady-state loop.
        while not self._stop_event.is_set():
            if self._source == "mlx_mem":
                payload = self._read_mlx_mem()
            else:
                # Placeholder for iokit / powermetrics implementations
                # — fall back to mlx_mem shape. Keeps the wire format
                # stable across sources.
                payload = self._read_mlx_mem()

            payload["ts"] = time.monotonic()
            try:
                self._broadcaster.emit("gpu", payload)
            except Exception as exc:
                logger.warning("GPU sampler emit raised: %s", exc)

            self._stop_event.wait(_SAMPLE_INTERVAL_S)


# Module-level singleton.
gpu_sampler: GpuSampler = GpuSampler()
