# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Phase 3 — live telemetry (GPU util + pre-filter token counts).

Streams go out over a separate ``/ws/telemetry`` WebSocket, NOT as an
SSE extension, because telemetry needs to outlive individual chat
turns, survive reconnects, and fan out to multiple tabs. The token
counter is *pre-filter* — it counts raw tokens from the MLX
``stream_generate`` output BEFORE ``strip_tool_markup`` /
``TOOL_XML_SIGNALS`` hold-back, so the user sees real work happening
during held-back reasoning where no visible text has changed.

Modules:

    broadcaster:    in-process pub/sub; thread-safe ``emit`` so the
                    MLX streaming hot path (worker thread) can push
                    events without blocking.
    gpu_sampler:    daemon thread, global-scope (not per-request)
                    Apple-Silicon GPU sampling with a fallback chain
                    (iokit -> mlx_mem -> powermetrics -> unavailable).
"""

from core.telemetry.broadcaster import TelemetryBroadcaster, broadcaster
from core.telemetry.gpu_sampler import GpuSampler, gpu_sampler

__all__ = [
    "TelemetryBroadcaster",
    "broadcaster",
    "GpuSampler",
    "gpu_sampler",
]
