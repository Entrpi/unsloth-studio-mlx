# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the GPU sampler fallback chain.

We don't assert on real sampled values (that would require Apple
Silicon hardware + MLX at test time). Instead we mock each probe
method and verify ``_pick_source`` picks the first one that succeeds.
"""

import pytest

from core.telemetry.gpu_sampler import GpuSampler


def test_fallback_picks_iokit_when_available(monkeypatch):
    s = GpuSampler()
    monkeypatch.setattr(s, "_probe_iokit", lambda: True)
    monkeypatch.setattr(s, "_probe_mlx_mem", lambda: True)
    monkeypatch.setattr(s, "_probe_powermetrics", lambda: True)
    assert s._pick_source() == "iokit"


def test_fallback_picks_mlx_mem_when_iokit_missing(monkeypatch):
    s = GpuSampler()
    monkeypatch.setattr(s, "_probe_iokit", lambda: False)
    monkeypatch.setattr(s, "_probe_mlx_mem", lambda: True)
    monkeypatch.setattr(s, "_probe_powermetrics", lambda: True)
    assert s._pick_source() == "mlx_mem"


def test_fallback_picks_powermetrics_when_others_missing(monkeypatch):
    s = GpuSampler()
    monkeypatch.setattr(s, "_probe_iokit", lambda: False)
    monkeypatch.setattr(s, "_probe_mlx_mem", lambda: False)
    monkeypatch.setattr(s, "_probe_powermetrics", lambda: True)
    assert s._pick_source() == "powermetrics"


def test_fallback_unavailable_when_all_fail(monkeypatch):
    s = GpuSampler()
    monkeypatch.setattr(s, "_probe_iokit", lambda: False)
    monkeypatch.setattr(s, "_probe_mlx_mem", lambda: False)
    monkeypatch.setattr(s, "_probe_powermetrics", lambda: False)
    assert s._pick_source() == "unavailable"


def test_notify_session_active_is_idempotent():
    s = GpuSampler()
    assert s._any_session_active() is False
    s.notify_session_active("A", True)
    s.notify_session_active("A", True)  # dup
    assert s._any_session_active() is True
    s.notify_session_active("A", False)
    assert s._any_session_active() is False


def test_powermetrics_probe_respects_env_flag(monkeypatch):
    s = GpuSampler()
    # Default: env flag unset -> skip probe entirely.
    monkeypatch.delenv("STUDIO_TELEMETRY_ALLOW_POWERMETRICS", raising=False)
    assert s._probe_powermetrics() is False
