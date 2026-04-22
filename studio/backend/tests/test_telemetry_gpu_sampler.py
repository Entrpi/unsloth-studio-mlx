# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the GPU sampler fallback chain.

We don't assert on real sampled values (that would require Apple
Silicon hardware + MLX at test time). Instead we mock each probe
method and verify ``_pick_source`` picks the first one that succeeds.
"""

import ctypes

import pytest

from core.telemetry import gpu_sampler as gpu_sampler_module
from core.telemetry.gpu_sampler import GpuSampler


def test_fallback_picks_ioreport_when_available(monkeypatch):
    s = GpuSampler()
    monkeypatch.setattr(s, "_probe_ioreport", lambda: True)
    monkeypatch.setattr(s, "_probe_iokit", lambda: True)
    monkeypatch.setattr(s, "_probe_mlx_mem", lambda: True)
    monkeypatch.setattr(s, "_probe_powermetrics", lambda: True)
    monkeypatch.delenv("STUDIO_TELEMETRY_GPU_SOURCE", raising=False)
    assert s._pick_source() == "ioreport"


def test_fallback_picks_iokit_when_ioreport_missing(monkeypatch):
    s = GpuSampler()
    monkeypatch.setattr(s, "_probe_ioreport", lambda: False)
    monkeypatch.setattr(s, "_probe_iokit", lambda: True)
    monkeypatch.setattr(s, "_probe_mlx_mem", lambda: True)
    monkeypatch.setattr(s, "_probe_powermetrics", lambda: True)
    monkeypatch.delenv("STUDIO_TELEMETRY_GPU_SOURCE", raising=False)
    assert s._pick_source() == "iokit"


def test_fallback_picks_mlx_mem_when_ioreport_and_iokit_missing(monkeypatch):
    s = GpuSampler()
    monkeypatch.setattr(s, "_probe_ioreport", lambda: False)
    monkeypatch.setattr(s, "_probe_iokit", lambda: False)
    monkeypatch.setattr(s, "_probe_mlx_mem", lambda: True)
    monkeypatch.setattr(s, "_probe_powermetrics", lambda: True)
    monkeypatch.delenv("STUDIO_TELEMETRY_GPU_SOURCE", raising=False)
    assert s._pick_source() == "mlx_mem"


def test_fallback_picks_powermetrics_when_others_missing(monkeypatch):
    s = GpuSampler()
    monkeypatch.setattr(s, "_probe_ioreport", lambda: False)
    monkeypatch.setattr(s, "_probe_iokit", lambda: False)
    monkeypatch.setattr(s, "_probe_mlx_mem", lambda: False)
    monkeypatch.setattr(s, "_probe_powermetrics", lambda: True)
    monkeypatch.delenv("STUDIO_TELEMETRY_GPU_SOURCE", raising=False)
    assert s._pick_source() == "powermetrics"


def test_fallback_unavailable_when_all_fail(monkeypatch):
    s = GpuSampler()
    monkeypatch.setattr(s, "_probe_ioreport", lambda: False)
    monkeypatch.setattr(s, "_probe_iokit", lambda: False)
    monkeypatch.setattr(s, "_probe_mlx_mem", lambda: False)
    monkeypatch.setattr(s, "_probe_powermetrics", lambda: False)
    monkeypatch.delenv("STUDIO_TELEMETRY_GPU_SOURCE", raising=False)
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


def test_env_override_forces_mlx_mem_even_when_ioreport_works(monkeypatch):
    s = GpuSampler()
    monkeypatch.setattr(s, "_probe_ioreport", lambda: True)
    monkeypatch.setattr(s, "_probe_mlx_mem", lambda: True)
    monkeypatch.setenv("STUDIO_TELEMETRY_GPU_SOURCE", "mlx_mem")
    assert s._pick_source() == "mlx_mem"


def test_env_override_falls_back_to_auto_when_probe_fails(monkeypatch):
    s = GpuSampler()
    monkeypatch.setattr(s, "_probe_ioreport", lambda: True)
    monkeypatch.setattr(s, "_probe_iokit", lambda: False)
    monkeypatch.setattr(s, "_probe_mlx_mem", lambda: False)
    monkeypatch.setattr(s, "_probe_powermetrics", lambda: False)
    # Asked for iokit, probe returns False -> falls through to
    # ioreport via auto chain.
    monkeypatch.setenv("STUDIO_TELEMETRY_GPU_SOURCE", "iokit")
    assert s._pick_source() == "ioreport"


def test_probe_ioreport_returns_false_when_dlopen_raises(monkeypatch):
    """Intel Macs / containers where libIOReport.dylib is absent."""
    original_cdll = ctypes.CDLL

    def _raise_on_ioreport(name, *args, **kwargs):
        if "IOReport" in str(name):
            raise OSError("image not found")
        return original_cdll(name, *args, **kwargs)

    monkeypatch.setattr(ctypes, "CDLL", _raise_on_ioreport)
    s = GpuSampler()
    assert s._probe_ioreport() is False
