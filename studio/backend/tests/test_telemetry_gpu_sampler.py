# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the cross-platform GPU sampler.

We don't assert on real sampled values (that would require the
underlying hardware + drivers installed at test time). Instead we
mock each probe and reader and verify:

* ``_candidate_sources`` returns the right ordered list per platform.
* ``_pick_source`` picks the first probe that succeeds.
* Each reader produces the expected wire shape.
* Graceful degradation: reader raising returns an all-None dict; env
  override to a missing backend falls back to auto.
* WSL detection skips the Linux sysfs paths.
* ``CUDA_VISIBLE_DEVICES`` selects the right device index.
"""

import ctypes
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# Import the MODULE (not the singleton — ``core.telemetry.gpu_sampler``
# resolves to the ``GpuSampler`` instance re-exported in
# ``core.telemetry/__init__.py``, which is NOT what we want to monkeypatch
# module-level helpers on). ``sys.modules`` gives us the real module
# object after the import; ``from ... import`` also works because the
# function-level names are bound directly.
from core.telemetry.gpu_sampler import GpuSampler, _cuda_visible_index, _is_wsl
gpu_sampler_module = sys.modules["core.telemetry.gpu_sampler"]


# ── Existing fallback-chain tests (Apple Silicon baseline) ───────────


def test_fallback_picks_ioreport_when_available(monkeypatch):
    s = GpuSampler()
    monkeypatch.setattr(s, "_probe_ioreport", lambda: True)
    monkeypatch.setattr(s, "_probe_iokit", lambda: True)
    monkeypatch.setattr(s, "_probe_mlx_mem", lambda: True)
    monkeypatch.setattr(s, "_probe_powermetrics", lambda: True)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    monkeypatch.delenv("STUDIO_TELEMETRY_GPU_SOURCE", raising=False)
    assert s._pick_source() == "ioreport"


def test_fallback_picks_iokit_when_ioreport_missing(monkeypatch):
    s = GpuSampler()
    monkeypatch.setattr(s, "_probe_ioreport", lambda: False)
    monkeypatch.setattr(s, "_probe_iokit", lambda: True)
    monkeypatch.setattr(s, "_probe_mlx_mem", lambda: True)
    monkeypatch.setattr(s, "_probe_powermetrics", lambda: True)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    monkeypatch.delenv("STUDIO_TELEMETRY_GPU_SOURCE", raising=False)
    assert s._pick_source() == "iokit"


def test_fallback_picks_mlx_mem_when_ioreport_and_iokit_missing(monkeypatch):
    s = GpuSampler()
    monkeypatch.setattr(s, "_probe_ioreport", lambda: False)
    monkeypatch.setattr(s, "_probe_iokit", lambda: False)
    monkeypatch.setattr(s, "_probe_mlx_mem", lambda: True)
    monkeypatch.setattr(s, "_probe_powermetrics", lambda: True)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    monkeypatch.delenv("STUDIO_TELEMETRY_GPU_SOURCE", raising=False)
    assert s._pick_source() == "mlx_mem"


def test_fallback_picks_powermetrics_when_others_missing(monkeypatch):
    s = GpuSampler()
    monkeypatch.setattr(s, "_probe_ioreport", lambda: False)
    monkeypatch.setattr(s, "_probe_iokit", lambda: False)
    monkeypatch.setattr(s, "_probe_mlx_mem", lambda: False)
    monkeypatch.setattr(s, "_probe_powermetrics", lambda: True)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    monkeypatch.delenv("STUDIO_TELEMETRY_GPU_SOURCE", raising=False)
    assert s._pick_source() == "powermetrics"


def test_fallback_unavailable_when_all_fail(monkeypatch):
    s = GpuSampler()
    for p in (
        "_probe_ioreport", "_probe_iokit", "_probe_mlx_mem",
        "_probe_powermetrics", "_probe_pynvml", "_probe_amdgpu_sysfs",
        "_probe_intel_sysfs", "_probe_pdh", "_probe_nvidia_smi_cli",
    ):
        monkeypatch.setattr(s, p, lambda: False)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
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
    monkeypatch.setattr(s, "_probe_pynvml", lambda: False)
    monkeypatch.setattr(s, "_probe_amdgpu_sysfs", lambda: False)
    monkeypatch.setattr(s, "_probe_intel_sysfs", lambda: False)
    monkeypatch.setattr(s, "_probe_pdh", lambda: False)
    monkeypatch.setattr(s, "_probe_nvidia_smi_cli", lambda: False)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
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


# ── Platform-dispatch tests ───────────────────────────────────────────


def test_candidate_sources_apple_silicon(monkeypatch):
    s = GpuSampler()
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    monkeypatch.delenv("STUDIO_TELEMETRY_GPU_SOURCE", raising=False)
    assert s._candidate_sources() == [
        "ioreport", "iokit", "mlx_mem", "powermetrics",
    ]


def test_candidate_sources_intel_mac_is_empty(monkeypatch):
    s = GpuSampler()
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    monkeypatch.delenv("STUDIO_TELEMETRY_GPU_SOURCE", raising=False)
    assert s._candidate_sources() == []


def test_candidate_sources_native_linux(monkeypatch):
    s = GpuSampler()
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(gpu_sampler_module, "_is_wsl", lambda: False)
    monkeypatch.delenv("STUDIO_TELEMETRY_GPU_SOURCE", raising=False)
    assert s._candidate_sources() == [
        "pynvml", "amdgpu_sysfs", "intel_sysfs", "nvidia_smi_cli",
    ]


def test_candidate_sources_wsl_skips_sysfs(monkeypatch):
    s = GpuSampler()
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(gpu_sampler_module, "_is_wsl", lambda: True)
    monkeypatch.delenv("STUDIO_TELEMETRY_GPU_SOURCE", raising=False)
    assert s._candidate_sources() == ["pynvml", "nvidia_smi_cli"]


def test_candidate_sources_windows(monkeypatch):
    s = GpuSampler()
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("STUDIO_TELEMETRY_GPU_SOURCE", raising=False)
    assert s._candidate_sources() == ["pynvml", "pdh", "nvidia_smi_cli"]


def test_candidate_sources_env_override_single_entry(monkeypatch):
    s = GpuSampler()
    monkeypatch.setenv("STUDIO_TELEMETRY_GPU_SOURCE", "pynvml")
    assert s._candidate_sources() == ["pynvml"]


def test_is_wsl_detects_microsoft_release(tmp_path, monkeypatch):
    # Simulate the read directly — monkeypatch open for the osrelease
    # path. Using a factory lets us test both the positive and
    # negative cases cleanly.
    real_open = open

    def fake_open_microsoft(path, *args, **kwargs):
        if str(path) == "/proc/sys/kernel/osrelease":
            from io import StringIO
            return StringIO("5.15.167.4-microsoft-standard-WSL2\n")
        return real_open(path, *args, **kwargs)

    def fake_open_native(path, *args, **kwargs):
        if str(path) == "/proc/sys/kernel/osrelease":
            from io import StringIO
            return StringIO("6.6.0-generic\n")
        return real_open(path, *args, **kwargs)

    import builtins
    monkeypatch.setattr(builtins, "open", fake_open_microsoft)
    assert _is_wsl() is True
    monkeypatch.setattr(builtins, "open", fake_open_native)
    assert _is_wsl() is False


# ── CUDA_VISIBLE_DEVICES handling ─────────────────────────────────────


def test_cuda_visible_index_honours_first_entry(monkeypatch):
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert _cuda_visible_index() == 2


def test_cuda_visible_index_defaults_to_zero(monkeypatch):
    for v in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(v, raising=False)
    assert _cuda_visible_index() == 0


def test_cuda_visible_index_uuid_falls_back_to_zero(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abcd1234")
    assert _cuda_visible_index() == 0


def test_cuda_visible_index_honours_hip(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    assert _cuda_visible_index() == 1


# ── pynvml probe + read ──────────────────────────────────────────────


def _make_fake_pynvml(util=42, used=5_000_000_000, total=24_000_000_000, clock=1830):
    """Build a MagicMock that looks like the pynvml module surface we
    touch. Used across multiple tests so keep it standalone."""
    fake = types.SimpleNamespace()

    class _Rates:
        gpu = util
    class _Mem:
        pass
    mem = _Mem()
    mem.used = used
    mem.total = total

    fake.nvmlInit = MagicMock()
    fake.nvmlShutdown = MagicMock()
    fake.nvmlDeviceGetCount = MagicMock(return_value=1)
    fake.nvmlDeviceGetHandleByIndex = MagicMock(return_value="HANDLE")
    fake.nvmlDeviceGetName = MagicMock(return_value="NVIDIA GeForce RTX 4090")
    fake.nvmlDeviceGetUtilizationRates = MagicMock(return_value=_Rates())
    fake.nvmlDeviceGetMemoryInfo = MagicMock(return_value=mem)
    fake.nvmlDeviceGetClockInfo = MagicMock(return_value=clock)
    fake.NVML_CLOCK_SM = 1
    return fake


def test_probe_pynvml_success(monkeypatch):
    fake = _make_fake_pynvml()
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    # Reset the module-level flag so the test exercises nvmlInit.
    monkeypatch.setattr(gpu_sampler_module, "_pynvml_initialised", False)
    for v in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(v, raising=False)

    s = GpuSampler()
    assert s._probe_pynvml() is True
    assert fake.nvmlInit.called
    fake.nvmlDeviceGetHandleByIndex.assert_called_with(0)


def test_probe_pynvml_honours_cuda_visible_devices(monkeypatch):
    fake = _make_fake_pynvml()
    fake.nvmlDeviceGetCount = MagicMock(return_value=4)
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    monkeypatch.setattr(gpu_sampler_module, "_pynvml_initialised", False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")

    s = GpuSampler()
    assert s._probe_pynvml() is True
    fake.nvmlDeviceGetHandleByIndex.assert_called_with(2)


def test_probe_pynvml_missing_module_returns_false(monkeypatch):
    # Force ImportError by removing the key and any cached version.
    monkeypatch.delitem(sys.modules, "pynvml", raising=False)

    import builtins
    original_import = builtins.__import__

    def _no_pynvml(name, *args, **kwargs):
        if name == "pynvml":
            raise ImportError("No module named 'pynvml'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_pynvml)
    s = GpuSampler()
    assert s._probe_pynvml() is False


def test_read_pynvml_produces_full_event(monkeypatch):
    fake = _make_fake_pynvml(util=42, used=5_000_000_000, total=24_000_000_000, clock=1830)
    s = GpuSampler()
    s._pynvml_module = fake
    s._pynvml_handle = "HANDLE"
    ev = s._read_pynvml()
    assert ev["source"] == "pynvml"
    assert ev["util_pct"] == 42.0
    assert ev["mem_used_gb"] == pytest.approx(5.0, abs=0.01)
    assert ev["mem_total_gb"] == pytest.approx(24.0, abs=0.01)
    assert ev["freq_mhz"] == 1830.0


def test_read_pynvml_degrades_on_exception():
    s = GpuSampler()
    fake = MagicMock()
    fake.nvmlDeviceGetUtilizationRates.side_effect = RuntimeError("driver gone")
    s._pynvml_module = fake
    s._pynvml_handle = "HANDLE"
    ev = s._read_pynvml()
    assert ev["source"] == "pynvml"
    assert ev["util_pct"] is None
    assert ev["mem_used_gb"] is None


# ── AMD sysfs probe + read ────────────────────────────────────────────


def _make_amdgpu_tree(tmp_path):
    """Build a fake /sys/class/drm tree with one AMD card."""
    drm = tmp_path / "drm"
    drm.mkdir()
    card = drm / "card0"
    card.mkdir()
    device = card / "device"
    device.mkdir()
    (device / "vendor").write_text("0x1002\n")
    (device / "gpu_busy_percent").write_text("77\n")
    (device / "mem_info_vram_used").write_text(str(3 * 1_000_000_000) + "\n")
    (device / "mem_info_vram_total").write_text(str(16 * 1_000_000_000) + "\n")
    return drm


def test_probe_amdgpu_sysfs_success(tmp_path, monkeypatch):
    drm = _make_amdgpu_tree(tmp_path)
    # Fake os.listdir / open for the sysfs path.
    real_listdir = __import__("os").listdir
    real_open = open

    def fake_listdir(path):
        if str(path) == "/sys/class/drm":
            return real_listdir(drm)
        return real_listdir(path)

    def fake_open(path, *args, **kwargs):
        p = str(path)
        if p.startswith("/sys/class/drm/"):
            p = p.replace("/sys/class/drm/", f"{drm}/")
        return real_open(p, *args, **kwargs)

    import builtins
    import os as _os
    monkeypatch.setattr(_os, "listdir", fake_listdir)
    monkeypatch.setattr(builtins, "open", fake_open)
    monkeypatch.setattr(sys, "platform", "linux")
    for v in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(v, raising=False)

    s = GpuSampler()
    assert s._probe_amdgpu_sysfs() is True
    assert s._amdgpu_card_path is not None
    ev = s._read_amdgpu_sysfs()
    assert ev["source"] == "amdgpu_sysfs"
    assert ev["util_pct"] == 77.0
    assert ev["mem_used_gb"] == pytest.approx(3.0, abs=0.01)
    assert ev["mem_total_gb"] == pytest.approx(16.0, abs=0.01)


def test_probe_amdgpu_sysfs_skips_nvidia_vendor(tmp_path, monkeypatch):
    drm = tmp_path / "drm"
    drm.mkdir()
    card = drm / "card0"
    card.mkdir()
    device = card / "device"
    device.mkdir()
    (device / "vendor").write_text("0x10de\n")
    (device / "gpu_busy_percent").write_text("12\n")

    real_listdir = __import__("os").listdir
    real_open = open

    def fake_listdir(path):
        if str(path) == "/sys/class/drm":
            return real_listdir(drm)
        return real_listdir(path)

    def fake_open(path, *args, **kwargs):
        p = str(path)
        if p.startswith("/sys/class/drm/"):
            p = p.replace("/sys/class/drm/", f"{drm}/")
        return real_open(p, *args, **kwargs)

    import builtins
    import os as _os
    monkeypatch.setattr(_os, "listdir", fake_listdir)
    monkeypatch.setattr(builtins, "open", fake_open)
    monkeypatch.setattr(sys, "platform", "linux")

    s = GpuSampler()
    assert s._probe_amdgpu_sysfs() is False


def test_probe_amdgpu_sysfs_skipped_off_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    s = GpuSampler()
    assert s._probe_amdgpu_sysfs() is False


# ── Intel sysfs probe + read ──────────────────────────────────────────


def _make_intel_tree(tmp_path, busy_ns=0):
    drm = tmp_path / "drm"
    drm.mkdir()
    card = drm / "card0"
    card.mkdir()
    device = card / "device"
    device.mkdir()
    (device / "vendor").write_text("0x8086\n")
    engine = device / "engine" / "rcs0"
    engine.mkdir(parents=True)
    (engine / "busy").write_text(str(busy_ns) + "\n")
    return drm


def test_probe_intel_sysfs_success(tmp_path, monkeypatch):
    drm = _make_intel_tree(tmp_path, busy_ns=1_000_000)
    import os as _os
    real_listdir = _os.listdir
    real_isfile = _os.path.isfile
    real_open = open

    def _remap(p):
        p = str(p)
        if p.startswith("/sys/class/drm/"):
            p = p.replace("/sys/class/drm/", f"{drm}/")
        return p

    def fake_listdir(path):
        p = str(path)
        if p == "/sys/class/drm":
            return real_listdir(drm)
        return real_listdir(_remap(p))

    def fake_isfile(path):
        return real_isfile(_remap(path))

    def fake_open(path, *args, **kwargs):
        return real_open(_remap(path), *args, **kwargs)

    import builtins
    monkeypatch.setattr(_os, "listdir", fake_listdir)
    monkeypatch.setattr(_os.path, "isfile", fake_isfile)
    monkeypatch.setattr(builtins, "open", fake_open)
    monkeypatch.setattr(sys, "platform", "linux")

    s = GpuSampler()
    assert s._probe_intel_sysfs() is True
    assert s._intel_engine_file is not None


def test_probe_intel_sysfs_skipped_off_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    s = GpuSampler()
    assert s._probe_intel_sysfs() is False


# ── PDH (Windows) probe + read ────────────────────────────────────────


def test_probe_pdh_skipped_off_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    s = GpuSampler()
    assert s._probe_pdh() is False


def test_probe_pdh_success_with_fake_module(monkeypatch):
    fake_pdh = MagicMock()
    fake_pdh.OpenQuery.return_value = "QUERY"
    fake_pdh.AddEnglishCounter.return_value = "COUNTER"
    monkeypatch.setitem(sys.modules, "win32pdh", fake_pdh)
    monkeypatch.setattr(sys, "platform", "win32")

    s = GpuSampler()
    assert s._probe_pdh() is True
    assert s._pdh_state is not None
    assert s._pdh_state["query"] == "QUERY"


def test_read_pdh_sums_instances(monkeypatch):
    fake_pdh = MagicMock()
    fake_pdh.PDH_FMT_DOUBLE = 0x200
    fake_pdh.GetFormattedCounterArray.return_value = (
        2,
        [("instance_0", 30.0), ("instance_1", 25.0)],
    )
    s = GpuSampler()
    s._pdh_state = {
        "module": fake_pdh,
        "query": "QUERY",
        "counter": "COUNTER",
    }
    ev = s._read_pdh()
    assert ev["source"] == "pdh"
    assert ev["util_pct"] == pytest.approx(55.0, abs=0.01)
    # PDH doesn't expose total — stays None.
    assert ev["mem_total_gb"] is None


# ── nvidia-smi CLI fallback ───────────────────────────────────────────


def test_probe_nvidia_smi_cli_uses_which(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None)

    class _FakeCompleted:
        returncode = 0
        stdout = "42"
        stderr = ""
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: _FakeCompleted())

    s = GpuSampler()
    assert s._probe_nvidia_smi_cli() is True
    assert s._nvsmi_path == "/usr/bin/nvidia-smi"


def test_probe_nvidia_smi_cli_absent(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    s = GpuSampler()
    assert s._probe_nvidia_smi_cli() is False


def test_read_nvidia_smi_cli_parses_csv(monkeypatch):
    class _FakeCompleted:
        returncode = 0
        stdout = "55, 3072, 24576\n"
        stderr = ""
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: _FakeCompleted())
    for v in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(v, raising=False)

    s = GpuSampler()
    s._nvsmi_path = "/usr/bin/nvidia-smi"
    ev = s._read_nvidia_smi_cli()
    assert ev["source"] == "nvidia_smi_cli"
    assert ev["util_pct"] == 55.0
    assert ev["mem_used_gb"] == pytest.approx(3.0, abs=0.01)
    assert ev["mem_total_gb"] == pytest.approx(24.0, abs=0.01)


# ── Dispatch table ────────────────────────────────────────────────────


def test_dispatch_read_picks_reader_by_source():
    s = GpuSampler()
    s._source = "pynvml"
    fake = _make_fake_pynvml(util=11)
    s._pynvml_module = fake
    s._pynvml_handle = "HANDLE"
    ev = s._dispatch_read()
    assert ev["source"] == "pynvml"
    assert ev["util_pct"] == 11.0


def test_dispatch_read_degrades_on_unknown_source():
    s = GpuSampler()
    s._source = "something_weird"
    ev = s._dispatch_read()
    assert ev["source"] == "something_weird"
    assert ev["util_pct"] is None


def test_dispatch_read_amdgpu_without_init_degrades():
    s = GpuSampler()
    s._source = "amdgpu_sysfs"
    # ``_amdgpu_card_path`` never set — the reader should return a
    # degraded event rather than raising.
    ev = s._dispatch_read()
    assert ev["source"] == "amdgpu_sysfs"
    assert ev["util_pct"] is None


def test_dispatch_read_intel_without_init_degrades():
    s = GpuSampler()
    s._source = "intel_sysfs"
    ev = s._dispatch_read()
    assert ev["source"] == "intel_sysfs"
    assert ev["util_pct"] is None


def test_probe_pynvml_device_name_logged(monkeypatch, caplog):
    """Startup log surfaces the device identifier, used for the DGX
    Spark / multi-box debugging case."""
    fake = _make_fake_pynvml()
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    monkeypatch.setattr(gpu_sampler_module, "_pynvml_initialised", False)
    for v in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(v, raising=False)

    # structlog wraps the stdlib logger, so caplog is a best-effort
    # check. We assert that the ``nvmlDeviceGetName`` call at least
    # fires — the log line format is exercised by the real path.
    s = GpuSampler()
    assert s._probe_pynvml() is True
    fake.nvmlDeviceGetName.assert_called_once()


def test_env_override_unknown_name_falls_back_to_auto(monkeypatch):
    """A typo like ``STUDIO_TELEMETRY_GPU_SOURCE=iorepoert`` should
    warn + walk the auto chain rather than hanging."""
    s = GpuSampler()
    monkeypatch.setattr(s, "_probe_ioreport", lambda: True)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    monkeypatch.setenv("STUDIO_TELEMETRY_GPU_SOURCE", "iorepoert")
    assert s._pick_source() == "ioreport"
