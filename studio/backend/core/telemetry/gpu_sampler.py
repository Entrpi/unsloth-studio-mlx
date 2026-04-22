# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cross-platform GPU sampler.

This module owns the GPU telemetry pipeline — a single daemon thread
that samples a per-platform "best-available" backend at 2 Hz and
publishes ``gpu`` events on the telemetry broadcaster.

Per-platform primary + fallback chain (first probe that succeeds wins):

* **macOS Apple Silicon**  (arm64 / aarch64 darwin)
    ``ioreport → iokit → mlx_mem → powermetrics → unavailable``
* **macOS Intel**
    ``unavailable``  (no usable public API; IOReport GPU Stats is
    absent, IOKit GPU stats on x86_64 Macs are vendor-private)
* **Linux (x86_64 and aarch64, incl. DGX Spark / Grace-Hopper)**
    ``pynvml → amdgpu_sysfs → intel_sysfs → nvidia_smi_cli →
    unavailable``
* **WSL2**
    ``pynvml → nvidia_smi_cli → unavailable``  (sysfs GPU paths
    aren't populated inside WSL — the NVIDIA driver lives on the
    Windows side)
* **Windows (win32 / cygwin)**
    ``pynvml → pdh → nvidia_smi_cli → unavailable``

Every backend produces the same wire event shape::

    {
        "ts": <monotonic>,
        "util_pct": float|None,        # 0-100
        "mem_used_gb": float|None,
        "mem_total_gb": float|None,
        "source": "<backend-name>",
        "freq_mhz": float|None,        # optional, bonus info
    }

Any numeric field may be ``None`` when the backend can't report it
(e.g. Intel iGPUs don't expose VRAM totals).

Override chain: set ``STUDIO_TELEMETRY_GPU_SOURCE`` to a specific
backend name to force it. Anything else (empty, ``auto``) walks the
platform-appropriate fallback chain. The frontend hides the GPU chip
entirely when ``source == "unavailable"``.

Multi-GPU scope: this module currently emits the first *visible*
device (index 0 after honouring ``CUDA_VISIBLE_DEVICES`` /
``HIP_VISIBLE_DEVICES``). The reader API is structured so adding
multi-GPU later is a single change — readers return ``dict`` today,
they would return ``list[dict]`` and the broadcaster loop would emit
one event per device. See the comment in ``_run``.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import platform as _py_platform
import shutil
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING, Optional, Tuple

from loggers import get_logger

if TYPE_CHECKING:
    from core.telemetry.broadcaster import TelemetryBroadcaster

logger = get_logger(__name__)


_SAMPLE_INTERVAL_S = 0.5

# States counted as "idle" when summing residencies — everything else
# (``P1``, ``P2``, ``BUSY``, ...) is active GPU work.
_IOREPORT_IDLE_STATES = ("OFF", "IDLE", "DOWN")

# If we hit N consecutive read failures we bail out of the sampling
# loop — letting it spin forever on a broken backend would burn CPU
# and spam logs with no way for the user to notice.
_MAX_CONSECUTIVE_READ_FAILURES = 20

# Module-level guard so two GpuSampler starts don't nvmlInit twice
# (pynvml's init/shutdown pair isn't reentrant across a process).
_pynvml_init_lock = threading.Lock()
_pynvml_initialised: bool = False


# ── IOReport ctypes binding (Apple Silicon) ──────────────────────────
#
# Lazily initialised on first use; dlopen + CF symbol lookup only
# happens if the ioreport source is actually picked. Keeping this at
# module scope (rather than inside ``GpuSampler``) lets the same
# CFString constants / signatures be shared across reader instances
# and tests.


class _IOReportBinding:
    """Thin wrapper over ``libIOReport.dylib`` + CoreFoundation.

    All CFRetain/CFRelease discipline lives here. An instance owns:

    * the subscription handle (``subbed``)
    * the channel-dict we subscribe over (``chans``)
    * the cached ``"IOReportChannels"`` CFString used every tick
    * the prior-tick sample, kept as ``s1`` so each delta covers the
      500 ms between sampler ticks (not since-boot).

    ``close()`` CFReleases everything; ``__del__`` calls it as a
    safety net if the sampler thread exits without a clean stop.
    """

    def __init__(self) -> None:
        self._cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
        self._ior = ctypes.CDLL("/usr/lib/libIOReport.dylib")

        VP = ctypes.c_void_p
        u64, i64, i32 = ctypes.c_uint64, ctypes.c_int64, ctypes.c_int32
        self._kUTF8 = 0x08000100

        def _sig(fn, restype, *argtypes):
            fn.restype, fn.argtypes = restype, argtypes

        cf, ior = self._cf, self._ior
        _sig(cf.CFStringCreateWithCString, VP, VP, ctypes.c_char_p, ctypes.c_uint32)
        _sig(cf.CFRelease, None, VP)
        _sig(cf.CFDictionaryGetValue, VP, VP, VP)
        _sig(cf.CFDictionaryCreateMutableCopy, VP, VP, ctypes.c_long, VP)
        _sig(cf.CFArrayGetCount, ctypes.c_long, VP)
        _sig(cf.CFArrayGetValueAtIndex, VP, VP, ctypes.c_long)
        _sig(cf.CFStringGetCString, ctypes.c_bool, VP, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32)

        _sig(ior.IOReportCopyChannelsInGroup, VP, VP, VP, u64, u64, u64)
        _sig(ior.IOReportMergeChannels, None, VP, VP, VP)
        _sig(ior.IOReportCreateSubscription, VP, VP, VP, ctypes.POINTER(VP), u64, VP)
        _sig(ior.IOReportCreateSamples, VP, VP, VP, VP)
        _sig(ior.IOReportCreateSamplesDelta, VP, VP, VP, VP)
        _sig(ior.IOReportChannelGetGroup, VP, VP)
        _sig(ior.IOReportChannelGetSubGroup, VP, VP)
        _sig(ior.IOReportChannelGetChannelName, VP, VP)
        _sig(ior.IOReportStateGetCount, i32, VP)
        _sig(ior.IOReportStateGetNameForIndex, VP, VP, i32)
        _sig(ior.IOReportStateGetResidency, i64, VP, i32)

        self._chans: Optional[int] = None
        self._subbed: Optional[ctypes.c_void_p] = None
        self._channels_key: Optional[int] = None
        self._prev_sample: Optional[int] = None

        # Pick up the GPU Stats group.
        group_name = self._cfstring("GPU Stats")
        try:
            gpu = ior.IOReportCopyChannelsInGroup(group_name, None, 0, 0, 0)
        finally:
            cf.CFRelease(group_name)
        if not gpu:
            raise RuntimeError("IOReportCopyChannelsInGroup(\"GPU Stats\") returned NULL")

        self._chans = cf.CFDictionaryCreateMutableCopy(None, 0, gpu)
        cf.CFRelease(gpu)
        if not self._chans:
            raise RuntimeError("CFDictionaryCreateMutableCopy failed for GPU channels")

        subbed = ctypes.c_void_p()
        sub = ior.IOReportCreateSubscription(
            None, self._chans, ctypes.byref(subbed), 0, None,
        )
        if not sub or not subbed.value:
            raise RuntimeError("IOReportCreateSubscription failed")
        self._subbed = sub
        self._channels_key = self._cfstring("IOReportChannels")

    # ── helpers ──────────────────────────────────────────────────

    def _cfstring(self, py: str) -> int:
        return self._cf.CFStringCreateWithCString(None, py.encode(), self._kUTF8)

    def _from_cfstring(self, cfs: Optional[int]) -> str:
        if not cfs:
            return ""
        buf = ctypes.create_string_buffer(128)
        ok = self._cf.CFStringGetCString(cfs, buf, 128, self._kUTF8)
        return buf.value.decode() if ok else ""

    # ── public API ───────────────────────────────────────────────

    def sample_active_pct(self) -> Optional[float]:
        """Return GPU active-residency % since the previous call.

        On first call (no prior sample), takes a fresh sample, stores
        it as ``s1``, and returns ``None`` — the caller should render
        that as "no data yet" for one tick. Subsequent calls use the
        stored ``s1`` against a new ``s2`` so each reading covers the
        true inter-tick window (not since-boot).
        """
        ior, cf = self._ior, self._cf
        s2 = ior.IOReportCreateSamples(self._subbed, self._chans, None)
        if not s2:
            return None

        if self._prev_sample is None:
            self._prev_sample = s2
            return None

        s1 = self._prev_sample
        delta = ior.IOReportCreateSamplesDelta(s1, s2, None)
        cf.CFRelease(s1)
        self._prev_sample = s2

        if not delta:
            return None

        try:
            arr = cf.CFDictionaryGetValue(delta, self._channels_key)
            if not arr:
                return None
            n = cf.CFArrayGetCount(arr)
            for i in range(n):
                item = cf.CFArrayGetValueAtIndex(arr, i)
                name = self._from_cfstring(ior.IOReportChannelGetChannelName(item))
                if name != "GPUPH":
                    continue
                n_states = ior.IOReportStateGetCount(item)
                total = 0
                active = 0
                for k in range(n_states):
                    r = ior.IOReportStateGetResidency(item, k)
                    if r < 0:
                        continue
                    total += r
                    state_name = self._from_cfstring(
                        ior.IOReportStateGetNameForIndex(item, k)
                    )
                    if state_name not in _IOREPORT_IDLE_STATES:
                        active += r
                if total <= 0:
                    return 0.0
                return 100.0 * active / total
            return None
        finally:
            cf.CFRelease(delta)

    def close(self) -> None:
        cf = self._cf
        if self._prev_sample is not None:
            cf.CFRelease(self._prev_sample)
            self._prev_sample = None
        if self._subbed is not None:
            cf.CFRelease(self._subbed)
            self._subbed = None
        if self._chans is not None:
            cf.CFRelease(self._chans)
            self._chans = None
        if self._channels_key is not None:
            cf.CFRelease(self._channels_key)
            self._channels_key = None

    def __del__(self) -> None:  # best-effort safety net
        try:
            self.close()
        except Exception:
            pass


# ── Platform-dispatch helpers ────────────────────────────────────────


def _is_wsl() -> bool:
    """Detect WSL from ``/proc/sys/kernel/osrelease`` signature.

    WSL reports ``sys.platform == "linux"``, but the osrelease string
    contains ``microsoft`` or ``WSL``. Inside WSL, pynvml works via
    the Windows NVIDIA driver but ``/sys/class/drm`` GPU nodes are
    not populated, so we skip the sysfs paths.
    """
    try:
        with open("/proc/sys/kernel/osrelease", "r") as fh:
            content = fh.read().lower()
        return "microsoft" in content or "wsl" in content
    except OSError:
        return False


def _cuda_visible_index() -> int:
    """Return the integer device index selected by CUDA/HIP/ROCR
    visibility env vars, or ``0`` if unset/malformed.

    NVIDIA / AMD / ROCm all use the same "comma separated list,
    first entry wins" convention. We only honour the first entry —
    multi-GPU is deferred scope.
    """
    for var in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        raw = os.environ.get(var, "").strip()
        if not raw:
            continue
        first = raw.split(",")[0].strip()
        if not first:
            continue
        try:
            return int(first)
        except ValueError:
            # Non-integer (UUID-style) IDs: fall back to index 0.
            return 0
    return 0


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

        # Per-backend handles, populated lazily on the sampler thread
        # so a non-matching-platform instance never imports the
        # underlying library.
        self._ioreport: Optional[_IOReportBinding] = None
        self._pynvml_module = None  # cached module ref
        self._pynvml_handle = None  # nvmlDevice_t
        self._pynvml_device_index: int = 0
        self._amdgpu_card_path: Optional[str] = None  # /sys/class/drm/card0
        self._intel_card_path: Optional[str] = None
        self._intel_engine_file: Optional[str] = None
        self._intel_prev_busy_ns: Optional[int] = None
        self._intel_prev_wall_ns: Optional[int] = None
        self._pdh_state: Optional[dict] = None  # opaque PDH handles
        self._nvsmi_path: Optional[str] = None
        self._last_event: Optional[dict] = None  # for degrading on read failure

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
    #
    # Probes must be CHEAP (<100ms, no subprocesses sitting on PATH
    # scans, no large allocations) and MUST NOT raise. Probing a
    # wrong-platform backend returning False is the normal case —
    # log at DEBUG only.

    def _probe_ioreport(self) -> bool:
        """Attempt to dlopen libIOReport and open/close a full
        subscription. The binding constructor raises cleanly on Intel
        (dylib missing) or non-Apple-Silicon SKUs (``GPU Stats`` group
        NULL), which is exactly the probe contract we need.
        """
        try:
            binding = _IOReportBinding()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("ioreport probe failed: %s", exc)
            return False
        binding.close()
        return True

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
            rc = subprocess.run(
                ["sudo", "-n", "true"],
                capture_output = True,
                timeout = 2.0,
            ).returncode
            return rc == 0
        except (OSError, subprocess.SubprocessError):
            return False

    def _probe_pynvml(self) -> bool:
        """Import ``pynvml`` (provided by the ``nvidia-ml-py`` pip
        package), call ``nvmlInit``, obtain a handle for the visible
        device, and leave the module initialised for the reader.

        Uses a module-level lock + flag so concurrent / repeated probe
        calls don't call nvmlInit twice within one process.
        """
        try:
            import pynvml  # type: ignore
        except ImportError:
            return False
        except Exception as exc:  # pragma: no cover - unusual failure
            logger.debug("pynvml import raised: %s", exc)
            return False

        global _pynvml_initialised
        try:
            with _pynvml_init_lock:
                if not _pynvml_initialised:
                    pynvml.nvmlInit()
                    _pynvml_initialised = True
            idx = _cuda_visible_index()
            count = pynvml.nvmlDeviceGetCount()
            if count <= 0:
                return False
            if idx >= count:
                idx = 0
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
        except Exception as exc:
            logger.debug("pynvml probe failed: %s", exc)
            return False

        self._pynvml_module = pynvml
        self._pynvml_handle = handle
        self._pynvml_device_index = idx

        # Device-name log line — surfaces "which GPU are we watching"
        # for the DGX Spark / multi-GPU-box case.
        try:
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", "replace")
            logger.info("GPU sampler pynvml device=%s index=%d", name, idx)
        except Exception:  # pragma: no cover
            pass
        return True

    def _probe_amdgpu_sysfs(self) -> bool:
        """Walk ``/sys/class/drm/card*`` for an AMD GPU (PCI vendor
        ``0x1002``) that exposes ``gpu_busy_percent``. No deps."""
        if not sys.platform.startswith("linux"):
            return False
        try:
            entries = sorted(os.listdir("/sys/class/drm"))
        except OSError:
            return False
        # Respect HIP_VISIBLE_DEVICES — if set, skip cards we don't
        # want. Index ordering: ``cardN`` is the N-th card DRM sees,
        # and ROCm's enumeration order usually matches that for a
        # single-vendor box. For multi-vendor systems ordering is
        # fuzzy; users in that case should set STUDIO_TELEMETRY_GPU_SOURCE
        # explicitly.
        target_idx = _cuda_visible_index()
        seen = 0
        for entry in entries:
            if not entry.startswith("card") or "-" in entry:
                continue
            dev_path = os.path.join("/sys/class/drm", entry, "device")
            vendor_path = os.path.join(dev_path, "vendor")
            busy_path = os.path.join(dev_path, "gpu_busy_percent")
            try:
                with open(vendor_path, "r") as fh:
                    vendor = fh.read().strip().lower()
                if vendor != "0x1002":
                    continue
                # Probe readability.
                with open(busy_path, "r") as fh:
                    _ = int(fh.read().strip())
            except (OSError, ValueError):
                continue
            if seen == target_idx:
                self._amdgpu_card_path = dev_path
                logger.info("GPU sampler amdgpu_sysfs device=%s", entry)
                return True
            seen += 1
        return False

    def _probe_intel_sysfs(self) -> bool:
        """Find an Intel GPU (PCI vendor ``0x8086``) with an i915
        engine busy counter under ``/sys/class/drm/card*/engine/*/busy``.

        Requires kernel 5.19+ for the i915 engine-counter interface.
        """
        if not sys.platform.startswith("linux"):
            return False
        try:
            entries = sorted(os.listdir("/sys/class/drm"))
        except OSError:
            return False
        for entry in entries:
            if not entry.startswith("card") or "-" in entry:
                continue
            dev_path = os.path.join("/sys/class/drm", entry, "device")
            vendor_path = os.path.join(dev_path, "vendor")
            try:
                with open(vendor_path, "r") as fh:
                    vendor = fh.read().strip().lower()
                if vendor != "0x8086":
                    continue
            except OSError:
                continue

            # Prefer rcs0 (render engine) — same convention intel_gpu_top
            # uses as its headline "Render/3D" busy. Fall back to the
            # first engine we can read.
            engine_root = os.path.join(dev_path, "engine")
            candidates = []
            try:
                classes = sorted(os.listdir(engine_root))
            except OSError:
                continue
            preferred = [c for c in classes if c.startswith("rcs")]
            other = [c for c in classes if not c.startswith("rcs")]
            for cls in preferred + other:
                busy_f = os.path.join(engine_root, cls, "busy")
                if os.path.isfile(busy_f):
                    candidates.append(busy_f)
                    break
            if not candidates:
                continue
            # Validate readability.
            try:
                with open(candidates[0], "r") as fh:
                    int(fh.read().strip())
            except (OSError, ValueError):
                continue
            self._intel_card_path = dev_path
            self._intel_engine_file = candidates[0]
            logger.info("GPU sampler intel_sysfs device=%s", entry)
            return True
        return False

    def _probe_pdh(self) -> bool:
        """Windows PDH (Performance Data Helper) probe via pywin32.

        Uses the same ``\\GPU Engine(*engtype_3D)\\Utilization
        Percentage`` counter Task Manager reads. Opens a query, adds
        the wildcard counter using English names (locale-robust), and
        takes the first sample — PDH counters only produce values from
        the second ``PdhCollectQueryData`` call onwards, so the
        reader takes care of the real two-sample pattern.
        """
        if sys.platform not in ("win32", "cygwin"):
            return False
        try:
            import win32pdh  # type: ignore
        except ImportError:
            return False
        except Exception as exc:  # pragma: no cover
            logger.debug("win32pdh import raised: %s", exc)
            return False

        try:
            query = win32pdh.OpenQuery()
            try:
                counter = win32pdh.AddEnglishCounter(
                    query, r"\GPU Engine(*engtype_3D)\Utilization Percentage"
                )
            except Exception:
                # Older systems may only have ``AddCounter``; try that
                # but note it's not locale-safe.
                counter = win32pdh.AddCounter(
                    query, r"\GPU Engine(*engtype_3D)\Utilization Percentage"
                )
            win32pdh.CollectQueryData(query)
        except Exception as exc:
            logger.debug("pdh probe failed: %s", exc)
            try:
                win32pdh.CloseQuery(query)  # type: ignore
            except Exception:
                pass
            return False

        self._pdh_state = {
            "module": win32pdh,
            "query": query,
            "counter": counter,
        }
        logger.info("GPU sampler pdh device=GPU Engine(*engtype_3D)")
        return True

    def _probe_nvidia_smi_cli(self) -> bool:
        """Last-ditch NVIDIA fallback: call ``nvidia-smi`` each tick.

        Detecting it here caches the path so the reader doesn't shell
        out to ``which`` 2x / second.
        """
        path = shutil.which("nvidia-smi")
        if not path:
            return False
        # One probe invocation to confirm it actually runs; 2s cap so
        # we never hang startup.
        try:
            rc = subprocess.run(
                [path, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output = True,
                timeout = 2.0,
                text = True,
            )
            if rc.returncode != 0:
                return False
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug("nvidia-smi probe failed: %s", exc)
            return False
        self._nvsmi_path = path
        logger.info("GPU sampler nvidia_smi_cli path=%s", path)
        return True

    def _probe_map(self) -> dict:
        """Name → probe-callable. Kept here (not a class attr) so
        tests that monkeypatch ``_probe_X`` on an instance take effect.
        """
        return {
            "ioreport": self._probe_ioreport,
            "iokit": self._probe_iokit,
            "mlx_mem": self._probe_mlx_mem,
            "powermetrics": self._probe_powermetrics,
            "pynvml": self._probe_pynvml,
            "amdgpu_sysfs": self._probe_amdgpu_sysfs,
            "intel_sysfs": self._probe_intel_sysfs,
            "pdh": self._probe_pdh,
            "nvidia_smi_cli": self._probe_nvidia_smi_cli,
        }

    def _candidate_sources(self) -> list:
        """Ordered list of backend names to try for this platform.

        ``STUDIO_TELEMETRY_GPU_SOURCE=<name>`` narrows the list to a
        single candidate (no fallback). ``auto`` or unset walks the
        platform-appropriate chain.
        """
        override = os.environ.get("STUDIO_TELEMETRY_GPU_SOURCE", "").strip().lower()
        if override and override != "auto":
            return [override]

        if sys.platform == "darwin":
            machine = _py_platform.machine().lower()
            if machine in ("arm64", "aarch64"):
                return ["ioreport", "iokit", "mlx_mem", "powermetrics"]
            # Intel Mac: no reliable public API. mlx-lm isn't
            # installed on Intel Macs anyway (pip marker guards it).
            return []

        if sys.platform.startswith("linux"):
            # WSL2: /sys/class/drm GPU paths aren't populated — the
            # driver lives on the Windows side. pynvml works via the
            # NVIDIA Windows driver. Skip sysfs.
            if _is_wsl():
                return ["pynvml", "nvidia_smi_cli"]
            # Native Linux: NVIDIA → AMD → Intel → nvidia-smi fallback.
            return ["pynvml", "amdgpu_sysfs", "intel_sysfs", "nvidia_smi_cli"]

        if sys.platform in ("win32", "cygwin"):
            # pynvml first (per-card detail, locale-safe); PDH second
            # for any-GPU coverage (AMD, Intel Arc); nvidia-smi last.
            return ["pynvml", "pdh", "nvidia_smi_cli"]

        return []

    def _pick_source(self) -> str:
        """Walk ``_candidate_sources`` and return the first whose
        probe returns True. Falls back to ``unavailable``.
        """
        probes = self._probe_map()
        candidates = self._candidate_sources()
        override = os.environ.get("STUDIO_TELEMETRY_GPU_SOURCE", "").strip().lower()

        # Env-forced path: single-shot, no fallback.
        if override and override != "auto":
            probe = probes.get(override)
            if probe is None:
                logger.warning(
                    "Unknown STUDIO_TELEMETRY_GPU_SOURCE=%r; walking auto chain",
                    override,
                )
                # Re-enter the auto chain.
                os.environ.pop("STUDIO_TELEMETRY_GPU_SOURCE", None)
                return self._pick_source()
            if probe():
                return override
            logger.warning(
                "STUDIO_TELEMETRY_GPU_SOURCE=%s probe failed; walking auto chain",
                override,
            )
            os.environ.pop("STUDIO_TELEMETRY_GPU_SOURCE", None)
            return self._pick_source()

        for name in candidates:
            probe = probes.get(name)
            if probe is None:
                continue
            try:
                if probe():
                    return name
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("probe %s raised: %s", name, exc)
        return "unavailable"

    # ── Sampling ─────────────────────────────────────────────────────

    def _read_mlx_memory_only(self) -> Tuple[Optional[float], Optional[float]]:
        """Return ``(mem_used_gb, mem_total_gb)`` via MLX.

        Shared between ``_read_mlx_mem`` (full reading) and the ioreport
        path (which reuses memory numbers because IOReport's GPU Stats
        channel doesn't carry them). Never called on non-macOS paths.
        """
        try:
            import mlx.core as mx
            try:
                active_bytes = int(mx.get_active_memory())
            except AttributeError:
                active_bytes = int(mx.metal.get_active_memory())
            try:
                info = mx.device_info()
            except Exception:
                info = mx.metal.device_info()  # deprecated fallback
            total_bytes = int(info.get("memory_size", 0)) or 0
            return (
                round(active_bytes / 1e9, 3),
                round(total_bytes / 1e9, 3) if total_bytes else None,
            )
        except Exception as exc:
            logger.debug("mlx memory read failed: %s", exc)
            return (None, None)

    def _read_mlx_mem(self) -> dict:
        """Read GPU memory via MLX. util is a binary 100/0 proxy based
        on whether any session is actively generating.
        """
        mem_used, mem_total = self._read_mlx_memory_only()
        util = 100.0 if self._any_session_active() else 0.0
        return {
            "util_pct": util,
            "mem_used_gb": mem_used,
            "mem_total_gb": mem_total,
            "source": "mlx_mem",
            "freq_mhz": None,
        }

    def _read_ioreport(self) -> dict:
        """IOReport util % + MLX memory numbers."""
        util: Optional[float] = None
        if self._ioreport is not None:
            try:
                util = self._ioreport.sample_active_pct()
            except Exception as exc:
                logger.warning("ioreport sample failed: %s", exc)
                util = None
        mem_used, mem_total = self._read_mlx_memory_only()
        return {
            "util_pct": util,
            "mem_used_gb": mem_used,
            "mem_total_gb": mem_total,
            "source": "ioreport",
            "freq_mhz": None,
        }

    def _read_pynvml(self) -> dict:
        """NVML read. Device handle was cached at probe time."""
        pynvml = self._pynvml_module
        handle = self._pynvml_handle
        if pynvml is None or handle is None:
            return self._degraded("pynvml")
        try:
            rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
            util = float(rates.gpu)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_gb = round(int(mem.used) / 1e9, 3)
            total_gb = round(int(mem.total) / 1e9, 3)
        except Exception as exc:
            logger.warning("pynvml read failed: %s", exc)
            return self._degraded("pynvml")

        freq_mhz: Optional[float] = None
        try:
            freq_mhz = float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
        except Exception:
            freq_mhz = None

        return {
            "util_pct": util,
            "mem_used_gb": used_gb,
            "mem_total_gb": total_gb,
            "source": "pynvml",
            "freq_mhz": freq_mhz,
        }

    @staticmethod
    def _read_sysfs_int(path: str) -> Optional[int]:
        try:
            with open(path, "r") as fh:
                return int(fh.read().strip())
        except (OSError, ValueError):
            return None

    def _read_amdgpu_sysfs(self) -> dict:
        dev = self._amdgpu_card_path
        if not dev:
            return self._degraded("amdgpu_sysfs")
        util_raw = self._read_sysfs_int(os.path.join(dev, "gpu_busy_percent"))
        used_raw = self._read_sysfs_int(os.path.join(dev, "mem_info_vram_used"))
        total_raw = self._read_sysfs_int(os.path.join(dev, "mem_info_vram_total"))
        freq_raw = self._read_sysfs_int(os.path.join(dev, "pp_dpm_sclk")) \
            if os.path.isfile(os.path.join(dev, "pp_dpm_sclk")) else None

        return {
            "util_pct": float(util_raw) if util_raw is not None else None,
            "mem_used_gb": round(used_raw / 1e9, 3) if used_raw is not None else None,
            "mem_total_gb": round(total_raw / 1e9, 3) if total_raw is not None else None,
            "source": "amdgpu_sysfs",
            # pp_dpm_sclk is multi-line "0: 300Mhz\n1: 800Mhz *\n..." —
            # parsing it is more fuss than it's worth for a bonus
            # field. Leave None.
            "freq_mhz": None,
        }

    def _read_intel_sysfs(self) -> dict:
        """Intel i915 engine busy counter. Take the diff of the
        monotonic ns counter against wall clock, same strategy as
        ``intel_gpu_top``."""
        busy_f = self._intel_engine_file
        card_path = self._intel_card_path
        if not busy_f or not card_path:
            return self._degraded("intel_sysfs")
        now_ns = time.monotonic_ns()
        busy_ns = self._read_sysfs_int(busy_f)
        if busy_ns is None:
            return self._degraded("intel_sysfs")

        util: Optional[float] = None
        if self._intel_prev_busy_ns is not None and self._intel_prev_wall_ns is not None:
            d_busy = busy_ns - self._intel_prev_busy_ns
            d_wall = now_ns - self._intel_prev_wall_ns
            if d_wall > 0 and d_busy >= 0:
                util = max(0.0, min(100.0, 100.0 * d_busy / d_wall))
        self._intel_prev_busy_ns = busy_ns
        self._intel_prev_wall_ns = now_ns

        # dGPUs (Arc) expose mem_info_vram_*; iGPUs don't.
        used_raw = self._read_sysfs_int(os.path.join(card_path, "mem_info_vram_used"))
        total_raw = self._read_sysfs_int(os.path.join(card_path, "mem_info_vram_total"))

        return {
            "util_pct": util,
            "mem_used_gb": round(used_raw / 1e9, 3) if used_raw is not None else None,
            "mem_total_gb": round(total_raw / 1e9, 3) if total_raw is not None else None,
            "source": "intel_sysfs",
            "freq_mhz": None,
        }

    def _read_pdh(self) -> dict:
        """Windows PDH read.

        Sums the ``Utilization Percentage`` across all 3D engine
        instances and clamps to [0, 100]. Task Manager does
        effectively the same thing (per-engine max, then summed across
        cards). For a single-GPU box the summation is equivalent.
        """
        if self._pdh_state is None:
            return self._degraded("pdh")
        win32pdh = self._pdh_state["module"]
        query = self._pdh_state["query"]
        counter = self._pdh_state["counter"]
        try:
            win32pdh.CollectQueryData(query)
            # GetFormattedCounterArray returns list of (instance, value).
            _total, items = win32pdh.GetFormattedCounterArray(
                counter, win32pdh.PDH_FMT_DOUBLE,
            )
            util = 0.0
            for _instance, value in items:
                try:
                    util += float(value)
                except (TypeError, ValueError):
                    continue
            util = max(0.0, min(100.0, util))
        except Exception as exc:
            logger.warning("pdh read failed: %s", exc)
            return self._degraded("pdh")

        return {
            "util_pct": util,
            # PDH has per-process memory counters but not a total.
            # Leaving these None keeps the wire shape stable — the
            # sparkline still renders the util line.
            "mem_used_gb": None,
            "mem_total_gb": None,
            "source": "pdh",
            "freq_mhz": None,
        }

    def _read_nvidia_smi_cli(self) -> dict:
        """Fallback NVIDIA read via subprocess."""
        path = self._nvsmi_path
        if not path:
            return self._degraded("nvidia_smi_cli")
        idx = _cuda_visible_index()
        try:
            rc = subprocess.run(
                [
                    path,
                    f"--id={idx}",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output = True,
                timeout = 2.0,
                text = True,
            )
            if rc.returncode != 0 or not rc.stdout.strip():
                return self._degraded("nvidia_smi_cli")
            # First CSV row; ``--id`` pins us to one device.
            first = rc.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in first.split(",")]
            util = float(parts[0])
            used_mb = float(parts[1])
            total_mb = float(parts[2])
        except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
            logger.warning("nvidia-smi read failed: %s", exc)
            return self._degraded("nvidia_smi_cli")

        return {
            "util_pct": util,
            "mem_used_gb": round(used_mb / 1024.0, 3),
            "mem_total_gb": round(total_mb / 1024.0, 3),
            "source": "nvidia_smi_cli",
            "freq_mhz": None,
        }

    def _degraded(self, source: str) -> dict:
        """Return an all-``None`` event for a given source, preserving
        the wire shape when a reader hits an error."""
        return {
            "util_pct": None,
            "mem_used_gb": None,
            "mem_total_gb": None,
            "source": source,
            "freq_mhz": None,
        }

    def _dispatch_read(self) -> dict:
        """Single source of truth for "which reader runs this tick".

        Keeping this small and switch-like makes the multi-GPU
        extension point obvious: return ``list[dict]`` here and let
        ``_run`` iterate.
        """
        src = self._source
        if src == "ioreport":
            return self._read_ioreport()
        if src == "mlx_mem":
            return self._read_mlx_mem()
        if src == "pynvml":
            return self._read_pynvml()
        if src == "amdgpu_sysfs":
            return self._read_amdgpu_sysfs()
        if src == "intel_sysfs":
            return self._read_intel_sysfs()
        if src == "pdh":
            return self._read_pdh()
        if src == "nvidia_smi_cli":
            return self._read_nvidia_smi_cli()
        # iokit / powermetrics: scaffold-only. Reuse mlx_mem shape so
        # the frontend still gets a usable event.
        if src == "iokit" or src == "powermetrics":
            return self._read_mlx_mem()
        return self._degraded(src)

    def _init_backend(self) -> None:
        """Do the per-backend "open handle" step on the sampler thread.

        Called after ``_pick_source`` returns. Only the backend that
        actually won the probe gets its handle initialised here —
        cross-platform bindings (CoreFoundation, nvmlInit, win32pdh)
        cost nothing on platforms that never run them.
        """
        src = self._source
        if src == "ioreport":
            try:
                self._ioreport = _IOReportBinding()
            except Exception as exc:
                logger.warning(
                    "ioreport init failed post-probe (%s); falling back to mlx_mem",
                    exc,
                )
                self._source = "mlx_mem" if self._probe_mlx_mem() else "unavailable"
        # pynvml / amdgpu_sysfs / intel_sysfs / pdh / nvidia_smi_cli
        # all cache their handles in the probe itself. Nothing to do
        # here.

    def _cleanup_backend(self) -> None:
        if self._ioreport is not None:
            try:
                self._ioreport.close()
            except Exception as exc:
                logger.debug("ioreport close raised: %s", exc)
            self._ioreport = None

        if self._pdh_state is not None:
            try:
                self._pdh_state["module"].CloseQuery(self._pdh_state["query"])
            except Exception as exc:
                logger.debug("pdh close raised: %s", exc)
            self._pdh_state = None

        # NVML: we deliberately do NOT nvmlShutdown here. Reasoning:
        # the sampler may stop/start during a single process (uvicorn
        # reload in dev), and pynvml.nvmlInit followed by
        # nvmlShutdown followed by nvmlInit in the same process is
        # best-effort at best across driver versions. The module-level
        # ``_pynvml_initialised`` flag makes the second start a no-op.
        # The OS reclaims the NVML handle on process exit.

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
                    "freq_mhz": None,
                },
            )
            return

        self._init_backend()
        if self._source == "unavailable":
            # Init downgraded us (e.g. ioreport post-probe failure
            # with no MLX). Emit the one-shot unavailable event.
            self._broadcaster.emit(
                "gpu",
                {
                    "ts": time.monotonic(),
                    "util_pct": None,
                    "mem_used_gb": None,
                    "mem_total_gb": None,
                    "source": "unavailable",
                    "freq_mhz": None,
                },
            )
            return

        consecutive_failures = 0
        try:
            while not self._stop_event.is_set():
                try:
                    payload = self._dispatch_read()
                    # Payload shape sanity: the reader may have returned
                    # an all-None degraded dict. That's still a valid
                    # tick — don't count it as a failure.
                    consecutive_failures = 0
                    self._last_event = payload
                except Exception as exc:
                    consecutive_failures += 1
                    logger.warning(
                        "GPU sampler read raised (%d/%d): %s",
                        consecutive_failures,
                        _MAX_CONSECUTIVE_READ_FAILURES,
                        exc,
                    )
                    payload = self._last_event or self._degraded(self._source)
                    if consecutive_failures >= _MAX_CONSECUTIVE_READ_FAILURES:
                        logger.warning(
                            "GPU sampler exiting: %d consecutive read failures",
                            consecutive_failures,
                        )
                        break

                payload["ts"] = time.monotonic()
                try:
                    # NOTE: multi-GPU extension point. To report all
                    # visible devices, call a ``_dispatch_read_all()``
                    # that returns ``list[dict]`` and emit one event
                    # per entry here.
                    self._broadcaster.emit("gpu", payload)
                except Exception as exc:
                    logger.warning("GPU sampler emit raised: %s", exc)

                self._stop_event.wait(_SAMPLE_INTERVAL_S)
        finally:
            self._cleanup_backend()


# Module-level singleton.
gpu_sampler: GpuSampler = GpuSampler()
