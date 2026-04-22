# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Phase 3 — Apple-Silicon GPU sampler.

Fallback chain (tried in order, first-works wins):

    1. **ioreport** — Apple's private ``libIOReport.dylib`` via ctypes.
                      Real GPU "Active Residency %" read from the
                      ``GPU Stats`` / ``GPUPH`` state channel — the same
                      source ``mactop`` / ``macmon`` use. No sudo, ~0.1%
                      CPU overhead, first-class on Apple Silicon. Returns
                      False on Intel / unusual SKUs where the dylib is
                      absent or the channel is NULL.
    2. **iokit**    — pyobjc ``IOReport``/``IOAccelerator`` for
                      device-util %. No sudo required. Skipped when
                      pyobjc is not installed (the default on a bare
                      Studio environment). Scaffold-only today.
    3. **mlx_mem**  — ``mlx.core.metal.get_active_memory()`` +
                      ``device_info()``. Always works on a box with
                      MLX installed, which is the Studio-supported
                      hardware baseline. Util % is synthesised from
                      the "is any session actively generating" flag:
                      100 while generating, 0 otherwise. Crude but
                      actionable.
    4. **powermetrics** — subprocess with sudo probe. Skipped unless
                      ``STUDIO_TELEMETRY_ALLOW_POWERMETRICS=1`` and
                      passwordless sudo is configured, because
                      prompting for a password would hang the app.
    5. **unavailable** — emits one ``{"source": "unavailable"}``
                      event and stops sampling.

Set ``STUDIO_TELEMETRY_GPU_SOURCE=ioreport|iokit|mlx_mem|powermetrics|auto``
to override the chain. ``auto`` (the default) walks the list above.

Samples every 500ms (2 Hz); the emit rate is conservative so the
WebSocket doesn't get spammed and the sparkline stays smooth at a
human-perceivable cadence.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
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


# ── IOReport ctypes binding ──────────────────────────────────────────
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
        self._ioreport: Optional[_IOReportBinding] = None

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

    def _probe_ioreport(self) -> bool:
        """Attempt to dlopen libIOReport and open/close a full
        subscription. The binding constructor raises cleanly on Intel
        (dylib missing) or non-Apple-Silicon SKUs (``GPU Stats`` group
        NULL), which is exactly the probe contract we need.
        """
        try:
            binding = _IOReportBinding()
        except (OSError, RuntimeError, Exception):  # pragma: no cover - defensive
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
        override = os.environ.get("STUDIO_TELEMETRY_GPU_SOURCE", "auto").strip().lower()
        if override and override != "auto":
            probe_map = {
                "ioreport": self._probe_ioreport,
                "iokit": self._probe_iokit,
                "mlx_mem": self._probe_mlx_mem,
                "powermetrics": self._probe_powermetrics,
            }
            probe = probe_map.get(override)
            if probe is None:
                logger.warning(
                    "Unknown STUDIO_TELEMETRY_GPU_SOURCE=%r; falling back to auto",
                    override,
                )
            elif probe():
                return override
            else:
                logger.warning(
                    "STUDIO_TELEMETRY_GPU_SOURCE=%s probe failed; falling back to auto",
                    override,
                )

        if self._probe_ioreport():
            return "ioreport"
        if self._probe_iokit():
            return "iokit"
        if self._probe_mlx_mem():
            return "mlx_mem"
        if self._probe_powermetrics():
            return "powermetrics"
        return "unavailable"

    # ── Sampling ─────────────────────────────────────────────────────

    def _read_mlx_memory_only(self) -> Tuple[Optional[float], Optional[float]]:
        """Return ``(mem_used_gb, mem_total_gb)`` via MLX.

        Shared between ``_read_mlx_mem`` (full reading) and the ioreport
        path (which reuses memory numbers because IOReport's GPU Stats
        channel doesn't carry them).
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

        # Initialize IOReport lazily on the sampler thread so a
        # non-Apple-Silicon instance (where the probe returned False)
        # never dlopens anything.
        if self._source == "ioreport":
            try:
                self._ioreport = _IOReportBinding()
            except Exception as exc:
                logger.warning(
                    "ioreport init failed post-probe (%s); falling back to mlx_mem",
                    exc,
                )
                self._source = "mlx_mem" if self._probe_mlx_mem() else "unavailable"

        try:
            # Steady-state loop.
            while not self._stop_event.is_set():
                if self._source == "ioreport":
                    payload = self._read_ioreport()
                elif self._source == "mlx_mem":
                    payload = self._read_mlx_mem()
                else:
                    # Placeholder for iokit / powermetrics impls — fall
                    # back to mlx_mem shape. Keeps the wire format
                    # stable across sources.
                    payload = self._read_mlx_mem()

                payload["ts"] = time.monotonic()
                try:
                    self._broadcaster.emit("gpu", payload)
                except Exception as exc:
                    logger.warning("GPU sampler emit raised: %s", exc)

                self._stop_event.wait(_SAMPLE_INTERVAL_S)
        finally:
            if self._ioreport is not None:
                try:
                    self._ioreport.close()
                except Exception as exc:
                    logger.debug("ioreport close raised: %s", exc)
                self._ioreport = None


# Module-level singleton.
gpu_sampler: GpuSampler = GpuSampler()
