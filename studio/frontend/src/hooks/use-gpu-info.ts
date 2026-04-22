// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";

export interface GpuInfo {
  available: boolean;
  name: string;
  memoryTotalGb: number;
  systemRamAvailableGb: number;
  /**
   * True iff the backend reported this host can run MLX (Apple Silicon
   * macOS). Used by the model picker to decide whether to surface MLX
   * quant suggestions — on Linux / Windows / DGX Spark the MLX sections
   * are hidden entirely since the weights couldn't be loaded anyway.
   */
  isAppleSilicon: boolean;
}

const DEFAULT_GPU: GpuInfo = {
  available: false,
  name: "Unknown",
  memoryTotalGb: 0,
  systemRamAvailableGb: 0,
  isAppleSilicon: false,
};

// Module-level cache so multiple components share one fetch.
let cachedGpu: GpuInfo | null = null;
let fetchPromise: Promise<GpuInfo> | null = null;

async function fetchGpuOnce(): Promise<GpuInfo> {
  if (cachedGpu) return cachedGpu;
  if (fetchPromise) return fetchPromise;

  fetchPromise = (async () => {
    try {
      const res = await fetch("/api/system");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const gpuData = data?.gpu;
      if (!gpuData?.available || !gpuData.devices?.length) {
        // Even when no GPU is present, preserve the Apple Silicon signal
        // so the picker can gate MLX rows correctly in edge cases (e.g.
        // Metal disabled, headless Mac).
        return { ...DEFAULT_GPU, isAppleSilicon: Boolean(data?.is_apple_silicon) };
      }
      const devices = gpuData.devices as Array<{ name?: string; memory_total_gb?: number }>;
      const totalGb = devices.reduce((sum, d) => sum + (d.memory_total_gb ?? 0), 0);
      const info: GpuInfo = {
        available: true,
        name: devices[0]?.name ?? "Unknown",
        memoryTotalGb: totalGb,
        systemRamAvailableGb: data?.memory?.available_gb ?? 0,
        isAppleSilicon: Boolean(data?.is_apple_silicon),
      };
      cachedGpu = info;
      return info;
    } catch {
      // Reset promise so subsequent calls retry (e.g. backend wasn't ready)
      fetchPromise = null;
      return DEFAULT_GPU;
    }
  })();

  return fetchPromise;
}

/**
 * Fetch GPU info from the backend /api/system endpoint.
 *
 * The result is cached at module level -- only one network request is made
 * regardless of how many components call this hook.
 */
export function useGpuInfo(): GpuInfo {
  const [gpu, setGpu] = useState<GpuInfo>(cachedGpu ?? DEFAULT_GPU);

  useEffect(() => {
    if (cachedGpu) return;

    let cancelled = false;
    fetchGpuOnce().then((info) => {
      if (!cancelled) setGpu(info);
    });
    return () => { cancelled = true; };
  }, []);

  return gpu;
}
