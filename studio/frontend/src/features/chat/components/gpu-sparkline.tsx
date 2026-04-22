// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Phase 3 — 60-second rolling GPU utilisation sparkline.
//
// Hand-rolled SVG polyline (no recharts) — the training charts use
// recharts but they're full-page cards with axes / tooltips / margins;
// a 60x18 px inline sparkline is an order of magnitude smaller and
// rendering a recharts <LineChart> just to draw 120 points costs more
// DOM than the whole surrounding composer.

import type { FC } from "react";
import { useMemo } from "react";
import { useGpuSamples } from "../hooks/use-telemetry-socket";

const WIDTH = 72;
const HEIGHT = 18;
const PAD = 1;
// After this many ms of no samples, fade the chip out — the backend
// sampler is best-effort and losing the stream for >10s is a reliable
// "something is wrong" signal worth communicating visually.
const STALE_MS = 10_000;

function buildPath(values: number[]): string {
  if (values.length < 2) return "";
  const usable = WIDTH - PAD * 2;
  const h = HEIGHT - PAD * 2;
  const maxIdx = values.length - 1;
  const parts: string[] = [];
  for (let i = 0; i < values.length; i += 1) {
    const x = PAD + (usable * i) / maxIdx;
    // util is 0-100; invert so high util is near the top.
    const norm = Math.max(0, Math.min(100, values[i])) / 100;
    const y = PAD + h * (1 - norm);
    parts.push(`${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`);
  }
  return parts.join(" ");
}

export const GpuSparkline: FC = () => {
  const { samples, source } = useGpuSamples();

  const { values, latest, stale } = useMemo(() => {
    const v: number[] = [];
    for (const s of samples) {
      if (s.util !== null && Number.isFinite(s.util)) v.push(s.util);
    }
    const lastSample = samples[samples.length - 1] ?? null;
    const latestUtil = lastSample?.util ?? null;
    // ``ts`` is server monotonic seconds — we use wall-clock elapsed
    // since the LAST frame arrived as a proxy (we don't sync clocks).
    // The telemetry hook updates the store on every frame, so "no
    // sample in STALE_MS ms" is truly quiet.
    const isStale = samples.length === 0;
    return { values: v, latest: latestUtil, stale: isStale };
  }, [samples]);

  if (source === "unavailable") return null;
  // Hide until we have at least 2 data points; a single-point sparkline
  // looks broken.
  if (values.length < 2 && !latest) return null;

  const path = buildPath(values);
  const latestText =
    latest === null || latest === undefined
      ? "–"
      : `${Math.round(latest)}%`;

  return (
    <div
      className="flex items-center gap-1.5 rounded-full border border-muted-foreground/15 bg-muted/20 px-2 py-0.5 text-[10px] text-muted-foreground"
      title={`GPU ${latestText}${source ? ` (${source})` : ""}`}
      data-testid="gpu-sparkline"
      style={{ opacity: stale ? 0.35 : 1, transition: "opacity 600ms" }}
    >
      <span className="tabular-nums">GPU {latestText}</span>
      <svg
        width={WIDTH}
        height={HEIGHT}
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        aria-hidden="true"
      >
        <path
          d={path}
          fill="none"
          stroke="currentColor"
          strokeWidth={1}
          strokeLinecap="round"
          strokeLinejoin="round"
          opacity={0.6}
        />
      </svg>
    </div>
  );
};
