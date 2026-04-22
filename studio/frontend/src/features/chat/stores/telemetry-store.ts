// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Phase 3 — Zustand store fed by ``/ws/telemetry``. Three event
// classes flow through:
//
//   - ``gpu``      device-wide GPU sample (util %, mem). 500ms cadence.
//                   Stored in a 120-sample ring (60s window).
//   - ``session``  per-session lifecycle (prompt_eval / generating /
//                   done / cancelled). Keyed by session_id; used to
//                   drive the token counter's "current turn" view.
//   - ``tokens``   pre-filter token count + tps, per session. Fires at
//                   ~4 Hz while generating.
//
// The store is deliberately decoupled from the WebSocket lifecycle —
// the hook pushes events in, the components read selectors out. That
// makes it easy to unit-test each side in isolation and lets a
// headless tab see the same state as the foreground tab.

import { create } from "zustand";

export const GPU_RING_SIZE = 120; // 60s at 2 Hz

export interface GpuSample {
  ts: number; // monotonic seconds from the backend
  util: number | null;
  memUsedGb: number | null;
  memTotalGb: number | null;
}

export interface TokensEvent {
  iteration: number;
  preFilterTokens: number;
  postFilterTokens: number;
  tps: number | null;
  promptTps: number | null;
  /** Monotonic-enough timestamp used for staleness detection. */
  receivedAt: number;
}

export interface SessionState {
  state: string; // "prompt_eval" | "generating" | "done" | "cancelled"
  iteration: number;
  updatedAt: number;
  tokens: TokensEvent | null;
  /** Running pre-filter token total across ALL iterations of this
   * agentic turn, not just the current iteration. */
  cumulativePreFilter: number;
  /** The last iteration index we tallied into ``cumulativePreFilter``.
   * Used to detect iteration boundaries and fold the previous iter's
   * tokens into the running total. */
  lastTalliedIter: number;
  /** Pre-filter tokens from iteration 0 — surfaced in the chip's
   * iteration-boundary flash so the user sees the breakdown. */
  preFilterByIter: Record<number, number>;
}

interface TelemetryStore {
  connected: boolean;
  gpu: { samples: GpuSample[]; source: string | null };
  sessions: Record<string, SessionState>;

  setConnected: (c: boolean) => void;
  pushGpu: (sample: GpuSample, source: string) => void;
  setSessionState: (
    sessionId: string,
    state: string,
    iteration: number,
  ) => void;
  setTokens: (sessionId: string, tokens: TokensEvent) => void;
  resetSession: (sessionId: string) => void;
}

export const useTelemetryStore = create<TelemetryStore>((set) => ({
  connected: false,
  gpu: { samples: [], source: null },
  sessions: {},

  setConnected: (connected) => set({ connected }),

  pushGpu: (sample, source) =>
    set((state) => {
      const next = state.gpu.samples.concat(sample);
      if (next.length > GPU_RING_SIZE) {
        next.splice(0, next.length - GPU_RING_SIZE);
      }
      return { gpu: { samples: next, source } };
    }),

  setSessionState: (sessionId, state, iteration) =>
    set((prev) => {
      const existing = prev.sessions[sessionId];
      const now = Date.now();
      if (state === "prompt_eval" && (!existing || iteration === 0)) {
        // Brand new agentic turn — reset cumulative counter.
        return {
          sessions: {
            ...prev.sessions,
            [sessionId]: {
              state,
              iteration,
              updatedAt: now,
              tokens: null,
              cumulativePreFilter: 0,
              lastTalliedIter: -1,
              preFilterByIter: {},
            },
          },
        };
      }
      return {
        sessions: {
          ...prev.sessions,
          [sessionId]: {
            state,
            iteration,
            updatedAt: now,
            tokens: existing?.tokens ?? null,
            cumulativePreFilter: existing?.cumulativePreFilter ?? 0,
            lastTalliedIter: existing?.lastTalliedIter ?? -1,
            preFilterByIter: existing?.preFilterByIter ?? {},
          },
        },
      };
    }),

  setTokens: (sessionId, tokens) =>
    set((prev) => {
      const existing = prev.sessions[sessionId];
      const now = Date.now();
      // Detect iteration boundaries: when iter advances, fold the
      // previous iter's max pre_filter_tokens into the cumulative
      // running total. This keeps the displayed "total tokens" monotonic
      // across iterations instead of resetting each turn.
      let cumulative = existing?.cumulativePreFilter ?? 0;
      const byIter: Record<number, number> = { ...(existing?.preFilterByIter ?? {}) };
      const lastTallied = existing?.lastTalliedIter ?? -1;
      byIter[tokens.iteration] = Math.max(
        byIter[tokens.iteration] ?? 0,
        tokens.preFilterTokens,
      );
      if (tokens.iteration > lastTallied + 1) {
        // Skipped an iteration (shouldn't normally happen) — fold every
        // buffered count between.
        for (let i = lastTallied + 1; i < tokens.iteration; i += 1) {
          cumulative += byIter[i] ?? 0;
        }
      }
      return {
        sessions: {
          ...prev.sessions,
          [sessionId]: {
            state: existing?.state ?? "generating",
            iteration: tokens.iteration,
            updatedAt: now,
            tokens,
            cumulativePreFilter: cumulative,
            lastTalliedIter:
              tokens.iteration > lastTallied ? tokens.iteration - 1 : lastTallied,
            preFilterByIter: byIter,
          },
        },
      };
    }),

  resetSession: (sessionId) =>
    set((prev) => {
      if (!(sessionId in prev.sessions)) return prev;
      const next = { ...prev.sessions };
      delete next[sessionId];
      return { sessions: next };
    }),
}));

// Dev-only window hook so the browser console (and the Chrome preview
// harness that verified Phase 3) can poll store state without React
// DevTools. Studio is a local dev tool — leaving this surface is a
// feature, not a leak — but we gate on ``import.meta.env.DEV`` so a
// future production build trims it out.
if (typeof window !== "undefined" && import.meta.env.DEV !== false) {
  // @ts-expect-error dev-only window hook
  window.__telemetryStore = useTelemetryStore;
}
