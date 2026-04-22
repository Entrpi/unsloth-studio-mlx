// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Phase 3 — ``/ws/telemetry`` WebSocket client hook.
//
// Responsibilities:
//   - Auto-connect on mount when a user is authenticated and the flag
//     ``VITE_ENABLE_TELEMETRY_WS`` isn't set to "0".
//   - Reconnect with exponential backoff (1s → 30s, ±25% jitter).
//     Reset backoff on successful connect.
//   - Subscribe to ``gpu`` (global) + ``session`` + ``tokens`` with
//     the current active session_id so fan-out on the server side
//     only sends this tab events that matter to it.
//   - Push events into the Zustand store.
//   - Clean up on unmount.

import { useEffect, useRef } from "react";
import { getAuthToken, hasAuthToken } from "@/features/auth/session";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { useTelemetryStore, type GpuSample, type TokensEvent } from "../stores/telemetry-store";

const WS_PATH = "/ws/telemetry";

function telemetryWsEnabled(): boolean {
  // Vite replaces ``import.meta.env.*`` at build time. Default to
  // enabled — setting ``VITE_ENABLE_TELEMETRY_WS=0`` at build time
  // disables the client entirely (the server flag is independent).
  const raw = (import.meta.env.VITE_ENABLE_TELEMETRY_WS as string | undefined) ?? "";
  return raw !== "0" && raw.toLowerCase() !== "false";
}

function buildWsUrl(token: string): string {
  const loc = window.location;
  const proto = loc.protocol === "https:" ? "wss:" : "ws:";
  const url = new URL(`${proto}//${loc.host}${WS_PATH}`);
  url.searchParams.set("token", token);
  return url.toString();
}

function backoffDelayMs(attempt: number): number {
  const base = Math.min(30_000, 1_000 * 2 ** attempt);
  const jitter = base * 0.25 * (Math.random() * 2 - 1);
  return Math.max(500, Math.floor(base + jitter));
}

type TelemetryFrame =
  | {
      type: "gpu";
      ts: number;
      util_pct: number | null;
      mem_used_gb: number | null;
      mem_total_gb: number | null;
      source: string;
    }
  | {
      type: "session";
      session_id?: string | null;
      state: string;
      iteration: number;
    }
  | {
      type: "tokens";
      session_id?: string | null;
      iteration: number;
      pre_filter_tokens: number;
      post_filter_tokens: number;
      tps: number | null;
      prompt_tps: number | null;
    };

export function useTelemetrySocket(): void {
  const activeThreadId = useChatRuntimeStore((s) => s.activeThreadId);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const attemptRef = useRef<number>(0);
  const closedByUnmountRef = useRef<boolean>(false);
  const currentSessionRef = useRef<string | null>(activeThreadId ?? null);

  useEffect(() => {
    currentSessionRef.current = activeThreadId ?? null;
    // If already open, send an updated subscription frame.
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          subscribe: ["gpu", "session", "tokens"],
          session_id: activeThreadId ?? null,
        }),
      );
    }
  }, [activeThreadId]);

  useEffect(() => {
    if (!telemetryWsEnabled()) return;
    if (!hasAuthToken()) return;

    const store = useTelemetryStore.getState();
    closedByUnmountRef.current = false;

    const connect = () => {
      const token = getAuthToken();
      if (!token) return;
      let ws: WebSocket;
      try {
        ws = new WebSocket(buildWsUrl(token));
      } catch (err) {
        // Some browsers throw on bad URL — schedule a retry.
        console.warn("[telemetry] WS construct failed", err);
        scheduleReconnect();
        return;
      }
      wsRef.current = ws;

      ws.onopen = () => {
        attemptRef.current = 0;
        store.setConnected(true);
        ws.send(
          JSON.stringify({
            subscribe: ["gpu", "session", "tokens"],
            session_id: currentSessionRef.current,
          }),
        );
      };

      ws.onmessage = (ev) => {
        let frame: TelemetryFrame;
        try {
          frame = JSON.parse(ev.data as string) as TelemetryFrame;
        } catch {
          return;
        }
        const s = useTelemetryStore.getState();
        if (frame.type === "gpu") {
          const sample: GpuSample = {
            ts: frame.ts,
            util: frame.util_pct,
            memUsedGb: frame.mem_used_gb,
            memTotalGb: frame.mem_total_gb,
          };
          s.pushGpu(sample, frame.source);
        } else if (frame.type === "session") {
          const sid = frame.session_id || currentSessionRef.current || "__global";
          s.setSessionState(sid, frame.state, frame.iteration);
        } else if (frame.type === "tokens") {
          const sid = frame.session_id || currentSessionRef.current || "__global";
          const tokens: TokensEvent = {
            iteration: frame.iteration,
            preFilterTokens: frame.pre_filter_tokens,
            postFilterTokens: frame.post_filter_tokens,
            tps: frame.tps,
            promptTps: frame.prompt_tps,
            receivedAt: Date.now(),
          };
          s.setTokens(sid, tokens);
        }
      };

      ws.onclose = () => {
        useTelemetryStore.getState().setConnected(false);
        wsRef.current = null;
        if (!closedByUnmountRef.current) scheduleReconnect();
      };

      ws.onerror = () => {
        // Let onclose handle reconnect — avoid double-scheduling.
      };
    };

    const scheduleReconnect = () => {
      if (closedByUnmountRef.current) return;
      const delay = backoffDelayMs(attemptRef.current);
      attemptRef.current = Math.min(attemptRef.current + 1, 6);
      reconnectTimerRef.current = setTimeout(connect, delay);
    };

    connect();

    return () => {
      closedByUnmountRef.current = true;
      if (reconnectTimerRef.current !== null) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      const ws = wsRef.current;
      if (ws) {
        try {
          ws.close();
        } catch {
          // ignore
        }
        wsRef.current = null;
      }
      useTelemetryStore.getState().setConnected(false);
    };
  }, []);
}

// ── Selectors ──────────────────────────────────────────────────────

export function useGpuSamples() {
  return useTelemetryStore((s) => s.gpu);
}

export function useSessionTelemetry(sessionId: string | null | undefined) {
  return useTelemetryStore((s) =>
    sessionId ? (s.sessions[sessionId] ?? null) : null,
  );
}
