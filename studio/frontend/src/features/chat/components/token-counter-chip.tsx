// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Phase 3 — live pre-filter token counter chip.
//
// Lives ABOVE the composer (not inside it — the composer's tool /
// progress chips occupy the inner slot already). Visible only while
// the thread is running AND at least one token has been counted so
// we don't flash an empty "0 tok" when the stream is still prefilling.
//
// The chip's primary text is "{cumulative} tok · {tps} t/s"; on an
// iteration boundary we briefly flash a "iter {N-1}: {count} tok"
// secondary line (1500ms fade) so the user sees which iteration
// produced what.

import type { FC } from "react";
import { useEffect, useState } from "react";
import { useAuiState } from "@assistant-ui/react";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { useSessionTelemetry } from "../hooks/use-telemetry-socket";

function formatTps(tps: number | null | undefined): string {
  if (tps === null || tps === undefined || !Number.isFinite(tps)) return "–";
  return tps >= 10 ? tps.toFixed(1) : tps.toFixed(2);
}

export const TokenCounterChip: FC = () => {
  const activeThreadId = useChatRuntimeStore((s) => s.activeThreadId);
  const telemetry = useSessionTelemetry(activeThreadId);
  const isThreadRunning = useAuiState(({ thread }) => thread.isRunning);
  const [flashMessage, setFlashMessage] = useState<string | null>(null);
  const [prevIter, setPrevIter] = useState<number>(-1);

  useEffect(() => {
    const iter = telemetry?.iteration ?? -1;
    if (iter > prevIter && prevIter >= 0) {
      // iteration advanced — flash the summary for the iteration that
      // just finished.
      const completed = prevIter;
      const count = telemetry?.preFilterByIter[completed] ?? 0;
      setFlashMessage(`iter ${completed}: ${count} tok`);
      const t = setTimeout(() => setFlashMessage(null), 1500);
      setPrevIter(iter);
      return () => clearTimeout(t);
    }
    if (iter !== prevIter) setPrevIter(iter);
  }, [telemetry?.iteration, telemetry?.preFilterByIter, prevIter]);

  if (!isThreadRunning) return null;
  if (!telemetry) return null;

  const currentIterCount = telemetry.tokens?.preFilterTokens ?? 0;
  const cumulative = telemetry.cumulativePreFilter + currentIterCount;
  if (cumulative <= 0) return null;

  const tps = telemetry.tokens?.tps ?? null;

  return (
    <div
      data-testid="token-counter-chip"
      className="mb-1 flex w-full flex-row items-center gap-2 px-1.5"
    >
      <div className="flex items-center gap-2 rounded-full border border-muted-foreground/15 bg-muted/20 px-2.5 py-1 text-[11px] text-muted-foreground tabular-nums">
        <span>
          <span className="font-medium text-foreground/80">{cumulative}</span> tok
        </span>
        <span className="opacity-40">·</span>
        <span>{formatTps(tps)} t/s</span>
        {flashMessage !== null && (
          <>
            <span className="opacity-40">·</span>
            <span className="animate-pulse text-foreground/70">{flashMessage}</span>
          </>
        )}
      </div>
    </div>
  );
};
