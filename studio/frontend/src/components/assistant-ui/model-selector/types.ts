// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReactNode } from "react";
import type { BackendKind } from "@/features/chat/types/api";

export interface ModelOption {
  id: string;
  name: string;
  description?: string;
  icon?: ReactNode;
  /**
   * Chunk G (G1) — primary source of truth for which backend owns the
   * option. Populated from ``/inference/status`` / ``/api/inference/load``
   * / ``/v1/models`` where the backend returns ``backend_kind``; derived
   * best-effort (e.g. ``.gguf`` suffix heuristic) for locally-constructed
   * options like search results.
   */
  backendKind?: BackendKind | null;
  /**
   * @deprecated Chunk G (G1) — prefer ``backendKind === "gguf"``. Still
   * populated on every option for compatibility with in-flight consumers;
   * removal is a future chunk.
   */
  isGguf?: boolean;
}

export interface LoraModelOption extends ModelOption {
  baseModel?: string;
  updatedAt?: number;
  source?: "training" | "exported" | "local";
  exportType?: "lora" | "merged" | "gguf";
}

export interface ModelSelectorChangeMeta {
  source: "hub" | "lora" | "exported" | "local";
  isLora: boolean;
  ggufVariant?: string;
  isDownloaded?: boolean;
  expectedBytes?: number;
}
