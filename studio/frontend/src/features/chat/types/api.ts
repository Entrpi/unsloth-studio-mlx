// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Chunk D (Phase 9+10) — additive backend discriminator enum.
 * Chunk F (F2) — promoted to primary source of truth. Prefer this
 * over the deprecated is_gguf / is_mlx / is_mlx_vlm / is_mlx_audio /
 * is_mlx_lora booleans in all UI code. The booleans are still
 * populated on every response for backwards compatibility with
 * external API consumers; removal is a future chunk.
 */
export type BackendKind =
  | "gguf"
  | "mlx"
  | "mlx+lora"
  | "mlx+vlm"
  | "mlx+audio"
  | "unsloth";

export interface BackendModelDetails {
  id: string;
  name?: string | null;
  is_vision?: boolean;
  is_lora?: boolean;
  /** @deprecated Chunk F (F2) — prefer backend_kind-style routing. */
  is_gguf?: boolean;
  /** @deprecated Chunk F (F2) — prefer backend_kind-style routing. */
  is_mlx?: boolean;
  /** @deprecated Chunk F (F2) — prefer backend_kind-style routing. */
  is_mlx_vlm?: boolean;
  /** @deprecated Chunk F (F2) — prefer backend_kind-style routing. */
  is_mlx_audio?: boolean;
  is_audio?: boolean;
  audio_type?: string | null;
  has_audio_input?: boolean;
}

export interface ListModelsResponse {
  models: BackendModelDetails[];
  default_models: string[];
}

export interface BackendLoraInfo {
  display_name: string;
  adapter_path: string;
  base_model?: string | null;
  source?: "training" | "exported" | null;
  export_type?: "lora" | "merged" | "gguf" | null;
}

export interface ListLorasResponse {
  loras: BackendLoraInfo[];
  outputs_dir: string;
}

export interface LoadModelRequest {
  model_path: string;
  hf_token: string | null;
  max_seq_length: number;
  load_in_4bit: boolean;
  is_lora: boolean;
  gguf_variant?: string | null;
  /** Allow loading models with custom code (e.g. NVIDIA Nemotron). Only enable for repos you trust. */
  trust_remote_code?: boolean;
  chat_template_override?: string | null;
  cache_type_kv?: string | null;
  speculative_type?: string | null;
  /** Phase 6 — absolute path to an MLX LoRA adapter (ignored for non-MLX backends). */
  adapter_path?: string | null;
  /** Phase 7 — absolute path or HF repo of an MLX draft model for speculative decoding. */
  draft_model_path?: string | null;
}

export interface ValidateModelResponse {
  valid: boolean;
  message: string;
  identifier?: string | null;
  display_name?: string | null;
  /** @deprecated Chunk F (F2) — prefer backend_kind-style routing. */
  is_gguf?: boolean;
  /** @deprecated Chunk F (F2) — prefer backend_kind-style routing. */
  is_mlx?: boolean;
  /** @deprecated Chunk F (F2) — prefer backend_kind-style routing. */
  is_mlx_vlm?: boolean;
  /** @deprecated Chunk F (F2) — prefer backend_kind-style routing. */
  is_mlx_audio?: boolean;
  is_lora?: boolean;
  is_vision?: boolean;
  requires_trust_remote_code?: boolean;
  /**
   * Primary source-of-truth for backend routing; set server-side from
   * ModelConfig during validation. Chunk G (G2) added this on the backend
   * schema; matching TS surface followed in the Chunk H-2 context-length
   * fix so pre-load routing can dispatch on the enum without another
   * round-trip.
   */
  backend_kind?: BackendKind | null;
}

export interface GgufVariantDetail {
  filename: string;
  quant: string;
  size_bytes: number;
  downloaded?: boolean;
}

export interface GgufVariantsResponse {
  repo_id: string;
  variants: GgufVariantDetail[];
  has_vision: boolean;
  default_variant: string | null;
}

export interface MlxVariantDetail {
  repo_id: string;
  quant: string;
  size_bytes: number;
  downloaded?: boolean;
}

export interface MlxVariantsResponse {
  repo_id: string;
  variants: MlxVariantDetail[];
  default_variant: string | null;
}

export interface LoadModelResponse {
  status: string;
  model: string;
  display_name: string;
  is_vision: boolean;
  is_lora: boolean;
  /** @deprecated Chunk F (F2) — prefer backend_kind === "gguf". */
  is_gguf?: boolean;
  /** @deprecated Chunk F (F2) — prefer backend_kind === "mlx". */
  is_mlx?: boolean;
  /**
   * @deprecated Chunk F (F2) — prefer backend_kind === "mlx+lora".
   * Phase 6 — the active MLX load has a LoRA adapter fused on top of the base model.
   */
  is_mlx_lora?: boolean;
  /** @deprecated Chunk F (F2) — prefer backend_kind === "mlx+vlm". */
  is_mlx_vlm?: boolean;
  /** @deprecated Chunk F (F2) — prefer backend_kind === "mlx+audio". */
  is_mlx_audio?: boolean;
  /** Chunk F (F2) — primary source of truth for which backend owns the active model. */
  backend_kind?: BackendKind | null;
  is_audio?: boolean;
  audio_type?: string | null;
  has_audio_input?: boolean;
  inference?: {
    temperature?: number;
    top_p?: number;
    top_k?: number;
    min_p?: number;
    presence_penalty?: number;
    trust_remote_code?: boolean;
  };
  requires_trust_remote_code?: boolean;
  context_length?: number | null;
  max_context_length?: number | null;
  native_context_length?: number | null;
  supports_reasoning?: boolean;
  reasoning_always_on?: boolean;
  supports_tools?: boolean;
  cache_type_kv?: string | null;
  chat_template?: string | null;
  speculative_type?: string | null;
  /**
   * Generic quant variant for the loaded model. Populated for both
   * GGUF (``"Q4_K_M"``, ``"UD-IQ1_S"``) and MLX (``"4bit"``, ``"8bit"``)
   * backends so the UI can render the chip with quant detail regardless
   * of which backend owns the model.
   */
  hf_variant?: string | null;
}

export interface UnloadModelRequest {
  model_path: string;
}

export interface InferenceStatusResponse {
  active_model: string | null;
  is_vision: boolean;
  /** @deprecated Chunk F (F2) — prefer backend_kind === "gguf". */
  is_gguf?: boolean;
  /** @deprecated Chunk F (F2) — prefer backend_kind === "mlx". */
  is_mlx?: boolean;
  /** @deprecated Chunk F (F2) — prefer backend_kind === "mlx+vlm". */
  is_mlx_vlm?: boolean;
  /** @deprecated Chunk F (F2) — prefer backend_kind === "mlx+audio". */
  is_mlx_audio?: boolean;
  /** Chunk F (F2) — primary source of truth for which backend owns the active model. */
  backend_kind?: BackendKind | null;
  gguf_variant?: string | null;
  /** Generic quant variant (GGUF or MLX). Symmetric with LoadModelResponse.hf_variant. */
  hf_variant?: string | null;
  is_audio?: boolean;
  audio_type?: string | null;
  has_audio_input?: boolean;
  loading: string[];
  loaded: string[];
  inference?: {
    temperature?: number;
    top_p?: number;
    top_k?: number;
    min_p?: number;
    presence_penalty?: number;
    trust_remote_code?: boolean;
  };
  requires_trust_remote_code?: boolean;
  supports_reasoning?: boolean;
  reasoning_always_on?: boolean;
  supports_tools?: boolean;
  context_length?: number | null;
  max_context_length?: number | null;
  native_context_length?: number | null;
  speculative_type?: string | null;
}

export interface AudioGenerationResponse {
  id: string;
  object: string;
  model: string;
  audio: {
    data: string;
    format: string;
    sample_rate: number;
  };
  choices: Array<{
    index: number;
    message: { role: string; content: string };
    finish_reason: string;
  }>;
}

export interface OpenAIChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface OpenAIChatCompletionsRequest {
  model: string;
  messages: OpenAIChatMessage[];
  stream: boolean;
  temperature: number;
  top_p: number;
  max_tokens: number;
  top_k: number;
  min_p: number;
  repetition_penalty: number;
  presence_penalty: number;
  image_base64?: string;
  audio_base64?: string;
  use_adapter?: boolean | string | null;
  enable_thinking?: boolean | null;
  enable_tools?: boolean | null;
  enabled_tools?: string[];
  auto_heal_tool_calls?: boolean;
  max_tool_calls_per_message?: number;
  tool_call_timeout?: number;
  session_id?: string;
}

export interface OpenAIChatDelta {
  role?: string;
  content?: string;
}

export interface OpenAIChatChunkChoice {
  delta?: OpenAIChatDelta;
  finish_reason?: string | null;
}

export interface OpenAIChatChunk {
  choices?: OpenAIChatChunkChoice[];
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  timings?: Record<string, number>;
}
