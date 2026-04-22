// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createElement, useCallback, useRef, useState } from "react";
import { toast } from "sonner";
import { ModelLoadDescription } from "../components/model-load-status";
import {
  getDownloadProgress,
  getGgufDownloadProgress,
  getInferenceStatus,
  getLoadProgress,
  listLoras,
  listModels,
  loadModel,
  unloadModel,
  validateModel,
} from "../api/chat-api";
import { formatEta, formatRate } from "../utils/format-transfer";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import type {
  BackendKind,
  InferenceStatusResponse,
  LoadModelResponse,
} from "../types/api";
import type {
  ChatLoraSummary,
  ChatModelSummary,
  InferenceParams,
} from "../types/runtime";

// Chunk F (F2): derive a BackendKind from the boolean flags for
// responses that don't carry a backend_kind field (e.g. ModelDetails,
// BackendModelDetails on /models). The function prefers an explicit
// backend_kind when available and falls back to the booleans so it
// stays compatible with older backends during the migration window.
function resolveBackendKind(source: {
  backend_kind?: BackendKind | null;
  is_gguf?: boolean;
  is_mlx?: boolean;
  is_mlx_lora?: boolean;
  is_mlx_vlm?: boolean;
  is_mlx_audio?: boolean;
}): BackendKind | null {
  if (source.backend_kind) return source.backend_kind;
  if (source.is_mlx_lora) return "mlx+lora";
  if (source.is_mlx_vlm) return "mlx+vlm";
  if (source.is_mlx_audio) return "mlx+audio";
  if (source.is_mlx) return "mlx";
  if (source.is_gguf) return "gguf";
  return null;
}

// Chunk F (F2): a cheap gate that asks "is this a backend that
// populates native context_length + owns its own KV cache?". All four
// MLX variants + GGUF share this shape; pure unsloth/transformers
// loads do not.
function kindHasNativeContextLength(kind: BackendKind | null): boolean {
  return (
    kind === "gguf" ||
    kind === "mlx" ||
    kind === "mlx+lora" ||
    kind === "mlx+vlm" ||
    kind === "mlx+audio"
  );
}

type SelectedModelInput = {
  id: string;
  isLora?: boolean;
  ggufVariant?: string;
  loadingDescription?: string;
  isDownloaded?: boolean;
  expectedBytes?: number;
  forceReload?: boolean;
};

const MODEL_LOAD_TOAST_CLASSNAMES = {
  toast: "items-start gap-2.5",
  content: "gap-0.5 flex-1 min-w-0",
  title: "leading-5",
  description: "mt-0 w-full",
} as const;

const LORA_SUFFIX_RE = /_(\d{9,})$/;

function parseTrailingEpoch(input: string): number | undefined {
  const match = input.match(LORA_SUFFIX_RE);
  if (!match) {
    return undefined;
  }
  const parsed = Number.parseInt(match[1], 10);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function stripTrailingEpoch(input: string): string {
  const cleaned = input.replace(LORA_SUFFIX_RE, "").replace(/[_-]+$/, "").trim();
  return cleaned || input;
}

function describeModel(model: {
  is_lora?: boolean;
  is_vision?: boolean;
  is_gguf?: boolean;
  is_mlx?: boolean;
  is_mlx_vlm?: boolean;
  is_mlx_audio?: boolean;
  is_audio?: boolean;
  has_audio_input?: boolean;
  backend_kind?: BackendKind | null;
}): string | undefined {
  // Chunk F (F2): drive the tag list off ``backend_kind`` first;
  // fall back to the deprecated booleans when the field is missing
  // (pre-Chunk-D servers, ModelDetails-shape payloads, etc).
  const kind = resolveBackendKind(model);
  const tags: string[] = [];
  switch (kind) {
    case "gguf":
      tags.push("GGUF");
      break;
    case "mlx+vlm":
      tags.push("MLX-VLM");
      break;
    case "mlx+audio":
      tags.push("MLX-Audio");
      break;
    case "mlx":
    case "mlx+lora":
      tags.push("MLX");
      break;
    default:
      break;
  }
  if (model.is_lora) tags.push("LoRA");
  if (model.is_vision && kind !== "mlx+vlm") tags.push("Vision");
  if (model.is_audio && kind !== "mlx+audio") tags.push("Audio");
  if (model.has_audio_input) tags.push("Audio Input");
  if (
    !model.is_lora &&
    !model.is_vision &&
    kind === null &&
    !model.is_audio &&
    !model.has_audio_input
  )
    tags.push("Base");
  return tags.join(" · ");
}

function toChatModelSummary(model: {
  id: string;
  name?: string | null;
  is_lora?: boolean;
  is_vision?: boolean;
  is_gguf?: boolean;
  is_mlx?: boolean;
  is_mlx_vlm?: boolean;
  is_mlx_audio?: boolean;
  is_audio?: boolean;
  audio_type?: string | null;
  has_audio_input?: boolean;
  backend_kind?: BackendKind | null;
}): ChatModelSummary {
  const kind = resolveBackendKind(model);
  return {
    id: model.id,
    name: model.name || model.id,
    description: describeModel(model),
    isLora: Boolean(model.is_lora),
    // A VLM model counts as "vision" for all UI gating: image composer,
    // vision badge, etc. The ``backend_kind === "mlx+vlm"`` check is
    // the new primary signal; the boolean is kept for fallback.
    isVision: Boolean(model.is_vision) || kind === "mlx+vlm",
    // Chunk F (F2): keep the deprecated booleans in the summary for
    // backwards compatibility with existing consumers; add backendKind
    // as the new preferred field. New UI code should read backendKind.
    isGguf: Boolean(model.is_gguf),
    isMlx: Boolean(model.is_mlx),
    isMlxVlm: Boolean(model.is_mlx_vlm),
    isMlxAudio: Boolean(model.is_mlx_audio),
    backendKind: kind,
    // Same treatment for audio: an MLX-Audio model is still "audio"
    // for gating the TTS composer even though the dedicated backend
    // is mlx-audio rather than the GGUF audio path.
    isAudio: Boolean(model.is_audio) || kind === "mlx+audio",
    audioType: model.audio_type ?? null,
    hasAudioInput: Boolean(model.has_audio_input),
  };
}

function toLoraSummary(lora: {
  display_name: string;
  adapter_path: string;
  base_model?: string | null;
  source?: "training" | "exported" | null;
  export_type?: "lora" | "merged" | "gguf" | null;
}): ChatLoraSummary {
  const idTail = lora.adapter_path.split("/").filter(Boolean).at(-1) ?? "";
  const updatedAt =
    parseTrailingEpoch(lora.display_name) ?? parseTrailingEpoch(idTail);

  return {
    id: lora.adapter_path,
    name: stripTrailingEpoch(lora.display_name),
    baseModel: lora.base_model || "Unknown base model",
    updatedAt,
    source: lora.source ?? undefined,
    exportType: lora.export_type ?? undefined,
  };
}

function toFiniteNumber(value: unknown): number | undefined {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return undefined;
  }
  return value;
}

function getTrustRemoteCodeRequiredMessage(modelName: string): string {
  return `${modelName} needs custom code enabled to load. Turn on "Enable custom code" in Chat Settings, then try again.`;
}

function mergeRecommendedInference(
  current: InferenceParams,
  response: LoadModelResponse | InferenceStatusResponse,
  modelId: string,
): InferenceParams {
  const inference = response.inference;
  // GGUF / MLX: use actual context length from model metadata, fallback to 131072
  // Other: 4096. Chunk F (F2): route via backend_kind first; fall back
  // to the deprecated booleans for older response payloads.
  const kind = resolveBackendKind(response);
  const defaultMaxTokens = kindHasNativeContextLength(kind)
    ? (response.context_length ?? 131072)
    : 4096;
  return {
    ...current,
    checkpoint: modelId,
    maxTokens: defaultMaxTokens,
    temperature:
      toFiniteNumber(inference?.temperature) ?? current.temperature,
    topP: toFiniteNumber(inference?.top_p) ?? current.topP,
    topK: toFiniteNumber(inference?.top_k) ?? current.topK,
    minP: toFiniteNumber(inference?.min_p) ?? current.minP,
    presencePenalty:
      toFiniteNumber(inference?.presence_penalty) ?? current.presencePenalty,
    trustRemoteCode:
      typeof inference?.trust_remote_code === "boolean"
        ? inference.trust_remote_code
        : current.trustRemoteCode,
  };
}

export function useChatModelRuntime() {
  const params = useChatRuntimeStore((state) => state.params);
  const models = useChatRuntimeStore((state) => state.models);
  const loras = useChatRuntimeStore((state) => state.loras);
  const setModels = useChatRuntimeStore((state) => state.setModels);
  const setLoras = useChatRuntimeStore((state) => state.setLoras);
  const setParams = useChatRuntimeStore((state) => state.setParams);
  const setModelsError = useChatRuntimeStore((state) => state.setModelsError);
  const setCheckpoint = useChatRuntimeStore((state) => state.setCheckpoint);
  const clearCheckpoint = useChatRuntimeStore((state) => state.clearCheckpoint);

  const [loadingModel, setLoadingModel] = useState<{
    id: string;
    displayName: string;
    isDownloaded?: boolean;
    isCachedLora?: boolean;
  } | null>(null);
  const [loadToastDismissed, setLoadToastDismissed] = useState(false);
  const [loadProgress, setLoadProgress] = useState<{
    percent: number | null;
    label: string | null;
    phase: "downloading" | "starting";
    // Chunk E (E4): non-fatal warnings emitted by the MLX backend during
    // load (e.g. memory-headroom advisories). Undefined / empty when none.
    warnings?: string[];
  } | null>(null);
  const loadAbortRef = useRef<AbortController | null>(null);
  const loadingModelRef = useRef<typeof loadingModel>(null);
  const loadToastIdRef = useRef<string | number | null>(null);
  const loadToastDismissedRef = useRef(false);

  const setLoadToastDismissedState = useCallback((dismissed: boolean) => {
    loadToastDismissedRef.current = dismissed;
    setLoadToastDismissed(dismissed);
  }, []);

  const resetLoadingUi = useCallback(() => {
    setLoadingModel(null);
    setLoadProgress(null);
    loadingModelRef.current = null;
    loadAbortRef.current = null;
    loadToastIdRef.current = null;
    setLoadToastDismissedState(false);
    useChatRuntimeStore.getState().setModelLoading(false);
  }, [setLoadToastDismissedState]);

  const renderLoadDescription = useCallback(
    (
      title: string,
      message: string,
      progressPercent?: number | null,
      progressLabel?: string | null,
      onStop?: () => void,
    ) =>
      createElement(ModelLoadDescription, {
        title,
        message,
        progressPercent,
        progressLabel,
        onStop,
      }),
    [],
  );

  const refresh = useCallback(async () => {
    setModelsError(null);
    try {
      const [listRes, statusRes, lorasRes] = await Promise.all([
        listModels(),
        getInferenceStatus(),
        listLoras(),
      ]);

      setModels(listRes.models.map(toChatModelSummary));
      setLoras(lorasRes.loras.map(toLoraSummary));

      if (statusRes.active_model) {
        setCheckpoint(statusRes.active_model, statusRes.gguf_variant);

        // Apply inference defaults on reconnect (page refresh with model already loaded)
        if (statusRes.inference) {
          const currentParams = useChatRuntimeStore.getState().params;
          const reconnectResponse: LoadModelResponse = {
            status: "already_loaded",
            model: statusRes.active_model,
            display_name: statusRes.active_model,
            is_vision: statusRes.is_vision,
            is_lora: false,
            is_gguf: statusRes.is_gguf,
            is_mlx: statusRes.is_mlx,
            is_mlx_vlm: statusRes.is_mlx_vlm,
            is_mlx_audio: statusRes.is_mlx_audio,
            backend_kind: statusRes.backend_kind,
            is_audio: statusRes.is_audio,
            audio_type: statusRes.audio_type,
            has_audio_input: statusRes.has_audio_input,
            inference: statusRes.inference,
            context_length: statusRes.context_length,
            max_context_length: statusRes.max_context_length,
            native_context_length: statusRes.native_context_length,
            supports_reasoning: statusRes.supports_reasoning,
            reasoning_always_on: statusRes.reasoning_always_on,
            supports_tools: statusRes.supports_tools,
            speculative_type: statusRes.speculative_type,
          };
          setParams(
            mergeRecommendedInference(currentParams, statusRes, statusRes.active_model),
          );
        }

        // Restore reasoning/tools support flags and context length
        const supportsReasoning = statusRes.supports_reasoning ?? false;
        const reasoningAlwaysOn = statusRes.reasoning_always_on ?? false;
        const supportsTools = statusRes.supports_tools ?? false;
        // Chunk F (F2): route off backend_kind with a boolean fallback.
        // GGUF / any MLX variant populates context_length from model
        // metadata; treat them identically for derivation of slider caps.
        const _statusKind = resolveBackendKind(statusRes);
        const _hasNativeCtx = kindHasNativeContextLength(_statusKind);
        const currentGgufContextLength = _hasNativeCtx
          ? (statusRes.context_length ?? null)
          : null;
        const ggufMaxContextLength = _hasNativeCtx
          ? (statusRes.max_context_length ?? null)
          : null;
        const ggufNativeContextLength = _hasNativeCtx
          ? (statusRes.native_context_length ?? null)
          : null;
        const currentSpecType = statusRes.speculative_type ?? null;
        useChatRuntimeStore.setState({
          supportsReasoning,
          reasoningAlwaysOn,
          supportsTools,
          // Phase 3/7/8: track MLX-active so the settings sheet can
          // gate KV / speculative / context UI on ``isGguf || isMlx``.
          // Chunk F (F2): prefer the enum; keep boolean mirrors in the
          // store for existing consumers until we flip those too.
          activeIsMlx:
            _statusKind === "mlx" || _statusKind === "mlx+lora",
          activeIsMlxVlm: _statusKind === "mlx+vlm",
          activeIsMlxAudio: _statusKind === "mlx+audio",
          activeBackendKind: _statusKind,
          // Status restore path: ``hf_variant`` is populated for GGUF
          // (mirrors ``gguf_variant``) and MLX. Fall back to the legacy
          // ``gguf_variant`` field for older backend builds so a stale
          // client still renders the chip correctly.
          activeHfVariant:
            statusRes.hf_variant ?? statusRes.gguf_variant ?? null,
          ggufContextLength: currentGgufContextLength,
          ggufMaxContextLength,
          ggufNativeContextLength,
          modelRequiresTrustRemoteCode:
            statusRes.requires_trust_remote_code ?? false,
          speculativeType: currentSpecType,
          loadedSpeculativeType: currentSpecType,
        });

        // Set reasoning default for Qwen3.5/3.6 small models
        if (supportsReasoning) {
          let reasoningDefault = true;
          const mid = statusRes.active_model.toLowerCase();
          if (mid.includes("qwen3.5") || mid.includes("qwen3.6")) {
            const sizeMatch = mid.match(/(\d+\.?\d*)\s*b/);
            if (sizeMatch && parseFloat(sizeMatch[1]) < 9) {
              reasoningDefault = false;
            }
          }
          useChatRuntimeStore.getState().setReasoningEnabled(reasoningDefault);
        }
      } else {
        useChatRuntimeStore.setState({
          modelRequiresTrustRemoteCode: false,
        });
      }
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Failed to load models";
      setModelsError(message);
      toast.error("Failed to refresh models", {
        description: message,
      });
    }
  }, [setCheckpoint, setLoras, setModels, setModelsError, setParams]);

  const cancelLoading = useCallback(() => {
    const model = loadingModelRef.current;
    if (!model) return;
    loadAbortRef.current?.abort();
    loadAbortRef.current = null;
    loadingModelRef.current = null;
    const tid = loadToastIdRef.current;
    loadToastIdRef.current = null;
    setLoadingModel(null);
    setLoadProgress(null);
    setLoadToastDismissedState(false);
    clearCheckpoint();
    if (tid != null) toast.dismiss(tid);
    const isCachedOrLocal = model.isDownloaded || model.isCachedLora;
    toast.info("Stopped loading model", {
      description: isCachedOrLocal
        ? undefined
        : "The current download may still finish in the background.",
    });
    // Fire-and-forget: tell backend to stop, don't block UI
    unloadModel({ model_path: model.id }).catch(() => {});
  }, [clearCheckpoint, setLoadToastDismissedState]);

  const selectModel = useCallback(
    async (selection: string | SelectedModelInput) => {
      const modelId = typeof selection === "string" ? selection : selection.id;
      const ggufVariant =
        typeof selection === "string" ? undefined : selection.ggufVariant;
      const forceReload =
        typeof selection === "string" ? false : selection.forceReload ?? false;
      const currentVariant = useChatRuntimeStore.getState().activeGgufVariant;
      if (!forceReload && (!modelId || (params.checkpoint === modelId && (ggufVariant ?? null) === (currentVariant ?? null)))) {
        return;
      }
      // Prevent duplicate loads if already loading this model
      if (loadingModelRef.current?.id === modelId) return;

      const explicitIsLora =
        typeof selection === "string" ? undefined : selection.isLora;
      const extraLoadingDescription =
        typeof selection === "string" ? undefined : selection.loadingDescription;
      const isDownloaded =
        typeof selection === "string" ? false : selection.isDownloaded ?? false;
      const model = models.find((entry) => entry.id === modelId);
      const lora = loras.find((entry) => entry.id === modelId);
      const loraIsAdapter = lora?.exportType === "lora";
      const isLora =
        explicitIsLora ?? model?.isLora ?? loraIsAdapter ?? false;
      const displayName = model?.name || lora?.name || modelId;
      const currentCheckpoint =
        useChatRuntimeStore.getState().params.checkpoint;
      const previousCheckpoint = currentCheckpoint;
      const previousVariant =
        useChatRuntimeStore.getState().activeGgufVariant ?? null;
      const previousModel = previousCheckpoint
        ? models.find((entry) => entry.id === previousCheckpoint)
        : undefined;
      const previousLora = previousCheckpoint
        ? loras.find((entry) => entry.id === previousCheckpoint)
        : undefined;
      const previousIsLora =
        previousModel?.isLora ?? (previousLora?.exportType === "lora");
      // Covers Unix absolute (/), relative (./  ../), tilde (~/), Windows drive (C:\), UNC (\\server)
      const isLocal = /^(\/|\.{1,2}[\\/]|~[\\/]|[A-Za-z]:[\\/]|\\\\)/.test(modelId);
      const isCachedLora = isLora && isLocal;
      const loadingDescription = [
        currentCheckpoint ? "Switching models." : null,
        extraLoadingDescription ?? null,
        isDownloaded ? "Loading cached model into memory." : null,
        !isDownloaded && isCachedLora ? "Loading trained model into memory." : null,
      ]
        .filter(Boolean)
        .join(" ");
      setModelsError(null);
      setLoadToastDismissedState(false);
      const loadInfo = { id: modelId, displayName, isDownloaded, isCachedLora };
      setLoadingModel(loadInfo);
      useChatRuntimeStore.getState().setModelLoading(true);
      setLoadProgress(
        isDownloaded || isCachedLora
          ? { percent: null, label: null, phase: "starting" }
          : { percent: 0, label: "Preparing download", phase: "downloading" },
      );
      loadingModelRef.current = loadInfo;
      const abortCtrl = new AbortController();
      loadAbortRef.current = abortCtrl;
      try {
        async function performLoad(): Promise<void> {
          if (abortCtrl.signal.aborted) throw new Error("Cancelled");
          let previousWasUnloaded = false;
          const currentCheckpoint =
            useChatRuntimeStore.getState().params.checkpoint;
          const paramsBeforeLoad = useChatRuntimeStore.getState().params;
          const trustRemoteCode = paramsBeforeLoad.trustRemoteCode ?? false;
          const maxSeqLength = paramsBeforeLoad.maxSeqLength;
          const hfToken = useChatRuntimeStore.getState().hfToken || null;
          const previousModelRequiresTrustRemoteCode =
            useChatRuntimeStore.getState().modelRequiresTrustRemoteCode;
          try {
            // Lightweight pre-flight validation: avoid unloading a working model
            // if the new identifier is clearly invalid (e.g. bad HF id / path).
            const validation = await validateModel({
              model_path: modelId,
              hf_token: hfToken,
              max_seq_length: maxSeqLength,
              load_in_4bit: true,
              is_lora: isLora,
              gguf_variant: ggufVariant ?? null,
            });
            if (validation.requires_trust_remote_code && !trustRemoteCode) {
              throw new Error(getTrustRemoteCodeRequiredMessage(displayName));
            }

            if (currentCheckpoint) {
              await unloadModel({ model_path: currentCheckpoint });
              previousWasUnloaded = true;
            }

            const { chatTemplateOverride, kvCacheDtype, customContextLength, ggufContextLength, speculativeType, draftModelPath } = useChatRuntimeStore.getState();
            // GGUF / MLX / MLX-VLM / MLX-Audio: use custom context length,
            // or 0 = model's native context (backend reads max_position_embeddings).
            // Unsloth/transformers: use the Max Seq Length slider value
            // because there's no native-context concept.
            //
            // Chunk H-2 follow-up: without the hasNativeCtx gate, MLX loads
            // clamped to the legacy 4096 slider default — e.g. Bonsai 8B
            // surfaced as 4096 despite max_position_embeddings=65536, because
            // MlxLmBackend.load_model applies min(native, n_ctx) when n_ctx>0.
            // Chunk G exposes backend_kind on ValidateModelResponse, so we
            // can dispatch before the load without another round-trip.
            const validatedKind = validation.backend_kind ?? null;
            const hasNativeCtx =
              validatedKind === "gguf" ||
              validatedKind === "mlx" ||
              validatedKind === "mlx+lora" ||
              validatedKind === "mlx+vlm" ||
              validatedKind === "mlx+audio";
            const isDirectGgufFile = modelId.toLowerCase().endsWith(".gguf");
            const effectiveMaxSeqLength = customContextLength != null
              ? customContextLength
              : hasNativeCtx || ggufVariant != null || isDirectGgufFile
                ? (ggufContextLength ?? 0)
                : maxSeqLength;
            const loadResponse = await loadModel({
              model_path: modelId,
              hf_token: hfToken,
              max_seq_length: effectiveMaxSeqLength,
              load_in_4bit: true,
              is_lora: isLora,
              gguf_variant: ggufVariant ?? null,
              trust_remote_code: trustRemoteCode,
              chat_template_override: chatTemplateOverride,
              cache_type_kv: kvCacheDtype,
              speculative_type: speculativeType,
              // Phase 7: MLX draft-model path for speculative decoding.
              // Only meaningful when speculativeType === "mlx-draft-model"
              // — the backend ignores it otherwise. We forward the raw
              // value so a user can leave it set across reloads.
              draft_model_path: draftModelPath,
            });

            // If cancelled while loading, don't update UI to show
            // the model as active -- it's being unloaded.
            if (abortCtrl.signal.aborted) throw new Error("Cancelled");

            const currentParams = useChatRuntimeStore.getState().params;
            setParams(
              mergeRecommendedInference(currentParams, loadResponse, modelId),
            );
            // Qwen3.5/3.6 small models (0.8B, 2B, 4B, 9B) disable thinking by default
            let reasoningDefault = loadResponse.supports_reasoning ?? false;
            if (reasoningDefault) {
              const mid = modelId.toLowerCase();
              if (mid.includes("qwen3.5") || mid.includes("qwen3.6")) {
                const sizeMatch = mid.match(/(\d+\.?\d*)\s*b/);
                if (sizeMatch && parseFloat(sizeMatch[1]) < 9) {
                  reasoningDefault = false;
                }
              }
            }
            const loadedKv = loadResponse.cache_type_kv ?? null;
            const loadedSpec = loadResponse.speculative_type ?? null;
            // Chunk F (F2): route off backend_kind with a boolean fallback.
            const _loadKind = resolveBackendKind(loadResponse);
            const _hasNativeCtx = kindHasNativeContextLength(_loadKind);
            const nativeCtx = _hasNativeCtx
              ? (loadResponse.context_length ?? 131072)
              : null;
            const reportedMaxCtx = _hasNativeCtx
              ? (loadResponse.max_context_length ?? null)
              : null;
            const reportedNativeCtx = _hasNativeCtx
              ? (loadResponse.native_context_length ?? null)
              : null;
            // A successful reload has applied settings, so clear pending custom
            // context state and display the backend-reported effective context.
            const keepCustomCtx = null;
            const reasoningAlwaysOn = loadResponse.reasoning_always_on ?? false;
            const ggufMaxContextLength = reportedMaxCtx;
            useChatRuntimeStore.setState({
              // Phase 3/7/8: MLX load also gates settings-sheet controls.
              // Chunk F (F2): derive from the resolved backend_kind.
              activeIsMlx:
                _loadKind === "mlx" || _loadKind === "mlx+lora",
              activeIsMlxVlm: _loadKind === "mlx+vlm",
              activeIsMlxAudio: _loadKind === "mlx+audio",
              activeBackendKind: _loadKind,
              // Chunk H-2 MLX parity — surface the backend-reported
              // quant variant for chip display. For GGUF this is the
              // same as ``activeGgufVariant`` (set by ``setCheckpoint``);
              // for MLX it comes from ``LoadResponse.hf_variant`` which
              // ``setCheckpoint`` doesn't know about.
              activeHfVariant:
                loadResponse.hf_variant ?? ggufVariant ?? null,
              ggufContextLength: nativeCtx,
              ggufMaxContextLength,
              ggufNativeContextLength: reportedNativeCtx,
              modelRequiresTrustRemoteCode:
                loadResponse.requires_trust_remote_code ?? false,
              supportsReasoning: loadResponse.supports_reasoning ?? false,
              reasoningAlwaysOn,
              reasoningEnabled: reasoningAlwaysOn ? true : reasoningDefault,
              supportsTools: loadResponse.supports_tools ?? false,
              toolsEnabled: loadResponse.supports_tools ?? false,
              codeToolsEnabled: loadResponse.supports_tools ?? false,
              kvCacheDtype: loadedKv,
              loadedKvCacheDtype: loadedKv,
              speculativeType: loadedSpec,
              loadedSpeculativeType: loadedSpec,
              // Phase 7: the backend doesn't echo the draft path in the
              // response (it's derived from the request), so reuse the
              // value we just sent as the "loaded" baseline.
              loadedDraftModelPath: draftModelPath,
              customContextLength: keepCustomCtx,
              defaultChatTemplate: loadResponse.chat_template ?? null,
              chatTemplateOverride: null,
            });
            // Qwen3/3.5/3.6: apply thinking-mode-specific params after load
            if (modelId.toLowerCase().includes("qwen3") && (loadResponse.supports_reasoning ?? false)) {
              const store = useChatRuntimeStore.getState();
              const mid = modelId.toLowerCase();
              const needsPresencePenalty = mid.includes("qwen3.5") || mid.includes("qwen3.6");
              const p = reasoningDefault
                ? { temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0, ...(needsPresencePenalty ? { presencePenalty: 1.5 } : {}) }
                : { temperature: 0.7, topP: 0.8, topK: 20, minP: 0.0, ...(needsPresencePenalty ? { presencePenalty: 1.5 } : {}) };
              store.setParams({ ...store.params, ...p });
            }
            await refresh();
          } catch (error) {
            // Skip rollback if user cancelled -- model is already being unloaded.
            if (abortCtrl.signal.aborted) throw error;
            // If we unloaded a previous model and the new load failed, attempt a rollback.
            if (previousWasUnloaded && previousCheckpoint) {
              try {
                await loadModel({
                  model_path: previousCheckpoint,
                  hf_token: hfToken,
                  max_seq_length: maxSeqLength,
                  load_in_4bit: true,
                  is_lora: previousIsLora,
                  gguf_variant: previousVariant,
                  trust_remote_code:
                    previousModelRequiresTrustRemoteCode || trustRemoteCode,
                });
                await refresh();
              } catch {
                // If rollback also fails, surface the original error.
              }
            }
            throw error;
          }
        }

        const isCachedLoad = isDownloaded || isCachedLora;
        const toastTitle = isCachedLoad ? "Starting model…" : "Downloading model…";
        const toastId = toast(
          null,
          {
            description: renderLoadDescription(
              toastTitle,
              loadingDescription,
              isCachedLoad ? null : 0,
              isCachedLoad ? null : "Preparing download",
              cancelLoading,
            ),
            duration: Infinity,
            closeButton: false,
            classNames: MODEL_LOAD_TOAST_CLASSNAMES,
            onDismiss: (dismissedToast) => {
              if (loadToastIdRef.current !== dismissedToast.id) {
                return;
              }
              setLoadToastDismissedState(true);
            },
          },
        );
        loadToastIdRef.current = toastId;

        // Poll download progress for non-cached models (GGUF and non-GGUF).
        // Then, once the download wraps (or for already-cached models),
        // poll the llama-server mmap phase so "Starting model..." no
        // longer looks frozen for several minutes on large MoE models.
        let progressInterval: ReturnType<typeof setInterval> | null = null;
        const expectedBytes =
          typeof selection !== "string" ? selection.expectedBytes ?? 0 : 0;

        // Rolling window of byte samples for rate / ETA estimation.
        // Shared across download + mmap phases so the estimator doesn't
        // reset when the phase flips.
        type Sample = { t: number; b: number };
        const MIN_SAMPLES = 3;
        const MIN_WINDOW = 3_000; // ms
        const MAX_WINDOW = 15_000; // ms
        const dlSamples: Sample[] = [];
        const mmapSamples: Sample[] = [];

        function estimate(
          samples: Sample[],
          bytes: number,
          total: number,
        ): { rate: number; eta: number; stable: boolean } {
          const now = Date.now();
          // Drop samples if the counter reset (e.g. phase flipped).
          if (samples.length > 0 && bytes < samples[samples.length - 1].b) {
            samples.length = 0;
          }
          samples.push({ t: now, b: bytes });
          const cutoff = now - MAX_WINDOW;
          while (samples.length > 2 && samples[0].t < cutoff) {
            samples.shift();
          }
          if (samples.length < MIN_SAMPLES) {
            return { rate: 0, eta: 0, stable: false };
          }
          const first = samples[0];
          const last = samples[samples.length - 1];
          const dt = (last.t - first.t) / 1000;
          const db = last.b - first.b;
          if (dt * 1000 < MIN_WINDOW || db <= 0) {
            return { rate: 0, eta: 0, stable: false };
          }
          const rate = db / dt;
          const eta =
            total > 0 && bytes < total && rate > 0 ? (total - bytes) / rate : 0;
          return { rate, eta, stable: true };
        }

        function composeProgressLabel(
          dlGb: number,
          totalGb: number,
          bytes: number,
          total: number,
          samples: Sample[],
        ): string {
          const base =
            totalGb > 0
              ? `${dlGb.toFixed(1)} of ${totalGb.toFixed(1)} GB`
              : `${dlGb.toFixed(1)} GB downloaded`;
          const est = estimate(samples, bytes, total);
          if (!est.stable) return base;
          const rateStr = formatRate(est.rate);
          const etaStr = total > 0 ? formatEta(est.eta) : "";
          return etaStr && etaStr !== "--"
            ? `${base} • ${rateStr} • ${etaStr} left`
            : `${base} • ${rateStr}`;
        }

        let downloadComplete = isDownloaded || isCachedLora;

        const pollDownload = async () => {
          if (abortCtrl.signal.aborted || !loadingModelRef.current) {
            if (progressInterval) clearInterval(progressInterval);
            return;
          }
          try {
            const prog =
              ggufVariant && expectedBytes > 0
                ? await getGgufDownloadProgress(modelId, ggufVariant, expectedBytes)
                : await getDownloadProgress(modelId);
            if (!loadingModelRef.current) return;

            if (prog.progress > 0 && prog.progress < 1) {
              hasShownProgress = true;
              const dlGb = prog.downloaded_bytes / (1024 ** 3);
              const totalGb = prog.expected_bytes / (1024 ** 3);
              const pct = Math.round(prog.progress * 100);
              const progressLabel = composeProgressLabel(
                dlGb,
                totalGb,
                prog.downloaded_bytes,
                prog.expected_bytes,
                dlSamples,
              );
              setLoadProgress({
                percent: pct,
                label: progressLabel,
                phase: "downloading",
              });
              if (loadToastDismissedRef.current) return;
              toast(null, {
                id: toastId,
                description: renderLoadDescription(
                  "Downloading model…",
                  loadingDescription,
                  pct,
                  progressLabel,
                  cancelLoading,
                ),
                duration: Infinity,
                closeButton: false,
                classNames: MODEL_LOAD_TOAST_CLASSNAMES,
                onDismiss: (dismissedToast) => {
                  if (loadToastIdRef.current !== dismissedToast.id) return;
                  setLoadToastDismissedState(true);
                },
              });
            } else if (
              prog.downloaded_bytes > 0 &&
              prog.expected_bytes === 0 &&
              prog.progress === 0
            ) {
              hasShownProgress = true;
              const dlGb = prog.downloaded_bytes / (1024 ** 3);
              const est = estimate(dlSamples, prog.downloaded_bytes, 0);
              const rateSuffix =
                est.stable ? ` • ${formatRate(est.rate)}` : "";
              setLoadProgress({
                percent: null,
                label: `${dlGb.toFixed(1)} GB downloaded${rateSuffix}`,
                phase: "downloading",
              });
            } else if (prog.progress >= 1 && hasShownProgress) {
              downloadComplete = true;
              setLoadProgress({
                percent: 100,
                label: "Download complete",
                phase: "starting",
              });
              if (!loadToastDismissedRef.current) {
                toast(null, {
                  id: toastId,
                  description: renderLoadDescription(
                    "Starting model…",
                    "Download complete. Loading the model into memory.",
                    100,
                    "Download complete",
                    cancelLoading,
                  ),
                  duration: Infinity,
                  closeButton: false,
                  classNames: MODEL_LOAD_TOAST_CLASSNAMES,
                  onDismiss: (dismissedToast) => {
                    if (loadToastIdRef.current !== dismissedToast.id) return;
                    setLoadToastDismissedState(true);
                  },
                });
              }
              // Keep polling: the mmap branch below takes over from here.
            }
          } catch {
            // Ignore polling errors; keep polling.
          }
        };

        const pollLoad = async () => {
          if (abortCtrl.signal.aborted || !loadingModelRef.current) {
            if (progressInterval) clearInterval(progressInterval);
            return;
          }
          try {
            const prog = await getLoadProgress();
            if (!loadingModelRef.current) return;
            if (!prog || prog.phase == null) return;
            if (prog.phase === "ready") {
              // Loaded. The chat flow will flip loadingModelRef shortly;
              // just stop polling.
              if (progressInterval) clearInterval(progressInterval);
              return;
            }
            if (prog.bytes_total <= 0) return; // nothing useful to render
            const loadedGb = prog.bytes_loaded / (1024 ** 3);
            const totalGb = prog.bytes_total / (1024 ** 3);
            const pct = Math.min(99, Math.round(prog.fraction * 100));
            const est = estimate(mmapSamples, prog.bytes_loaded, prog.bytes_total);
            const base = `${loadedGb.toFixed(1)} of ${totalGb.toFixed(1)} GB in memory`;
            const label = est.stable
              ? `${base} • ${formatRate(est.rate)}${
                  formatEta(est.eta) !== "--" ? ` • ${formatEta(est.eta)} left` : ""
                }`
              : base;
            setLoadProgress({
              percent: pct,
              label,
              phase: "starting",
              warnings: prog.warnings && prog.warnings.length > 0 ? prog.warnings : undefined,
            });
            if (loadToastDismissedRef.current) return;
            toast(null, {
              id: toastId,
              description: renderLoadDescription(
                "Starting model…",
                "Paging weights into memory.",
                pct,
                label,
                cancelLoading,
              ),
              duration: Infinity,
              closeButton: false,
              classNames: MODEL_LOAD_TOAST_CLASSNAMES,
              onDismiss: (dismissedToast) => {
                if (loadToastIdRef.current !== dismissedToast.id) return;
                setLoadToastDismissedState(true);
              },
            });
          } catch {
            // Ignore polling errors.
          }
        };

        const pollProgress = async () => {
          if (!downloadComplete) {
            await pollDownload();
          } else {
            await pollLoad();
          }
        };

        let hasShownProgress = false;
        setTimeout(pollProgress, 500);
        progressInterval = setInterval(pollProgress, 2000);

        try {
          await performLoad();
          if (loadToastDismissedRef.current) {
            toast.success(`${displayName} loaded`);
          } else {
            toast.success(`${displayName} loaded`, {
              id: toastId,
              description: undefined,
              closeButton: false,
              duration: 2000,
            });
          }
        } catch (err) {
          if (!abortCtrl.signal.aborted) {
            const message =
              err instanceof Error ? err.message : "Failed to load model";
            if (loadToastDismissedRef.current) {
              toast.error(message);
            } else {
              toast.error(message, {
                id: toastId,
                description: undefined,
                closeButton: false,
                duration: 5000,
              });
            }
          }
          throw err;
        } finally {
          if (progressInterval) clearInterval(progressInterval);
          resetLoadingUi();
        }
      } catch (error) {
        if (abortCtrl.signal.aborted) return; // User cancelled, nothing to report
        resetLoadingUi();
        const message =
          error instanceof Error ? error.message : "Failed to load model";
        setModelsError(message);
      }
    },
    [
      cancelLoading,
      loras,
      models,
      params.checkpoint,
      refresh,
      renderLoadDescription,
      resetLoadingUi,
      setLoadToastDismissedState,
      setModelsError,
      setParams,
    ],
  );

  const ejectModel = useCallback(async () => {
    if (!params.checkpoint) {
      return;
    }
    setModelsError(null);
    try {
      async function performUnload(): Promise<void> {
        await unloadModel({ model_path: params.checkpoint });
        clearCheckpoint();
        await refresh();
      }

      await toast.promise(performUnload(), {
        loading: "Unloading model",
        success: { message: "Model unloaded", duration: 1200 },
        error: (err) =>
          err instanceof Error ? err.message : "Failed to unload model",
        description: "Releases VRAM and resets inference state.",
      });
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Failed to unload model";
      setModelsError(message);
    }
  }, [clearCheckpoint, params.checkpoint, refresh, setModelsError]);

  return {
    refresh,
    selectModel,
    ejectModel,
    cancelLoading,
    loadingModel,
    loadProgress,
    loadToastDismissed,
  };
}
