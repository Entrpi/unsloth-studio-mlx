# Studio environment variables

Centralized reference for all `STUDIO_*` and `VITE_*` environment-flag
knobs Studio reads. Defaults shown are the values a fresh process
picks when the variable is unset. Each entry lists the value you'd
set to opt out / roll back if the feature misbehaves.

Values of `0` / `false` / `no` / `off` all parse as "disabled" for
boolean flags (case-insensitive). Any other non-empty value is
"enabled" unless documented otherwise.

## Streaming architecture (Chunk H-2)

### `STUDIO_SSE_KEEPALIVE_INTERVAL`

Seconds between SSE keepalive comments (`:ka\n\n`) emitted during
silent periods in the `/v1/chat/completions` stream — tool execution
and prompt re-eval, where the generator is blocked for many seconds
without yielding content. Keeps browsers / reverse-proxies from
idle-closing the connection.

- Type: float
- Default: `5.0`
- Range: clamped to `[1.0, 30.0]`
- Disable: `STUDIO_SSE_KEEPALIVE_INTERVAL=0`
- Source: `studio/backend/routes/inference.py` (`_sse_keepalive_interval`)
- Landed: Phase 1 (commit `952e2458`, hotfix `5876b833`)

### `STUDIO_EMIT_PROGRESS_EVENTS`

Whether the MLX agentic loop emits `{"type":"progress","phase":…,"iter":N}`
SSE events at iteration boundaries and the first-token transition.
The frontend renders these as "Re-reading conversation…" /
"Generating…" chips above the composer. Additive; external OpenAI
clients ignore unknown event types.

- Type: boolean
- Default: enabled
- Disable: `STUDIO_EMIT_PROGRESS_EVENTS=0` (or `false` / `no` / `off`)
- Source: `studio/backend/core/inference/mlx_lm.py` (`_progress_events_enabled`)
- Landed: Phase 2 (commit `54054933`)

## Telemetry (Chunk H-2)

### `STUDIO_ENABLE_TELEMETRY_WS`

Whether the `/ws/telemetry` WebSocket endpoint is mounted and the
GPU sampler + token broadcasting run. Governs the backend side of
the live token counter chip and GPU sparkline.

- Type: boolean
- Default: enabled
- Disable: `STUDIO_ENABLE_TELEMETRY_WS=0`
- Source: `studio/backend/main.py` (lifespan hook)
- Landed: Phase 3 (commit `edac4fb9`)

### `STUDIO_TELEMETRY_GPU_SOURCE`

Forces a specific GPU sampling backend instead of the auto-probe
fallback chain. Useful for testing or pinning a specific backend
when the probe picks the "wrong" one.

- Type: string
- Default: `auto` (walks the platform-appropriate chain)
- Values: `auto | ioreport | iokit | mlx_mem | powermetrics | pynvml | amdgpu_sysfs | intel_sysfs | pdh | nvidia_smi_cli | unavailable`
- Platform chains (auto mode):
  - macOS Apple Silicon: `ioreport → iokit → mlx_mem → powermetrics`
  - macOS Intel: (none — chip hides)
  - Linux (native): `pynvml → amdgpu_sysfs → intel_sysfs → nvidia_smi_cli`
  - WSL: `pynvml → nvidia_smi_cli` (sysfs is unreachable under WSL)
  - Windows: `pynvml → pdh → nvidia_smi_cli`
- Rollback: `STUDIO_TELEMETRY_GPU_SOURCE=unavailable` to hide the
  sparkline entirely while leaving token counter + session state
  active.
- Source: `studio/backend/core/telemetry/gpu_sampler.py` (`_candidate_sources`)
- Landed: commit `d8367731`

### `STUDIO_TELEMETRY_ALLOW_POWERMETRICS`

Opt-in gate for the macOS `powermetrics` sampling backend, which
requires passwordless sudo. Disabled by default because missing
sudo would hang the app at startup. If you have `sudo -n true`
configured for `powermetrics`, set this to `1`.

- Type: boolean
- Default: disabled
- Enable: `STUDIO_TELEMETRY_ALLOW_POWERMETRICS=1`
- Source: `studio/backend/core/telemetry/gpu_sampler.py` (`_probe_powermetrics`)
- Landed: Phase 3 (commit `edac4fb9`)

### `VITE_ENABLE_TELEMETRY_WS`

Frontend counterpart to `STUDIO_ENABLE_TELEMETRY_WS`. Controls
whether the React app opens the telemetry WebSocket at mount time.
Value is baked into the frontend bundle at `npm run build` time.

- Type: boolean
- Default: enabled
- Disable (build-time): `VITE_ENABLE_TELEMETRY_WS=0 npm run build`
- Source: `studio/frontend/src/features/chat/hooks/use-telemetry-socket.ts`
- Landed: Phase 3 (commit `edac4fb9`)

## Cross-cutting diagnostic / misc (pre-existing)

These predate Chunk H-2 and are documented here for completeness:

### `STUDIO_FOOTER`, `STUDIO_NODE_TONES`, `STUDIO_ONBOARDING_ICON_TONE`, `STUDIO_ONBOARDING_SURFACE_TONE`, `STUDIO_REFERENCE_BADGE_TONES`, `STUDIO_USER_NODE_TONE`, `STUDIO_WARNING_BADGE_TONE`, `STUDIO_WARNING_ICON_TONE`

Frontend theming / tone overrides. Covered by the theming layer,
not this chunk.

### `STUDIO_UPDATE_CMD`, `STUDIO_UPDATE_FALLBACK_UNIX_CMD`, `STUDIO_UPDATE_FALLBACK_WINDOWS_CMD`

Install / update entry points for the self-update flow. Not touched
by this chunk.

### `STUDIO_SKIP_FLASHATTN_INSTALL`

Existing flag that skips flash-attn install during Linux setup.
Not touched by this chunk.

### `VITE_DATA_DESIGNER_API`

Dataset-designer feature-flag. Not touched by this chunk.

## Rollback playbook

If one of this chunk's new surfaces regresses in production:

- **Keepalive breaks a reverse-proxy**: `STUDIO_SSE_KEEPALIVE_INTERVAL=0`
  restores pre-Phase-1 wire behavior.
- **Progress chip is noisy / wrong**: `STUDIO_EMIT_PROGRESS_EVENTS=0`
  silences the events (UI chip disappears).
- **Telemetry WS connection storms / server load**: `STUDIO_ENABLE_TELEMETRY_WS=0`
  shuts the endpoint + sampler off. Frontend degrades gracefully
  (chip + sparkline hide when WS fails to connect).
- **GPU sampler lies about utilization on a new SKU**:
  `STUDIO_TELEMETRY_GPU_SOURCE=unavailable` hides the sparkline only.
- **WebSocket auth breakage on a specific deployment**: same as
  above — `STUDIO_ENABLE_TELEMETRY_WS=0` is the safe off switch.

All rollbacks are runtime env-var changes — no rebuild or schema
migration required. Frontend-side opt-outs (`VITE_*`) require a
rebuild and so are reserved for permanent deployment-time choices
rather than incident response.
