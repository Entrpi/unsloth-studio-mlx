# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the MLX-LM backend skeleton.

These tests do not require ``mlx_lm`` to be installed — the module is
lazy-imported, so constructing a backend and poking its properties must
work on any platform.
"""

from __future__ import annotations

import sys
import threading

import pytest


def _fresh_backend():
    """Return a freshly-constructed ``MlxLmBackend``.

    Imports at call time so a single test can reload the module for the
    "importable without mlx_lm" case.
    """
    from core.inference.mlx_lm import MlxLmBackend

    return MlxLmBackend()


def test_import_does_not_load_mlx_lm() -> None:
    """Importing the module must not pull in ``mlx_lm``."""
    # Ensure ``mlx_lm`` is not in sys.modules just because we imported ours.
    from core.inference import mlx_lm as _mod  # noqa: F401

    # We cannot assert ``mlx_lm`` is absent from sys.modules — another
    # test module may have imported it. Instead, assert that the backend
    # module itself has no ``mlx_lm`` attribute at top level.
    assert not hasattr(_mod, "mlx_lm_module")
    assert not hasattr(_mod, "mx")


def test_backend_not_loaded_at_init() -> None:
    b = _fresh_backend()
    assert b.is_loaded is False
    assert b.is_active is False
    assert b.model_identifier is None


def test_backend_property_defaults() -> None:
    b = _fresh_backend()
    assert b.is_vision is False
    assert b.supports_tools is False
    assert b.supports_reasoning is False
    assert b.reasoning_always_on is False
    assert b.reasoning_default is False
    assert b.hf_variant is None
    assert b.context_length is None
    assert b.max_context_length is None
    assert b.native_context_length is None
    assert b.chat_template is None
    assert b.cache_type_kv is None
    assert b.speculative_type is None
    assert b.detect_audio_type() is None
    assert b.load_progress() is None


def test_backend_has_internal_lock() -> None:
    b = _fresh_backend()
    # Lock type-check: acquiring and releasing must work.
    with b._lock:
        pass


def test_backend_generate_raises_when_cold() -> None:
    b = _fresh_backend()
    with pytest.raises(NotImplementedError):
        # Phase 2 of the skeleton raises NotImplementedError; after step 5
        # this becomes a RuntimeError for the cold-backend case. The
        # integration test covers the loaded path.
        gen = b.generate_chat_completion(
            messages = [{"role": "user", "content": "hi"}],
        )
        next(gen)


def test_backend_load_model_raises_in_skeleton() -> None:
    b = _fresh_backend()
    with pytest.raises(NotImplementedError):
        b.load_model(local_path = "/nonexistent", model_identifier = "dummy")
