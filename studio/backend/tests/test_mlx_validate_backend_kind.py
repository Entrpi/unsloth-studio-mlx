# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Chunk G (G2): lock the ``backend_kind`` contract on
``ValidateModelResponse``.

Symmetric with the Chunk F deprecation suite: the ``/validate`` endpoint
now returns a ``backend_kind`` enum alongside the deprecated is_gguf /
is_mlx / is_mlx_vlm / is_mlx_audio booleans so a future removal of the
booleans doesn't make the validate response lossy.

Three contracts under test:

1. Constructing ``ValidateModelResponse`` with only the legacy booleans
   (no ``backend_kind``) still succeeds — the compat path is preserved.

2. Constructing with ``backend_kind="gguf"`` round-trips through
   ``model_dump()`` carrying both the enum AND the deprecated boolean.

3. The ``validate_model`` route populates ``backend_kind`` deterministically
   from the resolved ``ModelConfig`` — every ModelConfig shape that the
   route might see maps to the correct enum.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

_backend = os.path.join(os.path.dirname(__file__), "..")
if _backend not in sys.path:
    sys.path.insert(0, _backend)

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from models.inference import ValidateModelResponse  # noqa: E402


# ── Pydantic-level schema / compat contracts ──────────────────────


def test_validate_response_compat_no_backend_kind():
    """Old callers that set only the legacy ``is_gguf`` boolean (no
    ``backend_kind``) must keep working. Chunk G does not break the
    compat path."""
    resp = ValidateModelResponse(
        valid = True,
        message = "ok",
        is_gguf = True,
    )
    assert resp.valid is True
    assert resp.backend_kind is None
    # Reading the deprecated boolean still works (with a deprecation
    # warning — that's the Chunk F contract, unchanged here).
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert resp.is_gguf is True


def test_validate_response_dump_carries_both_enum_and_boolean():
    """Constructing with ``backend_kind`` populated round-trips through
    ``model_dump()`` carrying the enum AND the deprecated mirror boolean
    — the wire format keeps both for the duration of the deprecation
    window."""
    resp = ValidateModelResponse(
        valid = True,
        message = "ok",
        backend_kind = "gguf",
        is_gguf = True,
    )
    dumped = resp.model_dump()
    assert dumped["backend_kind"] == "gguf"
    assert dumped["is_gguf"] is True


def test_validate_response_backend_kind_schema_not_deprecated():
    """``backend_kind`` is the replacement field; it must NOT be marked
    deprecated in the generated JSON schema even though it sits
    alongside deprecated peers."""
    schema = ValidateModelResponse.model_json_schema()
    props = schema.get("properties", {})
    assert "backend_kind" in props, "backend_kind missing from schema"
    assert props["backend_kind"].get("deprecated") is not True


@pytest.mark.parametrize(
    "kind", ["gguf", "mlx", "mlx+lora", "mlx+vlm", "mlx+audio", "unsloth"]
)
def test_validate_response_accepts_every_backend_kind(kind):
    """Every ``BackendKind`` literal is accepted — catches a future
    refactor that drops a variant from the union only on this schema."""
    resp = ValidateModelResponse(
        valid = True,
        message = "ok",
        backend_kind = kind,
    )
    assert resp.backend_kind == kind


def test_validate_response_rejects_invalid_backend_kind():
    with pytest.raises(Exception):
        ValidateModelResponse(
            valid = True,
            message = "ok",
            backend_kind = "not-a-backend",  # type: ignore[arg-type]
        )


# ── Route-level contract (real /validate call site) ────────────────


@pytest.fixture
def validate_app():
    """Minimal FastAPI app mounting just the inference router, with
    auth overridden and no backend lifecycle hooks."""
    from routes import inference as inf_mod

    app = FastAPI()
    app.include_router(inf_mod.router, prefix = "/inference")

    async def _no_auth():
        return "test@local"

    app.dependency_overrides[inf_mod.get_current_subject] = _no_auth
    return app


def _fake_config(**overrides) -> SimpleNamespace:
    """Build a stub ``ModelConfig`` that ``validate_model`` can introspect
    via ``getattr``. Only the attributes the route touches need to be
    set — everything else falls through the ``getattr(..., default)``
    path in the route."""
    base = dict(
        identifier = "stub/model",
        display_name = "stub/model",
        is_gguf = False,
        is_mlx = False,
        is_mlx_lora = False,
        is_mlx_vlm = False,
        is_mlx_audio = False,
        is_lora = False,
        is_vision = False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.parametrize(
    "overrides,expected_kind",
    [
        pytest.param({"is_gguf": True}, "gguf", id = "gguf"),
        pytest.param({"is_mlx": True}, "mlx", id = "mlx"),
        pytest.param(
            {"is_mlx": True, "is_mlx_lora": True, "is_lora": True},
            "mlx+lora",
            id = "mlx+lora",
        ),
        pytest.param({"is_mlx_vlm": True}, "mlx+vlm", id = "mlx+vlm"),
        pytest.param({"is_mlx_audio": True}, "mlx+audio", id = "mlx+audio"),
        pytest.param({}, "unsloth", id = "unsloth-fallback"),
    ],
)
def test_validate_route_populates_backend_kind(
    validate_app, monkeypatch, overrides, expected_kind
):
    """End-to-end: a POST to ``/inference/validate`` returns a
    ``backend_kind`` derived from the resolved ``ModelConfig``. Every
    ModelConfig shape the route can see maps to the right enum."""
    from routes import inference as inf_mod

    cfg = _fake_config(**overrides)
    monkeypatch.setattr(
        inf_mod.ModelConfig,
        "from_identifier",
        classmethod(lambda cls, **kwargs: cfg),
    )
    # load_inference_config is called for the requires_trust_remote_code
    # side-channel; return an empty dict so the bool() falls to False.
    monkeypatch.setattr(
        inf_mod,
        "load_inference_config",
        lambda _identifier: {},
    )

    client = TestClient(validate_app)
    resp = client.post(
        "/inference/validate",
        json = {"model_path": "stub/model"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["backend_kind"] == expected_kind
    # The legacy booleans still mirror the config deterministically.
    if expected_kind == "gguf":
        assert body["is_gguf"] is True
    elif expected_kind in ("mlx", "mlx+lora"):
        assert body["is_mlx"] is True
    elif expected_kind == "mlx+vlm":
        assert body["is_mlx_vlm"] is True
    elif expected_kind == "mlx+audio":
        assert body["is_mlx_audio"] is True
