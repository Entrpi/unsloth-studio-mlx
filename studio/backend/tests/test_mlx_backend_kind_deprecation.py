# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Chunk F (F2): lock the ``deprecated=True`` contract on the legacy
backend-identity booleans.

The ``is_gguf`` / ``is_mlx`` / ``is_mlx_vlm`` / ``is_mlx_audio`` /
``is_mlx_lora`` fields on ``LoadResponse`` / ``InferenceStatusResponse``
/ ``ModelDetails`` / ``ValidateModelResponse`` are Chunk F deprecations
in favour of ``backend_kind``. The booleans stay populated for one
release, but callers should migrate.

Two contracts under test:

1. The generated JSON schema (OpenAPI passthrough) carries
   ``"deprecated": true`` for every boolean field. This is how API
   consumers discover the deprecation — Pydantic v2's ``deprecated=True``
   lands in the schema output.

2. Reading a deprecated field off an instance raises a
   ``DeprecationWarning``. Existing writers / readers are NOT broken —
   the warning is surfaced once per read, not an error. Tests that
   already construct responses with these booleans still work.
"""

from __future__ import annotations

import os
import sys
import warnings

import pytest

_backend = os.path.join(os.path.dirname(__file__), "..")
if _backend not in sys.path:
    sys.path.insert(0, _backend)

from models.inference import (  # noqa: E402
    InferenceStatusResponse,
    LoadResponse,
    ValidateModelResponse,
)
from models.models import ModelDetails  # noqa: E402


_DEPRECATED_BOOLEAN_FIELDS = (
    "is_gguf",
    "is_mlx",
    "is_mlx_vlm",
    "is_mlx_audio",
)


@pytest.mark.parametrize(
    "cls,fields",
    [
        pytest.param(
            LoadResponse,
            _DEPRECATED_BOOLEAN_FIELDS + ("is_mlx_lora",),
            id = "LoadResponse",
        ),
        pytest.param(
            InferenceStatusResponse,
            _DEPRECATED_BOOLEAN_FIELDS,
            id = "InferenceStatusResponse",
        ),
        pytest.param(
            ValidateModelResponse,
            _DEPRECATED_BOOLEAN_FIELDS,
            id = "ValidateModelResponse",
        ),
        pytest.param(
            ModelDetails,
            _DEPRECATED_BOOLEAN_FIELDS,
            id = "ModelDetails",
        ),
    ],
)
def test_legacy_booleans_marked_deprecated_in_schema(cls, fields):
    """Every legacy boolean field MUST carry ``deprecated: true`` in
    the generated JSON schema so OpenAPI consumers see the migration
    advice. This is how the FastAPI-generated ``/openapi.json`` flags
    the deprecation in the interactive docs."""
    schema = cls.model_json_schema()
    props = schema.get("properties", {})
    for field in fields:
        assert field in props, f"{cls.__name__}: missing field {field}"
        assert props[field].get("deprecated") is True, (
            f"{cls.__name__}.{field}: expected deprecated=True in schema, "
            f"got {props[field].get('deprecated')!r}"
        )


@pytest.mark.parametrize(
    "cls", [LoadResponse, InferenceStatusResponse]
)
def test_backend_kind_not_deprecated(cls):
    """The replacement field ``backend_kind`` must NOT be deprecated —
    it's the new primary source of truth on backend-status-carrying
    responses. Only ``LoadResponse`` and ``InferenceStatusResponse``
    carry the field today; ``ModelDetails`` / ``ValidateModelResponse``
    describe static model shape and don't need a runtime-backend
    discriminator."""
    schema = cls.model_json_schema()
    props = schema.get("properties", {})
    assert "backend_kind" in props, (
        f"{cls.__name__}: backend_kind missing from schema"
    )
    assert props["backend_kind"].get("deprecated") is not True, (
        f"{cls.__name__}.backend_kind: must NOT be deprecated"
    )


def test_reading_deprecated_boolean_warns():
    """Reading a deprecated boolean off an instance surfaces a
    ``DeprecationWarning`` — Pydantic v2's ``deprecated=True`` wires
    this up for us. Existing constructors that SET these fields still
    work without any warning (the write path is untouched)."""
    # Construct with the booleans set — NO warning expected on write.
    with warnings.catch_warnings(record = True) as w:
        warnings.simplefilter("always")
        resp = LoadResponse(
            status = "loaded",
            model = "some/model",
            display_name = "some/model",
            inference = {},
            is_gguf = True,
            backend_kind = "gguf",
        )
        # Construction alone should not warn.
        ctor_depwarns = [
            x for x in w if issubclass(x.category, DeprecationWarning)
        ]
        assert not ctor_depwarns, (
            f"construction should not warn, got: {ctor_depwarns}"
        )

    # Reading the deprecated attr WARNs.
    with warnings.catch_warnings(record = True) as w:
        warnings.simplefilter("always")
        _val = resp.is_gguf
        depwarns = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert depwarns, (
            "expected DeprecationWarning on is_gguf read, got none"
        )

    # Reading backend_kind does NOT warn.
    with warnings.catch_warnings(record = True) as w:
        warnings.simplefilter("always")
        _val = resp.backend_kind
        depwarns = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert not depwarns, (
            f"reading backend_kind should not warn, got: {depwarns}"
        )


def test_backend_kind_enum_populated_on_all_kinds():
    """Sanity-check that every ``BackendKind`` literal is accepted.
    Catches a future refactor that silently drops a variant from the
    Literal union."""
    valid = ("gguf", "mlx", "mlx+lora", "mlx+vlm", "mlx+audio", "unsloth")
    for kind in valid:
        r = LoadResponse(
            status = "loaded",
            model = "m",
            display_name = "m",
            inference = {},
            backend_kind = kind,
        )
        assert r.backend_kind == kind

    # Invalid value rejected.
    with pytest.raises(Exception):
        LoadResponse(
            status = "loaded",
            model = "m",
            display_name = "m",
            inference = {},
            backend_kind = "bogus-backend",  # type: ignore[arg-type]
        )
