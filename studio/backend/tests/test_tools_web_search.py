# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for ``core.inference.tools._web_search`` URL-sentinel
handling.

The GLM-4/4.6 tool-call dialect emits argument values that failed
JSON-decode as raw strings — so a schema-optional ``url`` parameter
that the model represents as ``None`` reaches ``_web_search`` as the
string ``"None"``. Without sentinel handling this truthy string sends
the request into ``_fetch_page_text("None")``, which returns
"Blocked: only http/https URLs are allowed (got '')" and burns an
iteration of the agentic loop on a retry that will emit the same
sentinel again.

These tests pin the behaviour: any string in the sentinel set
(case-insensitive, whitespace-trimmed) routes to query-mode search;
a real URL still routes to fetch-mode.
"""

from __future__ import annotations

import os
import sys

import pytest

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from unittest import mock  # noqa: E402

from core.inference.tools import _URL_FALSY_SENTINELS, _web_search  # noqa: E402


class TestWebSearchUrlSentinels:
    """``url`` values that models emit for "no URL" route to query."""

    @pytest.mark.parametrize(
        "sentinel",
        [
            "None",
            "none",
            "NONE",
            "null",
            "Null",
            "undefined",
            "N/A",
            "n/a",
            "NA",
            "nil",
            "false",
            "False",
            "",
            "   ",  # whitespace-only
            "  None  ",  # surrounding whitespace
        ],
    )
    def test_sentinel_url_falls_through_to_query(self, sentinel):
        with mock.patch(
            "core.inference.tools._fetch_page_text"
        ) as fetch_mock, mock.patch(
            "ddgs.DDGS"
        ) as ddgs_mock:
            ddgs_mock.return_value.text.return_value = [
                {"title": "T", "href": "https://example.com", "body": "Snippet"}
            ]
            out = _web_search("ternary bonsai", url=sentinel)

        # Crucial: _fetch_page_text must NOT be called on a sentinel.
        fetch_mock.assert_not_called()
        # Query-mode search ran: DDGS was constructed and queried.
        ddgs_mock.return_value.text.assert_called_once_with(
            "ternary bonsai", max_results=5
        )
        assert "example.com" in out
        assert "Snippet" in out

    def test_real_url_still_routes_to_fetch(self):
        with mock.patch(
            "core.inference.tools._fetch_page_text",
            return_value="page body markdown",
        ) as fetch_mock:
            out = _web_search("ignored", url="https://example.com/page")
        fetch_mock.assert_called_once()
        assert "page body markdown" in out

    def test_none_value_routes_to_query(self):
        # True ``None`` (not the string) should also route to query
        # — this is the case when the GLM value JSON-decodes as
        # ``null`` or the model omits the key entirely.
        with mock.patch(
            "core.inference.tools._fetch_page_text"
        ) as fetch_mock, mock.patch(
            "ddgs.DDGS"
        ) as ddgs_mock:
            ddgs_mock.return_value.text.return_value = []
            _web_search("q", url=None)
        fetch_mock.assert_not_called()
        ddgs_mock.return_value.text.assert_called_once()

    def test_empty_query_with_sentinel_url_reports_no_query(self):
        # When BOTH the query is empty AND the url is a sentinel,
        # we fall through to the query path and report "No query
        # provided." rather than blocking on the sentinel.
        with mock.patch("core.inference.tools._fetch_page_text") as fetch_mock:
            out = _web_search("", url="None")
        fetch_mock.assert_not_called()
        assert "No query provided" in out

    def test_sentinels_set_includes_expected_values(self):
        # Guardrail: the set should include at least the observed
        # problem values from GLM-4 + a few near-neighbours.
        assert "none" in _URL_FALSY_SENTINELS
        assert "null" in _URL_FALSY_SENTINELS
        assert "undefined" in _URL_FALSY_SENTINELS
        # Values are stored lower-case — matching is done after
        # ``.strip().lower()``.
        assert all(s == s.lower() for s in _URL_FALSY_SENTINELS)

    def test_cancel_event_threads_through_to_fetch(self):
        # Regression guard for the socket-leak bug: cancel_event must
        # reach _fetch_page_text so a hung remote doesn't keep an
        # ESTABLISHED connection open after the agentic loop times
        # out the tool invocation.
        import threading as _th
        evt = _th.Event()
        with mock.patch(
            "core.inference.tools._fetch_page_text",
            return_value="ok",
        ) as fetch_mock:
            _web_search(
                "",
                url="https://example.com/page",
                cancel_event=evt,
            )
        fetch_mock.assert_called_once()
        _, kwargs = fetch_mock.call_args
        assert kwargs.get("cancel_event") is evt


class TestFetchPageTextDeadline:
    """Wall-clock deadline bounds total fetch time even on slow-drip
    servers. Previously the per-socket-op ``timeout=`` only bounded
    individual reads, so a server trickling bytes slower than the
    timeout kept the connection open indefinitely."""

    def test_cancel_event_short_circuits_before_network(self):
        # When cancel_event is already set at entry time, the fetch
        # must bail out with a "cancelled" message rather than making
        # a network call. This matches the backend's tool-cancellation
        # contract (cancel_event checked cooperatively).
        import threading as _th
        from core.inference.tools import _fetch_page_text

        evt = _th.Event()
        evt.set()
        with mock.patch(
            "core.inference.tools._validate_and_resolve_host",
            return_value=(True, "", "93.184.216.34"),
        ):
            with mock.patch(
                "urllib.request.build_opener"
            ) as opener_ctor:
                # Set up a slow opener — if we actually call it, the
                # test fails because cancel_event should short-circuit
                # before we reach here.
                opener_ctor.return_value.open.side_effect = AssertionError(
                    "fetch must not call opener when cancel_event is pre-set"
                )
                out = _fetch_page_text(
                    "https://example.com",
                    timeout=30,
                    cancel_event=evt,
                )
        assert "cancelled" in out.lower()

    def test_deadline_exceeded_message(self):
        # With timeout=0 the deadline is immediate; the function
        # should return the deadline-exceeded message on the first
        # hop check without attempting the network.
        from core.inference.tools import _fetch_page_text
        import threading as _th

        with mock.patch(
            "core.inference.tools._validate_and_resolve_host",
            return_value=(True, "", "93.184.216.34"),
        ), mock.patch(
            "time.monotonic",
            side_effect=[0, 10, 10, 10, 10, 10, 10, 10],
        ):
            out = _fetch_page_text(
                "https://example.com",
                timeout=0,
                cancel_event=_th.Event(),
            )
        assert "timed out" in out.lower()

    def test_closes_response_on_success(self):
        # Socket hygiene: the response object must be closed even on
        # happy path so the connection doesn't linger in CLOSE_WAIT.
        from core.inference.tools import _fetch_page_text
        import threading as _th

        class _FakeResp:
            def __init__(self):
                self.headers = mock.MagicMock()
                self.headers.get_content_charset.return_value = "utf-8"
                self._data = [b"<html>hello</html>", b""]
                self._i = 0
                self.closed = False

            def read(self, n):
                if self._i >= len(self._data):
                    return b""
                chunk = self._data[self._i]
                self._i += 1
                return chunk

            def close(self):
                self.closed = True

        fake = _FakeResp()
        with mock.patch(
            "core.inference.tools._validate_and_resolve_host",
            return_value=(True, "", "93.184.216.34"),
        ), mock.patch(
            "urllib.request.build_opener"
        ) as opener_ctor:
            opener_ctor.return_value.open.return_value = fake
            _fetch_page_text(
                "https://example.com",
                timeout=10,
                cancel_event=_th.Event(),
            )
        assert fake.closed, "response must be explicitly closed"
