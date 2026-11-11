"""Shared HTTP client factory (tests inject an ``httpx.MockTransport``)."""
from __future__ import annotations

from typing import Optional

import httpx

_TRANSPORT: Optional[httpx.BaseTransport] = None


def set_transport(transport: Optional[httpx.BaseTransport]) -> None:
    """Route every world-model HTTP call through ``transport`` (``None`` resets)."""
    global _TRANSPORT
    _TRANSPORT = transport


def client(timeout: float, headers: Optional[dict] = None) -> httpx.Client:
    kw = {"timeout": httpx.Timeout(timeout, connect=min(10.0, timeout)), "headers": headers or {}}
    if _TRANSPORT is not None:
        kw["transport"] = _TRANSPORT
    return httpx.Client(**kw)


class BackendError(RuntimeError):
    """A backend call failed (connection, HTTP status or unusable payload)."""

    def __init__(self, msg: str, status: Optional[int] = None, retryable: bool = False):
        super().__init__(msg)
        self.status = status
        self.retryable = retryable


def raise_for(r: httpx.Response, what: str) -> None:
    if r.status_code >= 400:
        body = r.text[:800] if r.content else ""
        raise BackendError(f"{what}: HTTP {r.status_code} {body}", status=r.status_code,
                           retryable=r.status_code in (429, 500, 502, 503, 504))
