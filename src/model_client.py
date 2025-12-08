"""Lightweight HTTP client for the persistent LLM model server."""
from __future__ import annotations

import os
from typing import Optional

import requests

_DEFAULT_BASE_URL = "http://127.0.0.1:8001"
_SESSION = requests.Session()


class ModelServerError(RuntimeError):
    """Raised when the client cannot obtain a valid response from the server."""


def _base_url() -> str:
    return os.getenv("MODEL_SERVER_URL", _DEFAULT_BASE_URL).rstrip("/")


def is_server_available(timeout: float = 1.5) -> bool:
    """Check whether the model server is reachable."""
    try:
        resp = _SESSION.get(f"{_base_url()}/healthz", timeout=timeout)
        return resp.ok
    except requests.RequestException:
        return False


def generate_remote_reply(
    query: str,
    system_prompt: Optional[str] = None,
    model_name: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    timeout: Optional[float] = None,
) -> str:
    """Request a response from the persistent model server."""
    payload: dict[str, object] = {"query": query}
    if system_prompt:
        payload["system_prompt"] = system_prompt
    if model_name:
        payload["model_name"] = model_name
    if max_new_tokens is not None:
        payload["max_new_tokens"] = max_new_tokens

    request_timeout = timeout or float(os.getenv("MODEL_SERVER_TIMEOUT", "120"))

    try:
        response = _SESSION.post(
            f"{_base_url()}/generate",
            json=payload,
            timeout=request_timeout,
        )
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network path
        raise ModelServerError(str(exc)) from exc

    try:
        data = response.json()
    except ValueError as exc:  # pragma: no cover - unexpected JSON
        raise ModelServerError("Server returned invalid JSON") from exc

    reply = data.get("response")
    if not isinstance(reply, str):  # pragma: no cover - validation guard
        raise ModelServerError("Server response missing 'response' field")

    return reply
