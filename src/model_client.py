"""Lightweight HTTP client for the persistent model server."""
from __future__ import annotations

import os
from typing import Optional
from pathlib import Path

import requests

_DEFAULT_BASE_URL = "http://127.0.0.1:8000"
_SESSION = requests.Session()


class ModelServerError(RuntimeError):
    """Raised when the client cannot obtain a valid response from the server."""


def _base_url() -> str:
    return os.getenv("MODEL_SERVER_URL", _DEFAULT_BASE_URL).rstrip("/")


def is_server_available(timeout: float = 1.5) -> bool:
    """Check whether the model server is reachable."""
    try:
        resp = _SESSION.get(f"{_base_url()}/health", timeout=timeout)
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


def transcribe(audio_path: str, timeout: Optional[float] = None) -> str:
    """
    Transcribe audio file using the persistent model server.
    
    Args:
        audio_path: Path to audio file
        timeout: Request timeout in seconds
        
    Returns:
        Transcribed text
        
    Raises:
        ModelServerError: If server request fails
    """
    request_timeout = timeout or float(os.getenv("MODEL_SERVER_TIMEOUT", "120"))
    
    try:
        with open(audio_path, "rb") as audio_file:
            files = {"audio": audio_file}
            response = _SESSION.post(
                f"{_base_url()}/stt",
                files=files,
                timeout=request_timeout,
            )
            response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network path
        raise ModelServerError(str(exc)) from exc
    except FileNotFoundError as exc:  # pragma: no cover - file not found
        raise ModelServerError(f"Audio file not found: {audio_path}") from exc
    
    try:
        data = response.json()
    except ValueError as exc:  # pragma: no cover - unexpected JSON
        raise ModelServerError("Server returned invalid JSON") from exc
    
    transcription = data.get("transcription")
    if not isinstance(transcription, str):  # pragma: no cover - validation guard
        raise ModelServerError("Server response missing 'transcription' field")
    
    return transcription


def synthesize(text: str, output_path: Optional[str] = None, timeout: Optional[float] = None) -> str:
    """
    Synthesize speech from text using the persistent model server.
    
    Args:
        text: Text to synthesize
        output_path: Optional path to save audio file (defaults to temp file)
        timeout: Request timeout in seconds
        
    Returns:
        Path to saved audio file
        
    Raises:
        ModelServerError: If server request fails
    """
    request_timeout = timeout or float(os.getenv("MODEL_SERVER_TIMEOUT", "120"))
    
    payload = {"text": text}
    
    try:
        response = _SESSION.post(
            f"{_base_url()}/tts",
            json=payload,
            timeout=request_timeout,
        )
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network path
        raise ModelServerError(str(exc)) from exc
    
    # Determine output path
    if output_path is None:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            output_path = tmp_file.name
    
    # Save audio content
    with open(output_path, "wb") as f:
        f.write(response.content)
    
    return output_path
