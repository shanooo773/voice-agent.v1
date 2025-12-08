"""Minimal persistent model server for the OpenSourceLLM.

This module exposes a tiny FastAPI app that keeps the heavy language model in
memory so UI processes (e.g. the Gradio app) can issue lightweight HTTP
requests. Run with:

    uvicorn src.model_server:app --host 0.0.0.0 --port 8001

Environment variables:
    MODEL_SERVER_MODEL_NAME   Override the model name (default Qwen/Qwen2.5-7B-Instruct)
    MODEL_SERVER_MAX_NEW_TOKENS  Max tokens per response (default 200)
    MODEL_SERVER_PREFER_4BIT   Set to "0" to disable 4-bit quant preference
    MODEL_SERVER_HOST          Host to bind when executed as script (default 0.0.0.0)
    MODEL_SERVER_PORT          Port to bind when executed as script (default 8001)
"""
from __future__ import annotations

import logging
import os
from threading import Lock
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.llm import OpenSourceLLM

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_NAME = os.getenv("MODEL_SERVER_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
MAX_NEW_TOKENS = int(os.getenv("MODEL_SERVER_MAX_NEW_TOKENS", "200"))
PREFER_4BIT = os.getenv("MODEL_SERVER_PREFER_4BIT", "1") != "0"

app = FastAPI(title="Voice Agent LLM Server", version="1.0.0")

logger.info("Initialising OpenSourceLLM for model server (model=%s)", MODEL_NAME)
_llm = OpenSourceLLM(
    model_name=MODEL_NAME,
    max_new_tokens=MAX_NEW_TOKENS,
    prefer_4bit=PREFER_4BIT,
)
_generation_lock = Lock()


class GenerateRequest(BaseModel):
    query: str
    system_prompt: Optional[str] = None
    model_name: Optional[str] = None
    max_new_tokens: Optional[int] = None


class GenerateResponse(BaseModel):
    response: str


@app.get("/healthz", tags=["health"])
def healthcheck() -> dict[str, str]:
    """Simple health endpoint so callers can verify the server is alive."""
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/generate", response_model=GenerateResponse, tags=["inference"])
def generate_text(payload: GenerateRequest) -> GenerateResponse:
    """Synchronous text generation endpoint with a simple lock for safety."""
    if payload.model_name and payload.model_name != MODEL_NAME:
        raise HTTPException(status_code=400, detail="Model server was started with a different model.")

    effective_max_tokens = payload.max_new_tokens or MAX_NEW_TOKENS

    with _generation_lock:
        try:
            reply = _llm.generate_reply(
                text=payload.query,
                system_prompt=payload.system_prompt,
                max_new_tokens=effective_max_tokens,
            )
        except Exception as exc:  # pragma: no cover - pass through error to caller
            logger.exception("Model generation failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return GenerateResponse(response=reply)


def run() -> None:
    """Convenience entry-point for python -m src.model_server."""
    import uvicorn

    host = os.getenv("MODEL_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("MODEL_SERVER_PORT", "8001"))
    logger.info("Starting model server on %s:%s", host, port)
    uvicorn.run("src.model_server:app", host=host, port=port, factory=False)


if __name__ == "__main__":  # pragma: no cover - manual execution path
    run()
