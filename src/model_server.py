"""Minimal persistent model server for the OpenSourceLLM, STT, and TTS.

This module exposes a tiny FastAPI app that keeps the heavy models in
memory so UI processes (e.g. the Gradio app) can issue lightweight HTTP
requests. Run with:

    uvicorn src.model_server:app --host 0.0.0.0 --port 8000

Environment variables:
    MODEL_SERVER_MODEL_NAME       Override the LLM model name (default Qwen/Qwen2.5-7B-Instruct)
    MODEL_SERVER_MAX_NEW_TOKENS   Max tokens per response (default 200)
    MODEL_SERVER_PREFER_4BIT      Set to "0" to disable 4-bit quant preference
    MODEL_SERVER_HOST             Host to bind when executed as script (default 0.0.0.0)
    MODEL_SERVER_PORT             Port to bind when executed as script (default 8000)
    WHISPER_MODEL                 Whisper model for STT (default openai/whisper-large-v3)
    DIA2_MODEL                    Dia2 model for TTS (default diacritical/dia2-base)
"""
from __future__ import annotations

import logging
import os
from threading import Lock
from typing import Optional
import tempfile

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.llm import OpenSourceLLM
from src.stt import WhisperSTT
from src.tts import Dia2TTS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_NAME = os.getenv("MODEL_SERVER_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
MAX_NEW_TOKENS = int(os.getenv("MODEL_SERVER_MAX_NEW_TOKENS", "200"))
PREFER_4BIT = os.getenv("MODEL_SERVER_PREFER_4BIT", "1") != "0"
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "openai/whisper-large-v3")
DIA2_MODEL = os.getenv("DIA2_MODEL", "diacritical/dia2-base")

app = FastAPI(title="Voice Agent Model Server", version="1.0.0")

# Initialize models at startup
logger.info("=" * 80)
logger.info("Initializing Voice Agent Model Server")
logger.info("=" * 80)

logger.info(f"Loading LLM: {MODEL_NAME} (prefer_4bit={PREFER_4BIT})")
_llm = OpenSourceLLM(
    model_name=MODEL_NAME,
    max_new_tokens=MAX_NEW_TOKENS,
    prefer_4bit=PREFER_4BIT,
)
logger.info(f"✓ LLM loaded successfully")

logger.info(f"Loading STT: {WHISPER_MODEL}")
_stt = WhisperSTT(model_name=WHISPER_MODEL)
logger.info(f"✓ STT loaded successfully")

logger.info(f"Loading TTS: {DIA2_MODEL}")
_tts = Dia2TTS(model_name=DIA2_MODEL)
logger.info(f"✓ TTS loaded successfully")

logger.info("=" * 80)
logger.info("All models loaded - server ready")
logger.info("=" * 80)

_generation_lock = Lock()
_stt_lock = Lock()
_tts_lock = Lock()


class GenerateRequest(BaseModel):
    query: str
    system_prompt: Optional[str] = None
    model_name: Optional[str] = None
    max_new_tokens: Optional[int] = None


class GenerateResponse(BaseModel):
    response: str


class TranscribeResponse(BaseModel):
    transcription: str


class SynthesizeRequest(BaseModel):
    text: str


class HealthResponse(BaseModel):
    status: str
    models: dict


@app.get("/health", tags=["health"])
def health() -> HealthResponse:
    """
    Health endpoint with detailed model information.
    Returns loading strategy and device for each model.
    """
    import torch
    
    # Get LLM device info
    llm_device = str(next(_llm.model.parameters()).device) if _llm.model else "unknown"
    
    # Determine loading strategy
    loading_strategy = "unknown"
    if hasattr(_llm.model, "quantization_config"):
        loading_strategy = "4-bit quantized (BitsAndBytes)"
    elif llm_device.startswith("cuda"):
        loading_strategy = "fp16 on CUDA"
    else:
        loading_strategy = "CPU (full precision)"
    
    return HealthResponse(
        status="ok",
        models={
            "llm": {
                "name": MODEL_NAME,
                "device": llm_device,
                "loading_strategy": loading_strategy,
                "max_new_tokens": MAX_NEW_TOKENS,
            },
            "stt": {
                "name": WHISPER_MODEL,
                "device": _stt.device,
                "type": "manual" if _stt.use_manual else "pipeline",
            },
            "tts": {
                "name": DIA2_MODEL,
                "device": _tts.device,
                "type": "manual" if _tts.use_manual else "pipeline",
            },
        }
    )


@app.get("/healthz", tags=["health"])
def healthcheck() -> dict[str, str]:
    """Simple health endpoint for backward compatibility."""
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


@app.post("/stt", response_model=TranscribeResponse, tags=["inference"])
async def transcribe_audio(audio: UploadFile = File(...)) -> TranscribeResponse:
    """
    Transcribe audio to text using Whisper.
    
    Args:
        audio: Audio file (WAV, MP3, etc.)
        
    Returns:
        TranscribeResponse with transcription text
    """
    with _stt_lock:
        try:
            # Save uploaded file to temp location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                content = await audio.read()
                tmp_file.write(content)
                tmp_path = tmp_file.name
            
            # Transcribe
            transcription = _stt.transcribe(tmp_path)
            
            # Clean up
            os.unlink(tmp_path)
            
        except Exception as exc:  # pragma: no cover - pass through error to caller
            logger.exception("STT transcription failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    
    return TranscribeResponse(transcription=transcription)


@app.post("/tts", tags=["inference"])
def synthesize_speech(payload: SynthesizeRequest) -> FileResponse:
    """
    Convert text to speech using Dia2 TTS.
    
    Args:
        payload: SynthesizeRequest with text to synthesize
        
    Returns:
        Audio file (WAV format)
    """
    with _tts_lock:
        try:
            # Create temp output file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_path = tmp_file.name
            
            # Synthesize
            output_path = _tts.synthesize_speech(payload.text, tmp_path)
            
        except Exception as exc:  # pragma: no cover - pass through error to caller
            logger.exception("TTS synthesis failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    
    return FileResponse(
        output_path,
        media_type="audio/wav",
        filename="speech.wav"
    )


def run() -> None:
    """Convenience entry-point for python -m src.model_server."""
    import uvicorn

    host = os.getenv("MODEL_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("MODEL_SERVER_PORT", "8000"))
    logger.info("Starting model server on %s:%s", host, port)
    uvicorn.run("src.model_server:app", host=host, port=port, factory=False)


if __name__ == "__main__":  # pragma: no cover - manual execution path
    run()
