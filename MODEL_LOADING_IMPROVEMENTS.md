# Model Loading Improvements - Documentation

## Overview

This update makes model loading production-ready with robust fallback strategies, offline mode support, and persistent model serving.

## Key Improvements

### 1. **Robust Model Loading with Fallback Strategy**

All models now follow a comprehensive fallback strategy:

**LLM (src/llm.py):**
- 4-bit quantization (BitsAndBytes) → 8-bit → fp16 on CUDA → CPU
- Detailed logging at each step
- Device map with automatic offloading
- Low memory usage optimization

**STT (src/stt.py) & TTS (src/tts.py):**
- Manual model loading with processor/tokenizer
- Pipeline fallback if manual loading fails
- Device assignment (CUDA/CPU)
- Offline mode support

### 2. **Offline Mode Detection**

Models check for offline mode via:
- `transformers.utils.is_offline_mode()`
- Environment variables: `HF_OFFLINE`, `HF_FORCE_OFFLINE`, `TRANSFORMERS_OFFLINE`
- Fast failure with clear error messages when offline and model not cached

### 3. **Model Existence Validation**

New `ensure_model_exists()` helper in all model modules:
- Checks if path is local directory
- Validates HuggingFace Hub availability (when online)
- Falls back to `LOCAL_MODELS_DIR` if model not found
- Clear error messages for troubleshooting

### 4. **Configuration Constants**

All modules support these environment variables:
- `LOCAL_MODELS_DIR`: Local model storage (default: `./models`)
- `OFFLOAD_FOLDER`: CPU offloading folder (default: `/tmp/transformers_offload`)
- `HF_HOME`: HuggingFace cache directory (default: `/opt/hf_cache`)

### 5. **BitsAndBytes Safety Checks**

- Detects if native CUDA libraries are present
- Logs available `.so` files for diagnostics
- Provides actionable error messages
- Graceful fallback to lower quantization or full precision

### 6. **Deprecated TRANSFORMERS_CACHE Warning**

When `TRANSFORMERS_CACHE` is set without `HF_HOME`, a warning is logged recommending the use of `HF_HOME`.

### 7. **Persistent Model Server**

Enhanced `model_server.py`:
- Loads all models (LLM, STT, TTS) once at startup
- Exposes REST API endpoints:
  - `GET /health` - Detailed health check with model info
  - `POST /generate` - Text generation
  - `POST /stt` - Speech-to-text transcription
  - `POST /tts` - Text-to-speech synthesis
- Thread-safe with locks
- Comprehensive startup logging

### 8. **Model Client Functions**

Enhanced `model_client.py`:
- `generate_remote_reply()` - LLM generation
- `transcribe()` - STT transcription
- `synthesize()` - TTS synthesis
- Error handling and timeout support

## Usage

### Environment Setup

```bash
# Recommended environment variables
export HF_HOME=/opt/hf_cache
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Optional: Disable 4-bit quantization
export MODEL_SERVER_PREFER_4BIT=0

# Optional: Custom model paths
export LOCAL_MODELS_DIR=./models
export OFFLOAD_FOLDER=/tmp/transformers_offload
```

### Starting the Model Server

```bash
# Start in background
tmux new -s modelserver -d "bash -lc 'cd /path/to/voice-agent.v1; pipenv run python -m src.model_server'"

# Or directly with uvicorn
pipenv run uvicorn src.model_server:app --host 0.0.0.0 --port 8000

# Check health
curl -sS http://127.0.0.1:8000/health | jq
```

### Testing Model Loading

```bash
# Quick LLM test
pipenv run python -c "from src.llm import OpenSourceLLM; m=OpenSourceLLM(); print('OK', m.generate_reply('Hello'))"

# Or use the test script
pipenv run python test_model_loading.py

# Or use the quick shell script
./test_llm_quick.sh
```

### Offline Testing

```bash
# Simulate offline mode with empty cache
HF_HOME=/tmp/empty_hf HF_FORCE_OFFLINE=1 pipenv run python -c "from src.llm import OpenSourceLLM; OpenSourceLLM()"

# Expected output: Fast failure with clear offline error message
```

### Using the Model Client

```python
from src.model_client import generate_remote_reply, transcribe, synthesize, is_server_available

# Check if server is running
if is_server_available():
    # Generate text
    response = generate_remote_reply("Hello, how are you?", system_prompt="You are a helpful assistant")
    
    # Transcribe audio
    text = transcribe("path/to/audio.wav")
    
    # Synthesize speech
    audio_path = synthesize("Hello world", output_path="output.wav")
else:
    print("Model server not available")
```

## API Endpoints

### GET /health

Returns detailed model information including device and loading strategy.

```bash
curl http://127.0.0.1:8000/health
```

Response:
```json
{
  "status": "ok",
  "models": {
    "llm": {
      "name": "Qwen/Qwen2.5-7B-Instruct",
      "device": "cuda:0",
      "loading_strategy": "4-bit quantized (BitsAndBytes)",
      "max_new_tokens": 200
    },
    "stt": {
      "name": "openai/whisper-large-v3",
      "device": "cpu",
      "type": "manual"
    },
    "tts": {
      "name": "diacritical/dia2-base",
      "device": "cuda",
      "type": "manual"
    }
  }
}
```

### POST /generate

Generate text with the LLM.

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello", "system_prompt": "You are a helpful assistant"}'
```

### POST /stt

Transcribe audio to text.

```bash
curl -X POST http://127.0.0.1:8000/stt \
  -F "audio=@path/to/audio.wav"
```

### POST /tts

Convert text to speech.

```bash
curl -X POST http://127.0.0.1:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}' \
  --output speech.wav
```

## Logging

All modules provide comprehensive logging:

- **INFO**: Model loading, device selection, fallback choices
- **WARNING**: Fallback events, deprecated settings
- **ERROR**: Loading failures with troubleshooting steps

Example log output:
```
2024-12-08 23:10:00 - INFO - Loading LLM model: Qwen/Qwen2.5-7B-Instruct (device=cuda)
2024-12-08 23:10:00 - INFO - BitsAndBytes native libraries found: ['libbitsandbytes_cuda118.so']
2024-12-08 23:10:05 - INFO - Attempting 4-bit BitsAndBytes load for Qwen/Qwen2.5-7B-Instruct
2024-12-08 23:10:05 - INFO - Using device_map='auto' with offload_folder=/tmp/transformers_offload
2024-12-08 23:10:30 - INFO - ✓ Loaded model in 4-bit mode (BitsAndBytes quantization)
2024-12-08 23:10:30 - INFO - Model device: cuda:0
```

## Troubleshooting

### BitsAndBytes Issues

If you see warnings about missing native libraries:
```bash
pip install --force-reinstall bitsandbytes
```

### CUDA Version Mismatch

Check your CUDA version matches PyTorch:
```python
import torch
print(f"PyTorch CUDA version: {torch.version.cuda}")
```

### Memory Issues

Enable offloading and use smaller models:
```bash
export OFFLOAD_FOLDER=/tmp/transformers_offload
export MODEL_SERVER_MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
```

### Offline Mode

Pre-download models to cache:
```bash
python -c "from transformers import AutoModel; AutoModel.from_pretrained('model-name')"
```

## File Changes Summary

- **src/llm.py**: Enhanced with ensure_model_exists(), offline mode, quantization fallbacks, BNB diagnostics
- **src/tts.py**: Manual model loading, use_global=True attempt, offline handling
- **src/stt.py**: Manual model loading, offline handling
- **src/model_server.py**: Added STT/TTS endpoints, enhanced health endpoint, detailed logging
- **src/model_client.py**: Added transcribe() and synthesize() functions
- **test_model_loading.py**: Comprehensive test script
- **test_llm_quick.sh**: Quick one-liner test

## Migration from Old Code

No changes to business logic or user-facing outputs. The updates are backward compatible:

- `OpenSourceLLM()` class still works the same way
- `WhisperSTT()` and `Dia2TTS()` have the same public API
- Model server endpoints are enhanced but backward compatible
- Gradio app works without modifications
