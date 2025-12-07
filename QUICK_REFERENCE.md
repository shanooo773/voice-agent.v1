# Quick Reference Guide

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run with defaults
python main.py
```

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--stt-model` | `openai/whisper-large-v3` | Whisper model for STT |
| `--llm-model` | `Qwen/Qwen2.5-7B-Instruct` | LLM model |
| `--tts-model` | `facebook/mms-tts-eng` | TTS model |
| `--record-duration` | `5` | Recording duration in seconds |
| `--async` | `False` | Use async mode |
| `--system-prompt` | `None` | System prompt for LLM |

## Available Models

### STT (Speech-to-Text)
- `openai/whisper-large-v3` (default, most accurate)
- `openai/whisper-medium`
- `openai/whisper-small`
- `openai/whisper-base` (fastest)

### LLM (Language Model)
- `Qwen/Qwen2.5-7B-Instruct` (default, balanced)
- `meta-llama/Llama-3-8b-instruct` (requires auth)
- `sahil2801/gpt-oss` (smaller, faster)

### TTS (Text-to-Speech)
- `facebook/mms-tts-eng` (English, default)
- `facebook/mms-tts-urd` (Urdu)
- Other MMS-TTS models available on Hugging Face

## Common Use Cases

### Medical Assistant
```bash
python main.py \
  --system-prompt "You are a medical assistant. Provide helpful advice." \
  --record-duration 10
```

### Quick Q&A
```bash
python main.py \
  --llm-model sahil2801/gpt-oss \
  --record-duration 3
```

### Multilingual (Urdu)
```bash
python main.py \
  --tts-model facebook/mms-tts-urd
```

## Programmatic API

```python
from src.agent import VoiceAgent

# Initialize
agent = VoiceAgent(
    stt_model="openai/whisper-large-v3",
    llm_model="Qwen/Qwen2.5-7B-Instruct",
    tts_model="facebook/mms-tts-eng"
)

# Process audio file
transcription, response, audio = agent.process_audio("input.wav")

# Interactive loop
agent.run_voice_loop(record_duration=5)
```

## Module Structure

- `src/stt.py` - Speech-to-Text (Whisper)
- `src/llm.py` - Language Model
- `src/tts.py` - Text-to-Speech (MMS-TTS)
- `src/audio_utils.py` - Recording & Playback
- `src/agent.py` - Voice Agent Orchestration
- `main.py` - CLI Entry Point

## No API Keys Required

This is a 100% open-source solution. All models run locally:
- ✅ No OpenAI API key
- ✅ No ElevenLabs API key
- ✅ No Groq/Grok API key
- ✅ No internet required after models are downloaded

## Performance Tips

1. **Use GPU**: Modify `device=-1` to `device=0` in source files
2. **Smaller models**: Use `whisper-base` and `gpt-oss` for faster inference
3. **Reduce record duration**: Use `--record-duration 3` for quick responses
4. **Cache models**: Models are cached after first download

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No microphone detected | Check PortAudio installation |
| Model download slow | Check internet connection |
| Out of memory | Use smaller models or add more RAM |
| Audio playback fails | Check audio drivers and OS compatibility |
