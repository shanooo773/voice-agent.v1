# Architecture Comparison

## Before (Paid APIs + Image Processing)

```
┌─────────────────────────────────────────────────────────────┐
│                      Gradio UI                               │
│         (Audio Input + Image Input)                          │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
┌───────────────┐              ┌────────────────┐
│  Audio File   │              │  Image File    │
└───────┬───────┘              └────────┬───────┘
        │                               │
        ▼                               ▼
┌───────────────────┐          ┌─────────────────────┐
│   GROQ API STT    │          │  base64 encoding    │
│ (whisper-large-v3)│          │  (PIL/Pillow)       │
└───────┬───────────┘          └──────────┬──────────┘
        │                                 │
        ▼                                 ▼
    [Text]                    ┌─────────────────────┐
        │                     │  GROQ API Vision    │
        │                     │ (LLaMA-4-Scout)     │
        │                     └──────────┬──────────┘
        │                                │
        └───────────┬────────────────────┘
                    ▼
            [Combined Prompt]
                    │
                    ▼
          ┌─────────────────┐
          │  ElevenLabs TTS │
          │    (Aria voice) │
          └────────┬────────┘
                   ▼
              [Audio Output]
```

**Limitations:**
- ❌ Required API keys and internet
- ❌ Paid services ($$$)
- ❌ Privacy concerns (data sent to third parties)
- ❌ Image processing dependency
- ❌ Complex audio pipeline (pyaudio, pydub, speech_recognition)

---

## After (Open Source + Voice Only)

```
┌─────────────────────────────────────────────────────────────┐
│                      Gradio UI                               │
│              (Voice-Only Input)                              │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
                ┌───────────────┐
                │  Audio File   │
                └───────┬───────┘
                        │
        ┌───────────────┴───────────────┐
        │       src/audio_utils.py      │
        │      (sounddevice/soundfile)  │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │         src/stt.py            │
        │  (Whisper via Transformers)   │
        │   openai/whisper-large-v3     │
        └───────────────┬───────────────┘
                        │
                        ▼
                    [Text]
                        │
                        ▼
        ┌───────────────────────────────┐
        │         src/llm.py            │
        │   (LLM via Transformers)      │
        │   Qwen/Qwen2.5-7B-Instruct    │
        └───────────────┬───────────────┘
                        │
                        ▼
                [Response Text]
                        │
                        ▼
        ┌───────────────────────────────┐
        │         src/tts.py            │
        │   (Dia2 via Transformers)     │
        │    diacritical/dia2-base      │
        └───────────────┬───────────────┘
                        │
                        ▼
                  [Audio Output]
                        │
        ┌───────────────┴───────────────┐
        │       src/audio_utils.py      │
        │         (playback)            │
        └───────────────────────────────┘
```

**Benefits:**
- ✅ 100% open source
- ✅ No API keys needed
- ✅ Works offline (after initial download)
- ✅ Privacy-preserving (all local)
- ✅ No costs
- ✅ Modular and customizable
- ✅ Simpler audio pipeline

---

## Component Mapping

| Component | Before | After |
|-----------|--------|-------|
| **STT** | GROQ API (Whisper) | Local Whisper via Transformers |
| **LLM** | GROQ API (LLaMA-Vision) | Local Qwen/Llama via Transformers |
| **TTS** | ElevenLabs/gTTS | Local Dia2 via Transformers |
| **Audio I/O** | speech_recognition + pydub + pyaudio + subprocess | sounddevice + soundfile |
| **Image** | PIL + base64 encoding | ❌ Removed |
| **Vision** | LLaMA-4-Scout (GROQ) | ❌ Removed |
| **API Keys** | GROQ_API_KEY, ELEVEN_API_KEY | ❌ Not needed |
| **Privacy** | Data sent to APIs | All local processing |
| **Cost** | Pay per API call | Free |
| **Internet** | Required | Optional (after initial download) |

---

## File Structure Comparison

### Before
```
voice-agent.v1/
├── brain_of_the_doctor.py    (GROQ Vision API)
├── voice_of_the_patient.py   (GROQ STT + speech_recognition)
├── voice_of_the_doctor.py    (ElevenLabs/gTTS TTS)
├── gradio_app.py             (Audio + Image UI)
├── requirements.txt          (groq, elevenlabs, gtts, pillow, etc.)
└── Pipfile
```

### After
```
voice-agent.v1/
├── src/
│   ├── __init__.py           (Package exports)
│   ├── stt.py               (Whisper STT)
│   ├── llm.py               (Open-source LLM)
│   ├── tts.py               (Dia2 TTS)
│   ├── audio_utils.py       (sounddevice I/O)
│   └── agent.py             (Pipeline orchestration)
├── brain_of_the_doctor.py    (Updated: uses src/llm.py)
├── voice_of_the_patient.py   (Updated: uses src/stt.py)
├── voice_of_the_doctor.py    (Updated: uses src/tts.py)
├── gradio_app.py             (Updated: voice-only, uses src/*)
├── requirements.txt          (transformers, torch, sounddevice)
├── Pipfile                   (same as requirements.txt)
├── .gitignore               (Python artifacts)
├── README.md                (Updated documentation)
└── MIGRATION_SUMMARY.md     (This guide)
```
