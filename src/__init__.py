"""
Voice Agent - Open Source Implementation
STT: Whisper via Transformers
LLM: Open-source models (Qwen, Llama, etc.)
TTS: Dia2 via Transformers
"""

from src.stt import WhisperSTT, transcribe_audio
from src.llm import OpenSourceLLM, generate_text_reply
from src.tts import Dia2TTS, text_to_speech
from src.audio_utils import AudioRecorder, record_audio, play_audio
from src.agent import VoiceAgent, process_audio_async

__all__ = [
    'WhisperSTT',
    'transcribe_audio',
    'OpenSourceLLM',
    'generate_text_reply',
    'Dia2TTS',
    'text_to_speech',
    'AudioRecorder',
    'record_audio',
    'play_audio',
    'VoiceAgent',
    'process_audio_async',
]
