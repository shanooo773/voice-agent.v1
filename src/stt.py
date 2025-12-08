"""
Speech-to-Text module using OpenAI Whisper via Transformers pipeline.
Replaces GROQ API STT.
"""

from transformers import pipeline
import soundfile as sf
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class WhisperSTT:
    def __init__(self, model_name="openai/whisper-large-v3"):
        """Initialise the Whisper ASR pipeline."""

        logging.info(f"Loading Whisper STT model: {model_name}")
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=-1,  # Force CPU to avoid CUDA kernel issues on unsupported GPUs
        )
        logging.info("Whisper STT model loaded successfully")

    def transcribe(self, audio_filepath):
        """Transcribe an audio file to text using Whisper."""

        logging.info(f"Transcribing audio file: {audio_filepath}")

        audio_data, sample_rate = sf.read(audio_filepath)
        if audio_data.dtype != "float32":
            audio_data = audio_data.astype("float32")

        result = self.pipe({"array": audio_data, "sampling_rate": sample_rate})
        transcription = result["text"]
        logging.info(f"Transcription complete: {transcription[:50]}...")
        return transcription


def transcribe_audio(audio_filepath, model_name="openai/whisper-large-v3"):
    """Convenience helper that instantiates WhisperSTT and transcribes input."""

    stt = WhisperSTT(model_name=model_name)
    return stt.transcribe(audio_filepath)
