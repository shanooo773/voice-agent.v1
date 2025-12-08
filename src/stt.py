"""
Speech-to-Text module using OpenAI Whisper via Transformers pipeline.
Replaces GROQ API STT.
"""

from transformers import pipeline
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WhisperSTT:
    def __init__(self, model_name="openai/whisper-large-v3"):
        """
        Initialize Whisper STT pipeline.
        
        Args:
            model_name (str): Hugging Face model identifier for Whisper
        """
        logging.info(f"Loading Whisper STT model: {model_name}")
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=0  # Use CPU by default, change to 0 for GPU
        )
        logging.info("Whisper STT model loaded successfully")
    
    def transcribe(self, audio_filepath):
        """
        Transcribe audio file to text.
        
        Args:
            audio_filepath (str): Path to audio file
            
        Returns:
            str: Transcribed text
        """
        logging.info(f"Transcribing audio file: {audio_filepath}")
        result = self.pipe(audio_filepath)
        transcription = result["text"]
        logging.info(f"Transcription complete: {transcription[:50]}...")
        return transcription


def transcribe_audio(audio_filepath, model_name="openai/whisper-large-v3"):
    """
    Convenience function to transcribe audio without maintaining state.
    
    Args:
        audio_filepath (str): Path to audio file
        model_name (str): Whisper model to use
        
    Returns:
        str: Transcribed text
    """
    stt = WhisperSTT(model_name=model_name)
    return stt.transcribe(audio_filepath)
