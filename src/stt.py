"""
Speech-to-Text module using Whisper from Hugging Face Transformers
"""
import logging
from transformers import pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WhisperSTT:
    def __init__(self, model_name="openai/whisper-large-v3"):
        """
        Initialize Whisper STT model
        
        Args:
            model_name (str): Hugging Face model name for Whisper
        """
        logging.info(f"Loading Whisper STT model: {model_name}")
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=-1  # Use CPU, change to 0 for GPU
        )
        logging.info("Whisper STT model loaded successfully")
    
    def transcribe_audio(self, file_path):
        """
        Transcribe audio file to text
        
        Args:
            file_path (str): Path to the audio file
            
        Returns:
            str: Transcribed text
        """
        logging.info(f"Transcribing audio file: {file_path}")
        try:
            result = self.pipe(file_path)
            transcription = result["text"]
            logging.info(f"Transcription complete: {transcription}")
            return transcription
        except Exception as e:
            logging.error(f"Error during transcription: {e}")
            raise
