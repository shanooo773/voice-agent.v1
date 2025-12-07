# Voice of the Patient - STT Module
# Updated to use open-source Whisper via Transformers
# Removed: GROQ API, speech_recognition, pydub
# Uses: sounddevice for recording

import logging
from src.audio_utils import record_audio
from src.stt import transcribe_audio

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def record_patient_audio(file_path, duration=10):
    """
    Record audio from the patient's microphone.
    
    Args:
        file_path (str): Path to save the recorded audio file
        duration (int): Duration to record in seconds
        
    Returns:
        str: Path to saved audio file
    """
    return record_audio(file_path, duration=duration)

def transcribe_patient_audio(audio_filepath, model="openai/whisper-large-v3"):
    """
    Transcribe patient's audio to text using Whisper.
    
    Args:
        audio_filepath (str): Path to audio file
        model (str): Whisper model to use
        
    Returns:
        str: Transcribed text
    """
    return transcribe_audio(audio_filepath, model_name=model)
