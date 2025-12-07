# Voice of the Doctor - TTS Module
# Updated to use open-source Dia2 via Transformers
# Removed: ElevenLabs API, gTTS
# Uses: Dia2 TTS and sounddevice for playback

from src.tts import text_to_speech
from src.audio_utils import play_audio

def generate_doctor_voice(input_text, output_filepath, model="diacritical/dia2-base"):
    """
    Convert doctor's text response to speech using Dia2 TTS.
    
    Args:
        input_text (str): Text to convert to speech
        output_filepath (str): Path to save audio file (will be saved as WAV)
        model (str): Dia2 model to use
            - "diacritical/dia2-base" for English
            - "diacritical/dia2-multilingual" for Urdu and other languages
            
    Returns:
        str: Path to saved audio file
    """
    return text_to_speech(input_text, output_filepath, model_name=model)

def generate_and_play_doctor_voice(input_text, output_filepath, model="diacritical/dia2-base"):
    """
    Convert doctor's text response to speech and play it.
    
    Args:
        input_text (str): Text to convert to speech
        output_filepath (str): Path to save audio file (will be saved as WAV)
        model (str): Dia2 model to use
            
    Returns:
        str: Path to saved audio file
    """
    audio_path = generate_doctor_voice(input_text, output_filepath, model)
    play_audio(audio_path)
    return audio_path