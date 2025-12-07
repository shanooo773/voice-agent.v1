"""
Text-to-Speech module using Dia2 via Transformers pipeline.
Replaces ElevenLabs and gTTS.
"""

from transformers import pipeline
import logging
import numpy as np
import scipy.io.wavfile as wavfile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Dia2TTS:
    def __init__(self, model_name="diacritical/dia2-base", language="en"):
        """
        Initialize Dia2 TTS pipeline.
        
        Args:
            model_name (str): Hugging Face model identifier
                Options:
                - "diacritical/dia2-base" (English)
                - "diacritical/dia2-multilingual" (for Urdu and other languages)
            language (str): Language code (e.g., 'en' for English, 'ur' for Urdu)
        """
        logging.info(f"Loading Dia2 TTS model: {model_name}")
        self.pipe = pipeline(
            "text-to-speech",
            model=model_name,
            device=-1  # Use CPU by default, change to 0 for GPU
        )
        self.language = language
        logging.info("Dia2 TTS model loaded successfully")
    
    def synthesize_speech(self, text, output_filepath):
        """
        Synthesize speech from text and save to file.
        
        Args:
            text (str): Text to synthesize
            output_filepath (str): Path to save audio file (WAV format)
            
        Returns:
            str: Path to saved audio file
        """
        logging.info(f"Synthesizing speech for text: {text[:50]}...")
        
        # Generate audio
        result = self.pipe(text)
        
        # Extract audio data and sampling rate
        audio = result["audio"]
        sampling_rate = result["sampling_rate"]
        
        # Convert to numpy array if needed
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio)
        
        # Ensure the output file ends with .wav
        if not output_filepath.endswith('.wav'):
            output_filepath = output_filepath.rsplit('.', 1)[0] + '.wav'
        
        # Save as WAV file
        wavfile.write(output_filepath, sampling_rate, audio)
        
        logging.info(f"Audio saved to: {output_filepath}")
        return output_filepath


def text_to_speech(text, output_filepath, model_name="diacritical/dia2-base", language="en"):
    """
    Convenience function to synthesize speech without maintaining state.
    
    Args:
        text (str): Text to synthesize
        output_filepath (str): Path to save audio file
        model_name (str): Dia2 model to use
        language (str): Language code
        
    Returns:
        str: Path to saved audio file
    """
    tts = Dia2TTS(model_name=model_name, language=language)
    return tts.synthesize_speech(text, output_filepath)
