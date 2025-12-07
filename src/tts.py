"""
Text-to-Speech module using MMS-TTS from Hugging Face Transformers
"""
import logging
from transformers import pipeline
import scipy.io.wavfile as wavfile
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MMSTTS:
    def __init__(self, model_name="facebook/mms-tts-eng", language="eng"):
        """
        Initialize MMS-TTS model
        
        Args:
            model_name (str): Hugging Face model name for MMS-TTS
                Options: "facebook/mms-tts-eng" for English, "facebook/mms-tts-urd" for Urdu
            language (str): Language code (eng, urd, etc.)
        """
        logging.info(f"Loading MMS-TTS model: {model_name}")
        self.pipe = pipeline(
            "text-to-speech",
            model=model_name,
            device=-1  # Use CPU, change to 0 for GPU
        )
        self.language = language
        logging.info("MMS-TTS model loaded successfully")
    
    def synthesize_speech(self, text, output_path="output.wav"):
        """
        Synthesize speech from text
        
        Args:
            text (str): Text to convert to speech
            output_path (str): Path to save the audio file
            
        Returns:
            str: Path to the saved audio file
        """
        logging.info(f"Synthesizing speech for text: {text[:50]}...")
        try:
            # Generate speech
            result = self.pipe(text)
            
            # Extract audio data and sampling rate
            audio = result["audio"]
            sampling_rate = result["sampling_rate"]
            
            # Convert to int16 format for WAV file
            audio_int16 = np.int16(audio * 32767)
            
            # Save as WAV file
            wavfile.write(output_path, sampling_rate, audio_int16)
            
            logging.info(f"Speech synthesized and saved to: {output_path}")
            return output_path
        except Exception as e:
            logging.error(f"Error during speech synthesis: {e}")
            raise
