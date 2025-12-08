"""
Text-to-Speech module using Dia2 via Transformers.
Replaces ElevenLabs and gTTS.

Configuration constants:
- LOCAL_MODELS_DIR: Local directory for model storage fallback
- OFFLOAD_FOLDER: Temporary folder for CPU offloading
- DEFAULT_HF_HOME: Default Hugging Face cache directory
"""
import os
import logging
import numpy as np
import scipy.io.wavfile as wavfile
import torch
from pathlib import Path
from typing import Optional

# Configuration constants
LOCAL_MODELS_DIR = os.getenv("LOCAL_MODELS_DIR", "./models")
OFFLOAD_FOLDER = os.getenv("OFFLOAD_FOLDER", "/tmp/transformers_offload")
DEFAULT_HF_HOME = os.getenv("HF_HOME", "/opt/hf_cache")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def is_offline_mode() -> bool:
    """Check if we're in offline mode based on environment variables."""
    try:
        from transformers.utils import is_offline_mode as transformers_is_offline
        if transformers_is_offline():
            return True
    except Exception:
        pass
    
    # Check explicit offline environment variables
    if os.getenv("HF_OFFLINE", "0") == "1":
        return True
    if os.getenv("HF_FORCE_OFFLINE", "0") == "1":
        return True
    if os.getenv("TRANSFORMERS_OFFLINE", "0") == "1":
        return True
    
    return False


def ensure_model_exists(model_id_or_path: str) -> str:
    """
    Validate model existence and return the path to use.
    
    Args:
        model_id_or_path: HuggingFace model ID or local path
        
    Returns:
        str: Validated model path/ID to use
        
    Raises:
        RuntimeError: If model doesn't exist and cannot be downloaded
    """
    # If it's a local directory, use it directly
    if os.path.isdir(model_id_or_path):
        logging.info(f"Using local model directory: {model_id_or_path}")
        return model_id_or_path
    
    # Check if we're offline
    offline = is_offline_mode()
    
    if offline:
        # In offline mode, check if model exists in HF cache
        hf_home = os.getenv("HF_HOME", os.getenv("TRANSFORMERS_CACHE", DEFAULT_HF_HOME))
        logging.error(
            f"Offline mode: model {model_id_or_path} not found at local path. "
            f"Please download to HF cache (HF_HOME={hf_home}) or set path to a directory with model files."
        )
        raise RuntimeError(
            f"Offline mode: model {model_id_or_path} not found locally. "
            f"Cannot download. Please pre-download models or disable offline mode."
        )
    
    # Online mode - return model ID for download
    return model_id_or_path


class Dia2TTS:
    def __init__(self, model_name="diacritical/dia2-base", language="en", device: Optional[str] = None):
        """
        Initialize Dia2 TTS with pipeline (most reliable for TTS models).
        
        Args:
            model_name (str): Hugging Face model identifier
                Options:
                - "diacritical/dia2-base" (English)
                - "diacritical/dia2-multilingual" (for Urdu and other languages)
            language (str): Language code (e.g., 'en' for English, 'ur' for Urdu)
            device (str): Device to use ('cuda', 'cpu', or None for auto)
        """
        self.language = language
        self.model_name = model_name
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Validate model exists
        validated_model_path = ensure_model_exists(model_name)
        offline = is_offline_mode()
        
        logging.info(f"Loading Dia2 TTS model: {validated_model_path} (device={self.device}, offline={offline})")
        
        # Use pipeline for TTS (most reliable approach)
        from transformers import pipeline
        
        device_arg = 0 if self.device == "cuda" else -1
        self.pipe = pipeline(
            "text-to-speech",
            model=validated_model_path,
            device=device_arg,
            local_files_only=offline
        )
        self.use_manual = False  # Keep for compatibility
        logging.info(f"✓ Dia2 TTS pipeline loaded on device {device_arg}")
    
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
        
        # Generate audio using pipeline
        # Attempt with use_global=True first for better performance (if supported)
        try:
            # Try calling with use_global parameter
            result = self.pipe(text, use_global=True)
            logging.info("✓ Generated with use_global=True")
        except TypeError:
            # Fallback if use_global not supported by this model/pipeline
            logging.info("use_global=True not supported, using default generation")
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
