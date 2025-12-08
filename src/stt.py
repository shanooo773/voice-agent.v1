"""
Speech-to-Text module using OpenAI Whisper via Transformers.
Replaces GROQ API STT.

Configuration constants:
- LOCAL_MODELS_DIR: Local directory for model storage fallback
- OFFLOAD_FOLDER: Temporary folder for CPU offloading
- DEFAULT_HF_HOME: Default Hugging Face cache directory
"""
import os
import logging
import soundfile as sf
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


class WhisperSTT:
    def __init__(self, model_name="openai/whisper-large-v3", device: Optional[str] = None):
        """
        Initialize the Whisper ASR with manual model loading.
        
        Args:
            model_name (str): HuggingFace Whisper model ID
            device (str): Device to use ('cuda', 'cpu', or None for auto)
                         Default is CPU to avoid CUDA kernel issues on unsupported GPUs.
                         Set WHISPER_DEVICE env var or pass device='cuda' to override.
        """
        self.model_name = model_name
        
        # Determine device - check environment variable first, then parameter, then default to CPU
        # CPU is default for Whisper to avoid CUDA kernel compatibility issues
        if device is None:
            device = os.getenv("WHISPER_DEVICE", "cpu")
        
        self.device = device
        
        # Validate model exists
        validated_model_path = ensure_model_exists(model_name)
        offline = is_offline_mode()
        
        logging.info(f"Loading Whisper STT model: {validated_model_path} (device={self.device}, offline={offline})")
        
        # Try manual model loading first for better control
        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
            
            self.processor = AutoProcessor.from_pretrained(
                validated_model_path,
                local_files_only=offline
            )
            
            # Load model with appropriate dtype
            torch_dtype = torch.float16 if self.device == "cuda" and torch.cuda.is_available() else torch.float32
            
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                validated_model_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                local_files_only=offline
            ).to(self.device)
            
            self.use_manual = True
            logging.info(f"✓ Whisper model loaded manually on {self.device}")
        except Exception as e:
            # Fallback to pipeline
            logging.warning(f"Manual model loading failed, using pipeline: {e}")
            from transformers import pipeline
            
            device_arg = 0 if self.device == "cuda" else -1
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=validated_model_path,
                device=device_arg,
                local_files_only=offline
            )
            self.use_manual = False
            logging.info(f"✓ Whisper pipeline loaded on device {device_arg}")

    def transcribe(self, audio_filepath):
        """Transcribe an audio file to text using Whisper."""
        logging.info(f"Transcribing audio file: {audio_filepath}")

        # Load audio data
        audio_data, sample_rate = sf.read(audio_filepath)
        if audio_data.dtype != "float32":
            audio_data = audio_data.astype("float32")

        if self.use_manual:
            # Manual transcription
            try:
                inputs = self.processor(
                    audio_data,
                    sampling_rate=sample_rate,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    generated_ids = self.model.generate(**inputs)
                
                transcription = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0]
            except Exception as e:
                logging.error(f"Manual transcription failed: {e}")
                raise
        else:
            # Pipeline transcription
            result = self.pipe({"array": audio_data, "sampling_rate": sample_rate})
            transcription = result["text"]
        
        logging.info(f"Transcription complete: {transcription[:50]}...")
        return transcription


def transcribe_audio(audio_filepath, model_name="openai/whisper-large-v3"):
    """Convenience helper that instantiates WhisperSTT and transcribes input."""

    stt = WhisperSTT(model_name=model_name)
    return stt.transcribe(audio_filepath)
