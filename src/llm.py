"""
Large Language Model module using open-source models via Transformers.
This version loads the model manually and tries to use 4-bit (bnb) quantization
with device_map='auto' so the heavy model is loaded once and reused.

Behavior:
- Prefer 4-bit quantized load (BitsAndBytesConfig) when CUDA is available.
- Fall back to 8-bit quant (if configured) or full fp16/fp32 load if quant fails.
- Use manual model.generate(...) on the loaded model for fast reuse.

Configuration constants:
- LOCAL_MODELS_DIR: Local directory for model storage fallback
- OFFLOAD_FOLDER: Temporary folder for CPU offloading
- DEFAULT_HF_HOME: Default Hugging Face cache directory
"""
import logging
import os
import torch
from pathlib import Path
from typing import Optional

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
)

# Configuration constants
LOCAL_MODELS_DIR = os.getenv("LOCAL_MODELS_DIR", "./models")
OFFLOAD_FOLDER = os.getenv("OFFLOAD_FOLDER", "/tmp/transformers_offload")
DEFAULT_HF_HOME = os.getenv("HF_HOME", "/opt/hf_cache")

# Check for deprecated TRANSFORMERS_CACHE
if "TRANSFORMERS_CACHE" in os.environ and "HF_HOME" not in os.environ:
    logging.warning(
        "TRANSFORMERS_CACHE is deprecated; prefer HF_HOME — falling back to TRANSFORMERS_CACHE"
    )

# Optional import; presence of bitsandbytes will enable 4-bit loading
try:
    from transformers import BitsAndBytesConfig
    BNB_AVAILABLE = True
except Exception:
    # Older/newer transformers may expose BitsAndBytesConfig in a different way,
    # but we handle that at load time.
    BitsAndBytesConfig = None
    BNB_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _check_bnb_native_available():
    """Check if bitsandbytes native libraries are available (called lazily)."""
    if not BNB_AVAILABLE:
        return False
    
    try:
        import bitsandbytes as bnb
        # Try to detect if native library is available by checking for CUDA libs
        import glob
        bnb_path = Path(bnb.__file__).parent
        cuda_libs = list(bnb_path.glob("libbitsandbytes_cuda*.so"))
        if cuda_libs:
            logging.info(f"BitsAndBytes native libraries found: {[lib.name for lib in cuda_libs]}")
            return True
        else:
            logging.warning(
                "BitsAndBytes package found but native CUDA libraries missing. "
                "Quantization may fail. Install with: pip install --force-reinstall bitsandbytes"
            )
            return False
    except Exception as e:
        logging.warning(f"Could not verify BitsAndBytes native library: {e}")
        return False


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
    
    # Online mode - validate HF repo exists
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        # This will raise if model doesn't exist
        model_info = api.model_info(model_id_or_path)
        logging.info(f"Model {model_id_or_path} validated on HuggingFace Hub")
        return model_id_or_path
    except Exception as e:
        # Model not found on HF - try local models directory
        local_path = os.path.join(LOCAL_MODELS_DIR, model_id_or_path.split("/")[-1])
        if os.path.isdir(local_path):
            logging.info(f"Model not found on HF Hub, using local path: {local_path}")
            return local_path
        
        logging.error(
            f"Model {model_id_or_path} not found on HuggingFace Hub or local directory. "
            f"Error: {e}"
        )
        raise RuntimeError(
            f"Model {model_id_or_path} not found. "
            f"Please check model ID or download to {LOCAL_MODELS_DIR}"
        ) from e


class OpenSourceLLM:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        max_new_tokens: int = 200,
        device: Optional[str] = None,
        prefer_4bit: bool = True,
    ):
        """
        Manual LLM loader that attempts quantized loading to reduce GPU memory use.

        Args:
            model_name: HF model id
            max_new_tokens: default generation length
            device: if None, will use "cuda" if available else "cpu"
            prefer_4bit: attempt 4-bit bnb quantization when possible
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.prefer_4bit = prefer_4bit

        logging.info(f"Loading LLM model: {model_name} (device={self.device})")
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        # Validate model exists before loading
        validated_model_path = ensure_model_exists(self.model_name)
        offline = is_offline_mode()
        
        # Load tokenizer
        try:
            logging.info(f"Loading tokenizer for {validated_model_path} (offline={offline})")
            self.tokenizer = AutoTokenizer.from_pretrained(
                validated_model_path,
                trust_remote_code=True,
                local_files_only=offline
            )
        except Exception as e:
            logging.warning("Failed to load tokenizer with trust_remote_code=True, retrying without it: %s", e)
            self.tokenizer = AutoTokenizer.from_pretrained(
                validated_model_path,
                trust_remote_code=False,
                local_files_only=offline
            )

        # Prepare load kwargs
        load_kwargs = dict(
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=offline
        )
        
        # Create offload folder if needed
        if not os.path.exists(OFFLOAD_FOLDER):
            try:
                os.makedirs(OFFLOAD_FOLDER, exist_ok=True)
                logging.info(f"Created offload folder: {OFFLOAD_FOLDER}")
            except Exception as e:
                logging.warning(f"Could not create offload folder {OFFLOAD_FOLDER}: {e}")
        
        # Check BNB native availability (done lazily on first model load)
        bnb_native_available = _check_bnb_native_available()

        # Prefer quantized loads if CUDA available and bitsandbytes present
        if self.device.startswith("cuda") and BNB_AVAILABLE and bnb_native_available and self.prefer_4bit:
            try:
                # Attempt 4-bit quantization config (transformers >= 4.31+)
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                logging.info(f"Attempting 4-bit BitsAndBytes load for {validated_model_path}")
                logging.info(f"Using device_map='auto' with offload_folder={OFFLOAD_FOLDER}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    validated_model_path,
                    quantization_config=bnb_config,
                    device_map="auto",
                    offload_folder=OFFLOAD_FOLDER,
                    **load_kwargs,
                )
                logging.info("✓ Loaded model in 4-bit mode (BitsAndBytes quantization)")
                logging.info(f"Model device: {next(self.model.parameters()).device}")
                return
            except Exception as e:
                logging.warning(f"4-bit load failed: {e}")
                logging.info("Recommended: Check bitsandbytes installation or try smaller model")
                # try alternate older-style arg if supported
                try:
                    logging.info("Trying legacy load_in_4bit=True fallback")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        validated_model_path,
                        load_in_4bit=True,
                        device_map="auto",
                        offload_folder=OFFLOAD_FOLDER,
                        **load_kwargs,
                    )
                    logging.info("✓ Loaded model in legacy 4-bit mode (load_in_4bit=True)")
                    return
                except Exception as e2:
                    logging.warning(f"Legacy 4-bit fallback failed: {e2}")
                    logging.info("Falling back to 8-bit quantization")

        # Try 8-bit load (less aggressive) if CUDA available and bitsandbytes available
        if self.device.startswith("cuda") and BNB_AVAILABLE and bnb_native_available:
            try:
                logging.info(f"Attempting 8-bit BitsAndBytes load for {validated_model_path}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    validated_model_path,
                    load_in_8bit=True,
                    device_map="auto",
                    offload_folder=OFFLOAD_FOLDER,
                    **load_kwargs,
                )
                logging.info("✓ Loaded model in 8-bit mode (BitsAndBytes quantization)")
                logging.info(f"Model device: {next(self.model.parameters()).device}")
                return
            except Exception as e:
                logging.warning(f"8-bit load failed: {e}")
                logging.info("Falling back to fp16/full precision")

        # Final fallback: standard load to device (fp16 if CUDA available)
        try:
            if self.device.startswith("cuda"):
                torch_dtype = torch.float16
                device_map = "auto"
                logging.info(f"Loading model in fp16 mode on CUDA with device_map='auto'")
            else:
                torch_dtype = None
                device_map = None
                logging.info(f"Loading model on CPU (no quantization)")
                
            self.model = AutoModelForCausalLM.from_pretrained(
                validated_model_path,
                device_map=device_map,
                torch_dtype=torch_dtype,
                offload_folder=OFFLOAD_FOLDER if device_map else None,
                **load_kwargs,
            )
            
            if torch_dtype == torch.float16:
                logging.info("✓ Model loaded in fp16 mode on CUDA")
            else:
                logging.info("✓ Model loaded on CPU")
                
            logging.info(f"Model memory footprint: standard precision")
        except Exception as e:
            logging.error(f"Standard model load failed: {e}")
            logging.error(
                "Troubleshooting suggestions:\n"
                "  - Use a smaller model\n"
                f"  - Ensure sufficient memory (RAM/VRAM)\n"
                f"  - Enable offloading with offload_folder={OFFLOAD_FOLDER}\n"
                "  - For CUDA issues, verify torch CUDA version matches installed CUDA runtime"
            )
            raise

    def generate_reply(self, text: str, system_prompt: Optional[str] = None, max_new_tokens: Optional[int] = None):
        """
        Generate a reply using the loaded model.

        This uses tokenizer + model.generate so the model object remains resident.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model or tokenizer not loaded.")

        if system_prompt:
            prompt = f"{system_prompt}\n\nUser: {text}\nAssistant:"
        else:
            prompt = text

        max_nt = max_new_tokens or self.max_new_tokens

        # Tokenize and move to model device
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_nt,
                do_sample=False,
            )

        # Decode and return the generated text
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # If system_prompt present, try to return only assistant response
        if system_prompt and "Assistant:" in decoded:
            decoded = decoded.split("Assistant:")[-1].strip()

        logging.info(f"Response generated: {decoded[:120]}...")
        return decoded


def generate_text_reply(text: str, model_name: str = "Qwen/Qwen2.5-7B-Instruct", system_prompt: Optional[str] = None):
    """
    Convenience function to generate a single reply without keeping state.
    Note: Using this repeatedly will reload the model each call — prefer OpenSourceLLM()
    which keeps the model resident.
    """
    llm = OpenSourceLLM(model_name=model_name)
    return llm.generate_reply(text, system_prompt=system_prompt)
