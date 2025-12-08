"""
Large Language Model module using open-source models via Transformers.
This version loads the model manually and tries to use 4-bit (bnb) quantization
with device_map='auto' so the heavy model is loaded once and reused.

Behavior:
- Prefer 4-bit quantized load (BitsAndBytesConfig) when CUDA is available.
- Fall back to 8-bit quant (if configured) or full fp16/fp32 load if quant fails.
- Use manual model.generate(...) on the loaded model for fast reuse.
"""
import logging
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
)
from typing import Optional

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
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        except Exception as e:
            logging.warning("Failed to load tokenizer with trust_remote_code=True, retrying without it: %s", e)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=False)

        # Prepare load kwargs
        load_kwargs = dict(low_cpu_mem_usage=True, trust_remote_code=True)

        # Prefer quantized loads if CUDA available and bitsandbytes present
        if self.device.startswith("cuda") and BNB_AVAILABLE and self.prefer_4bit:
            try:
                # Attempt 4-bit quantization config (transformers >= 4.31+)
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                logging.info("Attempting 4-bit BitsAndBytes load (quantized).")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    **load_kwargs,
                )
                logging.info("Loaded model in 4-bit mode (BitsAndBytes).")
                return
            except Exception as e:
                logging.warning("4-bit load failed: %s", e)
                # try alternate older-style arg if supported
                try:
                    logging.info("Trying legacy load_in_4bit=True fallback.")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        load_in_4bit=True,
                        device_map="auto",
                        **load_kwargs,
                    )
                    logging.info("Loaded model in legacy 4-bit mode (load_in_4bit=True).")
                    return
                except Exception as e2:
                    logging.warning("Legacy 4-bit fallback failed: %s", e2)
                    # fall through to 8-bit or full precision

        # Try 8-bit load (less aggressive) if CUDA available and bitsandbytes available
        if self.device.startswith("cuda") and BNB_AVAILABLE:
            try:
                logging.info("Attempting 8-bit BitsAndBytes load (load_in_8bit=True).")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    load_in_8bit=True,
                    device_map="auto",
                    **load_kwargs,
                )
                logging.info("Loaded model in 8-bit mode (BitsAndBytes).")
                return
            except Exception as e:
                logging.warning("8-bit load failed: %s", e)

        # Final fallback: standard load to device (fp16 if CUDA available)
        try:
            torch_dtype = torch.float16 if self.device.startswith("cuda") else None
            logging.info("Loading model in standard mode (device_map='auto' or cpu).")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto" if self.device.startswith("cuda") else None,
                torch_dtype=torch_dtype,
                **load_kwargs,
            )
            logging.info("Model loaded (standard path).")
        except Exception as e:
            logging.error("Standard model load failed: %s", e)
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
