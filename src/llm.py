"""
Large Language Model module using open-source models from Hugging Face Transformers
"""
import logging
from transformers import pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OpenSourceLLM:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct"):
        """
        Initialize open-source LLM
        
        Args:
            model_name (str): Hugging Face model name
                Options: "Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3-8b-instruct", "sahil2801/gpt-oss"
        """
        logging.info(f"Loading LLM model: {model_name}")
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            max_new_tokens=200,
            device=-1  # Use CPU, change to 0 for GPU
        )
        self.model_name = model_name
        logging.info("LLM model loaded successfully")
    
    def generate_reply(self, text, system_prompt=None):
        """
        Generate reply from the LLM
        
        Args:
            text (str): Input text/query
            system_prompt (str): Optional system prompt for context
            
        Returns:
            str: Generated response
        """
        logging.info(f"Generating reply for input: {text[:50]}...")
        try:
            # Prepare the prompt
            if system_prompt:
                prompt = f"{system_prompt}\n\nUser: {text}\n\nAssistant:"
            else:
                prompt = f"User: {text}\n\nAssistant:"
            
            # Generate response
            result = self.pipe(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
            response = result[0]["generated_text"]
            
            # Extract only the assistant's response
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            
            logging.info(f"Generated response: {response[:50]}...")
            return response
        except Exception as e:
            logging.error(f"Error during text generation: {e}")
            raise
