"""
Large Language Model module using open-source models via Transformers pipeline.
Replaces GROQ API LLM and vision models.
"""

from transformers import pipeline
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OpenSourceLLM:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct"):
        """
        Initialize open-source LLM pipeline.
        
        Args:
            model_name (str): Hugging Face model identifier
                Options:
                - "sahil2801/gpt-oss" (lighter weight)
                - "meta-llama/Llama-3-8b-instruct" (requires HF token)
                - "Qwen/Qwen2.5-7B-Instruct" (recommended)
        """
        logging.info(f"Loading LLM model: {model_name}")
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            max_new_tokens=200,
            device=0  # Use CPU by default, change to 0 for GPU
        )
        logging.info("LLM model loaded successfully")
    
    def generate_reply(self, text, system_prompt=None):
        """
        Generate a reply to the given text.
        
        Args:
            text (str): Input text/query
            system_prompt (str): Optional system prompt for context
            
        Returns:
            str: Generated response
        """
        logging.info(f"Generating reply for: {text[:50]}...")
        
        # Combine system prompt and user text if provided
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser: {text}\nAssistant:"
        else:
            full_prompt = text
        
        result = self.pipe(full_prompt)
        response = result[0]["generated_text"]
        
        # Extract only the new generated text (after the prompt)
        if system_prompt:
            # Try to extract just the assistant's response
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
        
        logging.info(f"Response generated: {response[:50]}...")
        return response


def generate_text_reply(text, model_name="Qwen/Qwen2.5-7B-Instruct", system_prompt=None):
    """
    Convenience function to generate text reply without maintaining state.
    
    Args:
        text (str): Input text/query
        model_name (str): LLM model to use
        system_prompt (str): Optional system prompt
        
    Returns:
        str: Generated response
    """
    llm = OpenSourceLLM(model_name=model_name)
    return llm.generate_reply(text, system_prompt=system_prompt)
