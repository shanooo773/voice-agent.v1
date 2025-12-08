"""LLM bridge used by the Gradio app."""

import logging
import os

from src.llm import generate_text_reply
from src.model_client import ModelServerError, generate_remote_reply

# Default system prompt for medical consultation
DEFAULT_SYSTEM_PROMPT = """You have to act as a professional doctor, i know you are not but this is for learning purpose. 
Listen to the patient's concerns and provide helpful medical advice.
If you can make a differential diagnosis, suggest some remedies.
Do not add any numbers or special characters in your response.
Your response should be in one long paragraph.
Always answer as if you are answering to a real person.
Do not respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot.
Keep your answer concise (max 2 sentences). No preamble, start your answer right away please."""

def generate_medical_response(query, model="Qwen/Qwen2.5-7B-Instruct", system_prompt=None):
    """
    Generate a medical response to a patient query using open-source LLM.
    
    Args:
        query (str): Patient's question or concern
        model (str): LLM model to use
        system_prompt (str): Optional system prompt override
        
    Returns:
        str: Doctor's response
    """
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT
    
    # Prefer the persistent model server so the heavy model stays resident.
    if os.getenv("MODEL_SERVER_DISABLE", "0") != "1":
        try:
            return generate_remote_reply(
                query=query,
                system_prompt=system_prompt,
                model_name=model,
            )
        except ModelServerError as exc:
            logging.warning("Falling back to in-process generation: %s", exc)

    return generate_text_reply(query, model_name=model, system_prompt=system_prompt)
