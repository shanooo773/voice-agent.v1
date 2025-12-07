# Brain of the doctor - LLM Module
# Updated to use open-source LLM via Transformers
# Removed: GROQ API, image processing, vision models

from src.llm import OpenSourceLLM, generate_text_reply

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
    
    return generate_text_reply(query, model_name=model, system_prompt=system_prompt)
