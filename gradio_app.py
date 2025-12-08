# VoiceBot UI with Gradio
# Updated to use open-source local models
# Removed: Image processing, GROQ API, ElevenLabs API
# Uses: Local Whisper STT, open-source LLM, Dia2 TTS

import os

import gradio as gr
from brain_of_the_doctor import generate_medical_response
from voice_of_the_patient import transcribe_patient_audio
from voice_of_the_doctor import generate_doctor_voice


# System prompt for medical consultation
system_prompt = """You have to act as a professional doctor, i know you are not but this is for learning purpose. 
            Listen to the patient's concerns and provide helpful medical advice.
            If you can make a differential diagnosis, suggest some remedies.
            Do not add any numbers or special characters in your response.
            Your response should be in one long paragraph.
            Always answer as if you are answering to a real person.
            Do not say things like 'Based on your audio' but respond naturally.
            Do not respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot.
            Keep your answer concise (max 2 sentences). No preamble, start your answer right away please."""


def process_audio_input(audio_filepath):
    """
    Process audio input through the voice agent pipeline.
    Voice-only interaction - no image processing.
    
    Args:
        audio_filepath (str): Path to audio file from microphone
        
    Returns:
        tuple: (transcription, doctor_response, audio_output_path)
    """
    if isinstance(audio_filepath, dict):
        audio_filepath = audio_filepath.get("name") or audio_filepath.get("path")

    if not audio_filepath:
        raise gr.Error("No audio captured. Please record or upload a clip before submitting.")

    if not os.path.exists(audio_filepath):
        raise gr.Error("Audio file missing on disk. Please retry recording.")

    # Step 1: Transcribe patient's audio using Whisper
    speech_to_text_output = transcribe_patient_audio(audio_filepath)
    
    # Step 2: Generate doctor's response using open-source LLM
    doctor_response = generate_medical_response(
        query=speech_to_text_output,
        system_prompt=system_prompt
    )
    
    # Step 3: Convert doctor's response to speech using Dia2 TTS
    audio_output = generate_doctor_voice(
        input_text=doctor_response,
        output_filepath="final.wav"  # Changed to .wav for Dia2 TTS
    )
    
    return speech_to_text_output, doctor_response, audio_output


# Create the Gradio interface
iface = gr.Interface(
    fn=process_audio_input,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath", label="Patient's Voice")
    ],
    outputs=[
        gr.Textbox(label="Speech to Text (Patient's Words)"),
        gr.Textbox(label="Doctor's Response (Text)"),
        gr.Audio(label="Doctor's Voice Response")
    ],
    title="AI Doctor Voice Agent (Open Source)",
    description="Voice-only medical consultation using open-source models: Whisper STT, Qwen LLM, and Dia2 TTS"
)

iface.launch(debug=True)