"""
Voice Agent that connects STT → LLM → TTS pipeline.
Supports both synchronous and asynchronous processing.
"""

import logging
from src.stt import WhisperSTT
from src.llm import OpenSourceLLM
from src.tts import Dia2TTS
from src.audio_utils import play_audio

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VoiceAgent:
    def __init__(
        self,
        stt_model="openai/whisper-large-v3",
        llm_model="Qwen/Qwen2.5-7B-Instruct",
        tts_model="diacritical/dia2-base",
        system_prompt=None
    ):
        """
        Initialize the Voice Agent with STT, LLM, and TTS components.
        
        Args:
            stt_model (str): Whisper model for STT
            llm_model (str): LLM model for text generation
            tts_model (str): Dia2 model for TTS
            system_prompt (str): System prompt for LLM context
        """
        logging.info("Initializing Voice Agent...")
        
        self.stt = WhisperSTT(model_name=stt_model)
        self.llm = OpenSourceLLM(model_name=llm_model)
        self.tts = Dia2TTS(model_name=tts_model)
        self.system_prompt = system_prompt
        
        logging.info("Voice Agent initialized successfully")
    
    def process_audio(self, audio_filepath, output_filepath):
        """
        Process audio through the complete pipeline: STT → LLM → TTS.
        
        Args:
            audio_filepath (str): Path to input audio file
            output_filepath (str): Path to save output audio file
            
        Returns:
            dict: Results containing transcription, response, and output audio path
        """
        logging.info(f"Processing audio: {audio_filepath}")
        
        # Step 1: Speech to Text
        transcription = self.stt.transcribe(audio_filepath)
        
        # Step 2: Generate LLM response
        response = self.llm.generate_reply(transcription, system_prompt=self.system_prompt)
        
        # Step 3: Text to Speech
        output_audio = self.tts.synthesize_speech(response, output_filepath)
        
        logging.info("Audio processing complete")
        
        return {
            "transcription": transcription,
            "response": response,
            "output_audio": output_audio
        }
    
    def process_and_play(self, audio_filepath, output_filepath):
        """
        Process audio and play the response.
        
        Args:
            audio_filepath (str): Path to input audio file
            output_filepath (str): Path to save output audio file
            
        Returns:
            dict: Results containing transcription, response, and output audio path
        """
        result = self.process_audio(audio_filepath, output_filepath)
        play_audio(result["output_audio"])
        return result


async def process_audio_async(
    audio_filepath,
    output_filepath,
    stt_model="openai/whisper-large-v3",
    llm_model="Qwen/Qwen2.5-7B-Instruct",
    tts_model="diacritical/dia2-base",
    system_prompt=None
):
    """
    Async function to process audio through the pipeline.
    
    Args:
        audio_filepath (str): Path to input audio file
        output_filepath (str): Path to save output audio file
        stt_model (str): Whisper model for STT
        llm_model (str): LLM model for text generation
        tts_model (str): Dia2 model for TTS
        system_prompt (str): System prompt for LLM
        
    Returns:
        dict: Results containing transcription, response, and output audio path
    """
    agent = VoiceAgent(
        stt_model=stt_model,
        llm_model=llm_model,
        tts_model=tts_model,
        system_prompt=system_prompt
    )
    return agent.process_audio(audio_filepath, output_filepath)
