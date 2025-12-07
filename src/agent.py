"""
Voice Agent module that orchestrates STT -> LLM -> TTS pipeline
"""
import logging
import asyncio
from .stt import WhisperSTT
from .llm import OpenSourceLLM
from .tts import MMSTTS
from .audio_utils import AudioRecorder, AudioPlayer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VoiceAgent:
    def __init__(
        self,
        stt_model="openai/whisper-large-v3",
        llm_model="Qwen/Qwen2.5-7B-Instruct",
        tts_model="facebook/mms-tts-eng",
        system_prompt=None
    ):
        """
        Initialize Voice Agent with STT, LLM, and TTS models
        
        Args:
            stt_model (str): Whisper model name
            llm_model (str): LLM model name
            tts_model (str): TTS model name
            system_prompt (str): Optional system prompt for the LLM
        """
        logging.info("Initializing Voice Agent...")
        
        # Initialize components
        self.stt = WhisperSTT(model_name=stt_model)
        self.llm = OpenSourceLLM(model_name=llm_model)
        self.tts = MMSTTS(model_name=tts_model)
        self.audio_recorder = AudioRecorder()
        self.audio_player = AudioPlayer()
        
        self.system_prompt = system_prompt
        
        logging.info("Voice Agent initialized successfully")
    
    def process_audio(self, audio_file_path, output_audio_path="response.wav"):
        """
        Process audio file through the complete pipeline: STT -> LLM -> TTS
        
        Args:
            audio_file_path (str): Path to input audio file
            output_audio_path (str): Path to save response audio
            
        Returns:
            tuple: (transcription, llm_response, audio_path)
        """
        logging.info("Processing audio through voice pipeline...")
        
        try:
            # Step 1: Speech to Text
            transcription = self.stt.transcribe_audio(audio_file_path)
            logging.info(f"Transcription: {transcription}")
            
            # Step 2: Generate LLM response
            llm_response = self.llm.generate_reply(transcription, self.system_prompt)
            logging.info(f"LLM Response: {llm_response}")
            
            # Step 3: Text to Speech
            response_audio = self.tts.synthesize_speech(llm_response, output_audio_path)
            logging.info(f"Response audio saved to: {response_audio}")
            
            return transcription, llm_response, response_audio
        except Exception as e:
            logging.error(f"Error in voice pipeline: {e}")
            raise
    
    def run_voice_loop(self, record_duration=5, auto_play=True):
        """
        Run interactive voice loop
        
        Args:
            record_duration (int): Duration to record user input in seconds
            auto_play (bool): Whether to automatically play the response
        """
        logging.info("Starting voice loop. Press Ctrl+C to exit.")
        
        try:
            while True:
                # Record user input
                input("Press Enter to start recording...")
                audio_file = self.audio_recorder.record_audio(
                    duration=record_duration,
                    file_path="user_input.wav"
                )
                
                # Process through pipeline
                transcription, response, response_audio = self.process_audio(
                    audio_file,
                    output_audio_path="agent_response.wav"
                )
                
                # Display results
                print(f"\n{'='*50}")
                print(f"You said: {transcription}")
                print(f"Agent response: {response}")
                print(f"{'='*50}\n")
                
                # Play response
                if auto_play:
                    self.audio_player.play_audio(response_audio)
                
        except KeyboardInterrupt:
            logging.info("\nVoice loop stopped by user")
        except Exception as e:
            logging.error(f"Error in voice loop: {e}")
            raise
    
    async def process_audio_async(self, audio_file_path, output_audio_path="response.wav"):
        """
        Async version of process_audio
        
        Args:
            audio_file_path (str): Path to input audio file
            output_audio_path (str): Path to save response audio
            
        Returns:
            tuple: (transcription, llm_response, audio_path)
        """
        logging.info("Processing audio through voice pipeline (async)...")
        
        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            
            # Step 1: Speech to Text
            transcription = await loop.run_in_executor(
                None, self.stt.transcribe_audio, audio_file_path
            )
            logging.info(f"Transcription: {transcription}")
            
            # Step 2: Generate LLM response
            llm_response = await loop.run_in_executor(
                None, self.llm.generate_reply, transcription, self.system_prompt
            )
            logging.info(f"LLM Response: {llm_response}")
            
            # Step 3: Text to Speech
            response_audio = await loop.run_in_executor(
                None, self.tts.synthesize_speech, llm_response, output_audio_path
            )
            logging.info(f"Response audio saved to: {response_audio}")
            
            return transcription, llm_response, response_audio
        except Exception as e:
            logging.error(f"Error in async voice pipeline: {e}")
            raise
    
    async def run_voice_loop_async(self, record_duration=5, auto_play=True):
        """
        Async version of run_voice_loop
        
        Args:
            record_duration (int): Duration to record user input in seconds
            auto_play (bool): Whether to automatically play the response
        """
        logging.info("Starting async voice loop. Press Ctrl+C to exit.")
        
        try:
            while True:
                # Record user input
                input("Press Enter to start recording...")
                loop = asyncio.get_event_loop()
                audio_file = await loop.run_in_executor(
                    None,
                    self.audio_recorder.record_audio,
                    record_duration,
                    "user_input.wav"
                )
                
                # Process through pipeline
                transcription, response, response_audio = await self.process_audio_async(
                    audio_file,
                    output_audio_path="agent_response.wav"
                )
                
                # Display results
                print(f"\n{'='*50}")
                print(f"You said: {transcription}")
                print(f"Agent response: {response}")
                print(f"{'='*50}\n")
                
                # Play response
                if auto_play:
                    await loop.run_in_executor(
                        None, self.audio_player.play_audio, response_audio
                    )
                
        except KeyboardInterrupt:
            logging.info("\nAsync voice loop stopped by user")
        except Exception as e:
            logging.error(f"Error in async voice loop: {e}")
            raise
