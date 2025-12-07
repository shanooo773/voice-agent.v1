"""
Audio utilities for recording and playback using sounddevice and scipy
"""
import logging
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
import subprocess
import platform
import os
import shlex

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AudioRecorder:
    def __init__(self, sample_rate=16000):
        """
        Initialize audio recorder
        
        Args:
            sample_rate (int): Sample rate for recording (default: 16000 Hz)
        """
        self.sample_rate = sample_rate
        logging.info(f"Audio recorder initialized with sample rate: {sample_rate} Hz")
    
    def record_audio(self, duration=5, file_path="recording.wav"):
        """
        Record audio from microphone
        
        Args:
            duration (int): Duration to record in seconds
            file_path (str): Path to save the recorded audio
            
        Returns:
            str: Path to the saved audio file
        """
        logging.info(f"Recording audio for {duration} seconds...")
        try:
            # Record audio
            audio_data = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()  # Wait until recording is finished
            
            # Convert to int16 format
            audio_int16 = np.int16(audio_data * 32767)
            
            # Save as WAV file
            wavfile.write(file_path, self.sample_rate, audio_int16)
            
            logging.info(f"Audio recorded and saved to: {file_path}")
            return file_path
        except Exception as e:
            logging.error(f"Error during audio recording: {e}")
            raise
    
    def record_until_silence(self, file_path="recording.wav", silence_threshold=0.01, silence_duration=2):
        """
        Record audio until silence is detected
        
        Args:
            file_path (str): Path to save the recorded audio
            silence_threshold (float): Threshold for detecting silence
            silence_duration (int): Duration of silence to stop recording (in seconds)
            
        Returns:
            str: Path to the saved audio file
        """
        logging.info("Recording audio until silence is detected...")
        try:
            recording = []
            silence_counter = 0
            chunk_duration = 0.5  # Record in 0.5 second chunks
            
            while True:
                # Record a chunk
                chunk = sd.rec(
                    int(chunk_duration * self.sample_rate),
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype='float32'
                )
                sd.wait()
                
                # Add chunk to recording
                recording.append(chunk)
                
                # Check if chunk is silent
                if np.max(np.abs(chunk)) < silence_threshold:
                    silence_counter += chunk_duration
                    if silence_counter >= silence_duration:
                        logging.info("Silence detected, stopping recording")
                        break
                else:
                    silence_counter = 0
            
            # Concatenate all chunks
            audio_data = np.concatenate(recording)
            
            # Convert to int16 format
            audio_int16 = np.int16(audio_data * 32767)
            
            # Save as WAV file
            wavfile.write(file_path, self.sample_rate, audio_int16)
            
            logging.info(f"Audio recorded and saved to: {file_path}")
            return file_path
        except Exception as e:
            logging.error(f"Error during audio recording: {e}")
            raise


class AudioPlayer:
    @staticmethod
    def play_audio(file_path):
        """
        Play audio file using system's default audio player
        
        Args:
            file_path (str): Path to the audio file to play
        
        Note:
            File paths are safely handled via subprocess list form,
            which automatically escapes special characters and spaces.
        """
        logging.info(f"Playing audio file: {file_path}")
        os_name = platform.system()
        try:
            if os_name == "Darwin":  # macOS
                subprocess.run(['afplay', file_path], check=True)
            elif os_name == "Windows":  # Windows
                # Use list form for proper path handling
                subprocess.run(['powershell', '-c', f'(New-Object Media.SoundPlayer "{file_path}").PlaySync();'], check=True)
            elif os_name == "Linux":  # Linux
                # Try multiple players in order of preference
                # Using list form ensures proper handling of paths with spaces/special chars
                players = [
                    ['aplay', file_path],
                    ['mpg123', file_path],
                    ['ffplay', '-nodisp', '-autoexit', file_path]
                ]
                for player_cmd in players:
                    try:
                        subprocess.run(player_cmd, check=True)
                        break
                    except FileNotFoundError:
                        continue
            else:
                raise OSError("Unsupported operating system")
            logging.info("Audio playback complete")
        except Exception as e:
            logging.error(f"Error during audio playback: {e}")
            raise
