"""
Audio utilities for recording and playback using sounddevice.
Replaces speech_recognition and pydub for recording, and subprocess for playback.
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import logging
from scipy.io import wavfile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AudioRecorder:
    def __init__(self, sample_rate=16000):
        """
        Initialize audio recorder.
        
        Args:
            sample_rate (int): Sample rate for recording (default 16000 Hz for Whisper)
        """
        self.sample_rate = sample_rate
        
    def record_audio(self, output_filepath, duration=10):
        """
        Record audio from microphone.
        
        Args:
            output_filepath (str): Path to save recorded audio
            duration (int): Duration to record in seconds
            
        Returns:
            str: Path to saved audio file
        """
        logging.info(f"Recording audio for {duration} seconds...")
        logging.info("Start speaking now...")
        
        # Record audio
        audio_data = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()  # Wait until recording is finished
        
        logging.info("Recording complete.")
        
        # Save as WAV file
        sf.write(output_filepath, audio_data, self.sample_rate)
        logging.info(f"Audio saved to {output_filepath}")
        
        return output_filepath


def play_audio(audio_filepath):
    """
    Play audio file using sounddevice.
    
    Args:
        audio_filepath (str): Path to audio file to play
    """
    logging.info(f"Playing audio: {audio_filepath}")
    
    # Read audio file
    data, sample_rate = sf.read(audio_filepath)
    
    # Play audio
    sd.play(data, sample_rate)
    sd.wait()  # Wait until playback is finished
    
    logging.info("Playback complete.")


def record_audio(output_filepath, duration=10, sample_rate=16000):
    """
    Convenience function to record audio without maintaining state.
    
    Args:
        output_filepath (str): Path to save recorded audio
        duration (int): Duration to record in seconds
        sample_rate (int): Sample rate for recording
        
    Returns:
        str: Path to saved audio file
    """
    recorder = AudioRecorder(sample_rate=sample_rate)
    return recorder.record_audio(output_filepath, duration=duration)
