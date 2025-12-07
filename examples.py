"""
Example script demonstrating different use cases of the Open-Source Voice Agent
"""

# Example 1: Basic usage with default settings
def example_basic():
    """Run the voice agent with default settings"""
    print("Example 1: Basic Usage")
    print("-" * 50)
    print("Command: python main.py")
    print("\nThis will:")
    print("  - Use Whisper Large V3 for STT")
    print("  - Use Qwen 2.5-7B-Instruct for LLM")
    print("  - Use MMS-TTS-Eng for TTS")
    print("  - Record 5 seconds of audio")
    print("\n")


# Example 2: Using a different LLM model
def example_different_llm():
    """Run the voice agent with Llama-3"""
    print("Example 2: Using Llama-3")
    print("-" * 50)
    print("Command: python main.py --llm-model meta-llama/Llama-3-8b-instruct")
    print("\nThis will use Llama-3 instead of Qwen for generating responses")
    print("\n")


# Example 3: Using Urdu TTS
def example_urdu_tts():
    """Run the voice agent with Urdu TTS"""
    print("Example 3: Using Urdu TTS")
    print("-" * 50)
    print("Command: python main.py --tts-model facebook/mms-tts-urd")
    print("\nThis will synthesize speech in Urdu")
    print("\n")


# Example 4: Custom system prompt
def example_system_prompt():
    """Run the voice agent with a custom system prompt"""
    print("Example 4: Custom System Prompt")
    print("-" * 50)
    print('Command: python main.py --system-prompt "You are a helpful medical assistant. Keep responses concise."')
    print("\nThis will configure the LLM to act as a medical assistant")
    print("\n")


# Example 5: Longer recording duration
def example_longer_recording():
    """Run the voice agent with longer recording duration"""
    print("Example 5: Longer Recording Duration")
    print("-" * 50)
    print("Command: python main.py --record-duration 10")
    print("\nThis will record 10 seconds of audio instead of the default 5")
    print("\n")


# Example 6: Async mode
def example_async_mode():
    """Run the voice agent in async mode"""
    print("Example 6: Async Mode")
    print("-" * 50)
    print("Command: python main.py --async")
    print("\nThis will run the voice agent in asynchronous mode")
    print("\n")


# Example 7: Programmatic usage
def example_programmatic():
    """Example of using the voice agent programmatically"""
    print("Example 7: Programmatic Usage")
    print("-" * 50)
    print("""
from src.agent import VoiceAgent

# Initialize the voice agent
agent = VoiceAgent(
    stt_model="openai/whisper-large-v3",
    llm_model="Qwen/Qwen2.5-7B-Instruct",
    tts_model="facebook/mms-tts-eng",
    system_prompt="You are a helpful assistant."
)

# Process a pre-recorded audio file
transcription, response, audio_path = agent.process_audio(
    audio_file_path="user_input.wav",
    output_audio_path="agent_response.wav"
)

print(f"User said: {transcription}")
print(f"Agent responded: {response}")

# Play the response
agent.audio_player.play_audio(audio_path)
""")
    print("\n")


# Example 8: Full medical assistant configuration
def example_medical_assistant():
    """Full example for a medical assistant"""
    print("Example 8: Medical Assistant Configuration")
    print("-" * 50)
    print("""
Command: python main.py \\
  --llm-model Qwen/Qwen2.5-7B-Instruct \\
  --record-duration 10 \\
  --system-prompt "You are a professional medical assistant. Provide concise and helpful medical advice. Always remind users to consult with a healthcare professional for serious concerns."

This configures a complete medical assistant setup.
""")
    print("\n")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("Open-Source Voice Agent - Usage Examples")
    print("=" * 70 + "\n")
    
    example_basic()
    example_different_llm()
    example_urdu_tts()
    example_system_prompt()
    example_longer_recording()
    example_async_mode()
    example_programmatic()
    example_medical_assistant()
    
    print("=" * 70)
    print("For more information, see README.md")
    print("=" * 70 + "\n")
