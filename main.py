"""
Main entry point for the Open-Source Voice Agent
"""
import logging
import argparse
from src.agent import VoiceAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description='Open-Source Voice Agent')
    parser.add_argument(
        '--stt-model',
        type=str,
        default='openai/whisper-large-v3',
        help='Whisper model for Speech-to-Text'
    )
    parser.add_argument(
        '--llm-model',
        type=str,
        default='Qwen/Qwen2.5-7B-Instruct',
        help='LLM model (options: Qwen/Qwen2.5-7B-Instruct, meta-llama/Llama-3-8b-instruct, sahil2801/gpt-oss)'
    )
    parser.add_argument(
        '--tts-model',
        type=str,
        default='facebook/mms-tts-eng',
        help='TTS model (options: facebook/mms-tts-eng, facebook/mms-tts-urd)'
    )
    parser.add_argument(
        '--record-duration',
        type=int,
        default=5,
        help='Duration to record audio in seconds'
    )
    parser.add_argument(
        '--async',
        dest='use_async',
        action='store_true',
        help='Use async version of the voice loop'
    )
    parser.add_argument(
        '--system-prompt',
        type=str,
        default=None,
        help='System prompt for the LLM'
    )
    
    args = parser.parse_args()
    
    # Initialize the voice agent
    logging.info("Initializing Open-Source Voice Agent...")
    agent = VoiceAgent(
        stt_model=args.stt_model,
        llm_model=args.llm_model,
        tts_model=args.tts_model,
        system_prompt=args.system_prompt
    )
    
    # Run the voice loop
    if args.use_async:
        import asyncio
        asyncio.run(agent.run_voice_loop_async(record_duration=args.record_duration))
    else:
        agent.run_voice_loop(record_duration=args.record_duration)

if __name__ == '__main__':
    main()
