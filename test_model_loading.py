#!/usr/bin/env python3
"""
Sanity check script for model loading.
Tests that models can be loaded and basic inference works.
"""
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def test_llm_loading():
    """Test LLM model loading and generation."""
    print("\n" + "=" * 80)
    print("Testing LLM Loading")
    print("=" * 80)
    
    try:
        from src.llm import OpenSourceLLM
        
        print("Creating OpenSourceLLM instance...")
        llm = OpenSourceLLM()
        
        print("\nGenerating test reply...")
        response = llm.generate_reply("Hello")
        
        print(f"\n✓ LLM Test PASSED")
        print(f"Response: {response[:100]}...")
        
        return True
    except Exception as e:
        print(f"\n✗ LLM Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_stt_loading():
    """Test STT model loading."""
    print("\n" + "=" * 80)
    print("Testing STT Loading")
    print("=" * 80)
    
    try:
        from src.stt import WhisperSTT
        
        print("Creating WhisperSTT instance...")
        stt = WhisperSTT()
        
        print(f"\n✓ STT Test PASSED")
        print(f"Model loaded on device: {stt.device}")
        
        return True
    except Exception as e:
        print(f"\n✗ STT Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tts_loading():
    """Test TTS model loading."""
    print("\n" + "=" * 80)
    print("Testing TTS Loading")
    print("=" * 80)
    
    try:
        from src.tts import Dia2TTS
        
        print("Creating Dia2TTS instance...")
        tts = Dia2TTS()
        
        print(f"\n✓ TTS Test PASSED")
        print(f"Model loaded on device: {tts.device}")
        
        return True
    except Exception as e:
        print(f"\n✗ TTS Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    results = []
    
    # Test each component
    results.append(("LLM", test_llm_loading()))
    results.append(("STT", test_stt_loading()))
    results.append(("TTS", test_tts_loading()))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")
    
    # Exit with error if any test failed
    if not all(result[1] for result in results):
        print("\n⚠️  Some tests failed")
        sys.exit(1)
    else:
        print("\n✓ All tests passed!")
        sys.exit(0)
