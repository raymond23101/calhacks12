#!/usr/bin/env python3
"""
Simple Microphone Test
Tests TTS (text-to-speech) and STT (speech-to-text)
"""

from speech_interface import SpeechInterface
import time

def main():
    print("="*60)
    print("MICROPHONE & SPEAKER TEST")
    print("="*60)
    print()
    
    # Initialize speech interface
    print("Initializing speech interface...")
    try:
        speech = SpeechInterface(wake_word="assistant")
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return
    
    print("✓ Speech interface initialized\n")
    
    # Test 1: Text-to-Speech (Speaker)
    print("TEST 1: Speaker (Text-to-Speech)")
    print("-" * 60)
    speech.speak("Hello! I am testing the speaker. Can you hear me?")
    print("✓ Speaker test complete\n")
    
    time.sleep(1)
    
    # Test 2: List available microphones
    print("TEST 2: Available Microphones")
    print("-" * 60)
    speech.get_microphone_list()
    print()
    
    # Test 3: Speech-to-Text (Microphone)
    print("TEST 3: Microphone (Speech-to-Text)")
    print("-" * 60)
    speech.speak("Please say something after the beep.")
    time.sleep(0.5)
    
    print("\n🎤 Listening for 5 seconds... SPEAK NOW!")
    text = speech.listen_once(timeout=5, phrase_time_limit=10)
    
    if text:
        print(f"✓ SUCCESS! I heard: \"{text}\"")
        speech.speak(f"I heard you say: {text}")
    else:
        print("❌ No speech detected or couldn't understand")
        speech.speak("I didn't hear anything. Please try again.")
    
    print()
    
    # Test 4: Interactive Test
    print("TEST 4: Interactive Mode")
    print("-" * 60)
    print("Say 'hello' or 'test' when you hear the prompt...")
    speech.speak("Say hello or test")
    time.sleep(0.5)
    
    print("🎤 Listening...")
    text = speech.listen_once(timeout=5)
    
    if text:
        print(f"You said: {text}")
        
        if "hello" in text.lower():
            speech.speak("Hello to you too!")
        elif "test" in text.lower():
            speech.speak("Test successful!")
        else:
            speech.speak(f"I heard: {text}")
    else:
        print("No speech detected")
    
    print()
    print("="*60)
    print("✓ All tests complete!")
    print("="*60)
    print()
    print("To run continuous listening demo, use:")
    print("  python speech_interface.py")


if __name__ == "__main__":
    main()

