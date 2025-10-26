#!/usr/bin/env python3
"""
Simple interactive microphone test
"""

from speech_interface import SpeechInterface

print("="*60)
print("INTERACTIVE MICROPHONE TEST")
print("="*60)
print()

# Initialize
speech = SpeechInterface()

# Test speaker
print("Testing speaker...")
speech.speak("Speaker test. Can you hear me?")
print("✓ Did you hear the speaker? If yes, speaker works!\n")

# Test microphone
while True:
    input("Press Enter to test microphone (or Ctrl+C to quit): ")
    
    print("\n🎤 SPEAK NOW! Say anything...")
    text = speech.listen_once(timeout=5, phrase_time_limit=10)
    
    if text:
        print(f"✅ SUCCESS! You said: \"{text}\"")
        speech.speak(f"I heard: {text}")
    else:
        print("❌ Didn't hear anything. Try speaking louder or closer to the mic.\n")
    
    print()

