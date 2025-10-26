#!/usr/bin/env python3
"""
Simple Speaker Test (Text-to-Speech only)
"""

from speech_interface import SpeechInterface

print("="*60)
print("SPEAKER TEST (Text-to-Speech)")
print("="*60)
print()

# Initialize speech interface
print("Initializing speech interface...")
speech = SpeechInterface()
print("✓ Initialized\n")

# Test 1: Basic speech
print("TEST 1: Basic Speech")
print("-" * 60)
print("You should hear: 'Testing speaker output'")
speech.speak("Testing speaker output")
print("✓ Test 1 complete\n")

# Test 2: Longer sentence
print("TEST 2: Longer Sentence")
print("-" * 60)
print("You should hear a full sentence...")
speech.speak("This is a longer sentence to test the text to speech system. Can you hear me clearly?")
print("✓ Test 2 complete\n")

# Test 3: Numbers and punctuation
print("TEST 3: Numbers and Punctuation")
print("-" * 60)
print("Testing numbers...")
speech.speak("Testing numbers: 1, 2, 3, 4, 5. Ten, twenty, thirty.")
print("✓ Test 3 complete\n")

# Test 4: Speed test
print("TEST 4: Multiple Short Phrases")
print("-" * 60)
speech.speak("Hello.")
speech.speak("How are you?")
speech.speak("Speaker test complete.")
print("✓ Test 4 complete\n")

print("="*60)
print("✓ ALL SPEAKER TESTS COMPLETE!")
print("="*60)
print()
print("If you heard all the messages clearly, your speaker works!")
print("If not, check:")
print("  - Volume is turned up")
print("  - Correct output device selected in System Preferences")
print("  - Speakers are not muted")

