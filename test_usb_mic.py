#!/usr/bin/env python3
"""
USB Microphone Test
Uses USBAudio1.0 (device 1)
"""

from speech_interface import SpeechInterface

print("="*60)
print("USB MICROPHONE TEST (USBAudio1.0)")
print("="*60)
print()

# Initialize with USB microphone
print("Initializing with USB microphone (device 1)...")
speech = SpeechInterface()

# Set to USB microphone
speech.set_microphone(device_index=1)
print("✓ USB microphone selected\n")

# Test speaker
print("Testing speaker...")
speech.speak("USB microphone test. Speaker is working.")
print("✓ Speaker OK\n")

# Interactive test loop
print("Press Enter to start recording, then speak into USB mic")
print("Press Ctrl+C to quit\n")

try:
    while True:
        input("Press Enter to record: ")
        
        print("\n🎤 RECORDING - Speak into USB microphone now!")
        text = speech.listen_once(timeout=5, phrase_time_limit=10)
        
        if text:
            print(f"✅ SUCCESS! Heard: \"{text}\"")
            speech.speak(f"You said: {text}")
        else:
            print("❌ No speech detected. Make sure you're speaking into the USB mic.\n")
        
        print()

except KeyboardInterrupt:
    print("\n\n✓ Test complete!")

