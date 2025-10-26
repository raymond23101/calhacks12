#!/usr/bin/env python3
"""
Live microphone test - shows real-time what's being heard
"""

import speech_recognition as sr
import time

print("=" * 70)
print("LIVE MICROPHONE TEST")
print("=" * 70)

# Show devices
mic_list = sr.Microphone.list_microphone_names()
print("\nAvailable devices:")
for i, name in enumerate(mic_list):
    marker = "→" if i == 1 else " "
    print(f"  {marker} Device {i}: {name}")

# Use device 1 (USB microphone)
device_index = 1
print(f"\n🎤 Using Device {device_index}: {mic_list[device_index]}")

recognizer = sr.Recognizer()
microphone = sr.Microphone(device_index=device_index)

# Configure
recognizer.pause_threshold = 0.8
recognizer.non_speaking_duration = 0.5
recognizer.dynamic_energy_threshold = True

# Calibrate
print("\n🔊 Calibrating for 2 seconds (be quiet)...")
with microphone as source:
    recognizer.adjust_for_ambient_noise(source, duration=2)

print(f"✓ Energy threshold: {recognizer.energy_threshold}")
print(f"✓ Pause threshold: {recognizer.pause_threshold}s")

print("\n" + "=" * 70)
print("🎙️  SAY 'ASSISTANT' NOW!")
print("   Listening continuously... (Press Ctrl+C to stop)")
print("=" * 70)
print()

try:
    while True:
        try:
            with microphone as source:
                print("⏳ Listening...", end="", flush=True)
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=10)
                print(" ✓ Got audio!")
            
            print("🔄 Recognizing...", end="", flush=True)
            text = recognizer.recognize_google(audio).lower()
            print(f" ✓")
            print(f"✅ HEARD: '{text}'")
            
            # Check for assistant
            if "assistant" in text:
                print("\n" + "🎯" * 20)
                print("WAKE WORD 'ASSISTANT' DETECTED!!!")
                print("🎯" * 20 + "\n")
            
        except sr.WaitTimeoutError:
            print("\r⏱️  Timeout (no speech detected)              ", end="", flush=True)
            time.sleep(0.1)
            print("\r", end="", flush=True)
        except sr.UnknownValueError:
            print(" ❓ Couldn't understand")
        except Exception as e:
            print(f" ❌ Error: {e}")

except KeyboardInterrupt:
    print("\n\n✓ Test stopped")

