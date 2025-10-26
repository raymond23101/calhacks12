#!/usr/bin/env python3
"""
Test wake word detection to debug microphone issues
"""

import speech_recognition as sr
import time

def test_wake_word(device_index):
    """Test wake word detection on specific device"""
    print("=" * 70)
    print(f"TESTING WAKE WORD DETECTION - Device {device_index}")
    print("=" * 70)
    
    # List available microphones
    mic_list = sr.Microphone.list_microphone_names()
    if device_index < len(mic_list):
        print(f"\n🎤 Using: {mic_list[device_index]}")
    else:
        print(f"\n❌ Device {device_index} not found!")
        return
    
    recognizer = sr.Recognizer()
    microphone = sr.Microphone(device_index=device_index)
    
    # Configure
    recognizer.pause_threshold = 0.8
    recognizer.non_speaking_duration = 0.5
    recognizer.dynamic_energy_threshold = True
    
    # Calibrate
    print("\n🔊 Adjusting for ambient noise...")
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
    print(f"✓ Energy threshold: {recognizer.energy_threshold}")
    
    print("\n" + "=" * 70)
    print("🎯 SAY 'ASSISTANT' followed by a question")
    print("   (e.g., 'Assistant, what do you see?')")
    print("   Will listen for 30 seconds...")
    print("=" * 70)
    
    start_time = time.time()
    while time.time() - start_time < 30:
        try:
            print("\n⏳ Listening...")
            with microphone as source:
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=10)
            
            print("🔄 Processing audio...")
            text = recognizer.recognize_google(audio).lower()
            print(f"✅ Heard: '{text}'")
            
            # Check for wake word
            if "assistant" in text:
                print("\n" + "=" * 70)
                print("🎯 WAKE WORD 'ASSISTANT' DETECTED!")
                print("=" * 70)
                
                # Extract question
                question = text.replace("assistant", "").strip()
                if question:
                    print(f"❓ Question extracted: '{question}'")
                else:
                    print("❓ No question detected (just wake word)")
                
                print("\n✅ Wake word detection WORKING!")
                return True
            else:
                print(f"⚠️  Wake word NOT found in: '{text}'")
            
        except sr.WaitTimeoutError:
            print("⏱️  Timeout (no speech detected)")
            continue
        except sr.UnknownValueError:
            print("❓ Audio captured but couldn't understand")
            continue
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    print("\n" + "=" * 70)
    print("❌ 30 seconds elapsed - No wake word detected")
    print("=" * 70)
    return False

if __name__ == "__main__":
    print("\n📋 Available Microphones:")
    mic_list = sr.Microphone.list_microphone_names()
    for i, name in enumerate(mic_list):
        is_usb = "USB" in name.upper() or "AUDIO" in name.upper()
        marker = "🎤 USB →" if is_usb else "  "
        print(f"{marker} Device {i}: {name}")
    
    print("\n" + "=" * 70)
    device = int(input("Enter device number to test (default 1): ") or "1")
    
    success = test_wake_word(device)
    
    if not success:
        print("\n💡 TROUBLESHOOTING:")
        print("   1. Make sure you're speaking into the correct microphone")
        print("   2. Check System Preferences → Sound → Input")
        print("   3. Speak clearly and loudly")
        print("   4. Try a different device number")
        print(f"\n   Try testing device 0 or 2 if device {device} didn't work")

