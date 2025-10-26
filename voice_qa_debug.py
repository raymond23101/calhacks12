#!/usr/bin/env python3
"""
Voice Q&A Demo - Debug version with verbose output
"""

from multi_ai import MultiAI
import speech_recognition as sr
from gtts import gTTS
import pygame
import tempfile
import os
import time

def speak(text):
    """Simple speak function"""
    print(f"[SPEAKING] {text}")
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            temp_file = fp.name
            tts.save(temp_file)
        
        pygame.mixer.init()
        pygame.mixer.music.load(temp_file)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        
        pygame.mixer.quit()
        try:
            os.remove(temp_file)
        except:
            pass
    except Exception as e:
        print(f"[ERROR] Speech failed: {e}")

def listen_with_debug():
    """Listen with detailed debugging"""
    recognizer = sr.Recognizer()
    
    # Use USB microphone (device 1)
    mic = sr.Microphone(device_index=1)
    
    print("\n[DEBUG] Adjusting for ambient noise...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)
        print(f"[DEBUG] Ambient noise level: {recognizer.energy_threshold}")
    
    print("[DEBUG] Listening for speech...")
    print("[DEBUG] Speak clearly into the USB microphone...")
    
    with mic as source:
        try:
            # Listen with longer timeout
            audio = recognizer.listen(source, timeout=15, phrase_time_limit=10)
            print(f"[DEBUG] Audio captured! Duration: ~{len(audio.frame_data)/audio.sample_rate:.1f}s")
        except sr.WaitTimeoutError:
            print("[ERROR] Timeout - no speech detected in 15 seconds")
            return None
        except Exception as e:
            print(f"[ERROR] Listen failed: {e}")
            return None
    
    # Try to recognize
    print("[DEBUG] Recognizing speech...")
    try:
        text = recognizer.recognize_google(audio)
        print(f"[SUCCESS] Recognized: '{text}'")
        return text
    except sr.UnknownValueError:
        print("[ERROR] Could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"[ERROR] Recognition service error: {e}")
        return None

def main():
    print("=" * 70)
    print("VOICE Q&A ASSISTANT (DEBUG MODE)")
    print("=" * 70)
    print()
    
    # Initialize AI
    print("Initializing AI...")
    try:
        models_to_try = [
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
        ]
        
        ai = None
        for model in models_to_try:
            try:
                print(f"  Trying model: {model}")
                ai = MultiAI(provider="openrouter", model=model)
                print(f"✓ AI ready (using {model})")
                break
            except Exception as e:
                print(f"  ✗ {model} failed")
                continue
        
        if not ai:
            print("❌ No AI models available")
            return
            
    except Exception as e:
        print(f"❌ AI initialization failed: {e}")
        return
    
    speak("Hello! Voice assistant ready.")
    
    print()
    print("=" * 70)
    print("READY - Press Enter to start listening")
    print("=" * 70)
    print()
    
    while True:
        user_input = input("\nPress Enter to ask a question (or 'q' to quit): ").strip().lower()
        
        if user_input == 'q':
            print("\nExiting...")
            speak("Goodbye!")
            break
        
        # Listen with debug info
        question = listen_with_debug()
        
        if not question:
            speak("I didn't hear anything. Please try again.")
            continue
        
        print(f"\n✓ YOU ASKED: '{question}'")
        
        # Get AI response
        print("\n[DEBUG] Sending to AI...")
        speak("Let me think about that.")
        
        try:
            response = ai.ask_with_context(
                question=question,
                image=None,
                depth_grid=None,
                use_history=True
            )
            
            if response:
                print(f"\n✓ AI RESPONSE:\n{response}\n")
                speak(response)
            else:
                print("❌ No response from AI")
                speak("Sorry, I couldn't generate a response.")
                
        except Exception as e:
            print(f"❌ AI Error: {e}")
            speak("Sorry, I encountered an error.")

if __name__ == "__main__":
    main()

