#!/usr/bin/env python3
"""
Speech Interface - Text-to-Speech and Speech-to-Text
Handles microphone input and speaker output for voice interaction
"""

import pyttsx3
import speech_recognition as sr
import threading
import time
from typing import Optional, Callable


class SpeechInterface:
    """
    Speech interface for voice interaction
    - Text-to-Speech (TTS) for speaking responses
    - Speech-to-Text (STT) for listening to user
    - Wake word detection
    """
    
    def __init__(self, wake_word="assistant", rate=175, volume=0.9):
        """
        Initialize speech interface
        
        Args:
            wake_word: Word to activate listening (default: "assistant")
            rate: Speech rate in words per minute (default: 175)
            volume: Speaker volume 0.0 to 1.0 (default: 0.9)
        """
        self.wake_word = wake_word.lower()
        
        # Text-to-Speech engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', rate)
        self.tts_engine.setProperty('volume', volume)
        
        # Speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Adjust for ambient noise on first run
        print("Calibrating microphone for ambient noise...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        print("✓ Microphone calibrated")
        
        # Threading
        self.listening = False
        self.listen_thread = None
        self.callback = None
        
    def speak(self, text, blocking=True):
        """
        Speak text through speaker using Fish Audio TTS (Emma Watson voice)
        
        Args:
            text: Text to speak
            blocking: Wait for speech to complete (default: True)
        """
        try:
            if blocking:
                print(f"[SPEECH] Text to speak ({len(text)} chars): '{text[:80]}...'")
                
                import requests
                import pygame
                import tempfile
                import os
                
                # Fish Audio TTS API configuration
                FISH_AUDIO_API_KEY = "b5f66e1290844c56b8095b053acad4c1"
                FISH_AUDIO_URL = "https://api.fish.audio/v1/tts"
                
                # Emma Watson voice reference ID (you may need to adjust this)
                # Common voice IDs: try "emma-watson" or check Fish Audio docs for exact ID
                VOICE_REFERENCE = "41db41746b9c4bd18053c2bfc213b476"  # Emma Watson voice ID
                
                # Generate speech audio file
                print(f"[SPEECH] Generating audio with Fish Audio (Emma Watson voice)...")
                
                headers = {
                    "Authorization": f"Bearer {FISH_AUDIO_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "text": text,
                    "reference_id": VOICE_REFERENCE,
                    "format": "mp3",
                    "mp3_bitrate": 128,
                    "normalize": True
                }
                
                response = requests.post(FISH_AUDIO_URL, json=payload, headers=headers, timeout=30)
                
                if response.status_code != 200:
                    print(f"[SPEECH] Fish Audio API error: {response.status_code} - {response.text}")
                    raise Exception(f"Fish Audio API error: {response.status_code}")
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                    temp_file = fp.name
                    fp.write(response.content)
                
                # Play audio using pygame
                print(f"[SPEECH] Playing audio...")
                try:
                    # Ensure pygame mixer is properly initialized
                    if pygame.mixer.get_init() is None:
                        pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
                        pygame.mixer.init()
                    
                    pygame.mixer.music.load(temp_file)
                    pygame.mixer.music.play()
                    
                    # Wait for audio to finish
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                    
                    # Don't quit mixer - keep it alive for next use
                    
                except pygame.error as e:
                    print(f"[SPEECH] Pygame audio error: {e}")
                    print(f"[SPEECH] Falling back to system audio...")
                    # Fallback to system audio
                    import subprocess
                    try:
                        subprocess.run(['afplay', temp_file], check=True)
                        print(f"[SPEECH] ✓ System audio successful")
                    except Exception as fallback_error:
                        print(f"[SPEECH] System audio also failed: {fallback_error}")
                try:
                    os.remove(temp_file)
                except:
                    pass
                
                print(f"[SPEECH] ✓ Speech completed")
            else:
                # Non-blocking speech
                thread = threading.Thread(target=self._speak_async, args=(text,))
                thread.daemon = True
                thread.start()
        except Exception as e:
            print(f"[ERROR] Speech failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _speak_async(self, text):
        """Internal method for non-blocking speech"""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
    
    def listen_once(self, timeout=10, phrase_time_limit=10) -> Optional[str]:
        """
        Listen for a single speech input with improved USB mic handling
        
        Args:
            timeout: Max seconds to wait for speech to start
            phrase_time_limit: Max seconds for the phrase
            
        Returns:
            Recognized text or None if failed
        """
        try:
            with self.microphone as source:
                # Quick ambient noise adjustment before listening
                print("🎤 Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                print(f"🎤 Listening... (speak now, will auto-stop after {self.recognizer.pause_threshold}s of silence)")
                print(f"   Energy threshold: {self.recognizer.energy_threshold:.0f}")
                
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )
                
                print(f"✓ Audio captured ({len(audio.frame_data)} bytes)")
            
            print("🔄 Sending to Google Speech Recognition...")
            text = self.recognizer.recognize_google(audio, show_all=False)
            print(f"✓ Recognized: '{text}'")
            return text
            
        except sr.WaitTimeoutError:
            print("⏱️  Timeout: No speech detected")
            return None
        except sr.UnknownValueError:
            print("❌ Could not understand audio (speak louder/clearer)")
            return None
        except sr.RequestError as e:
            print(f"❌ Speech recognition service error: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None
    
    def start_continuous_listening(self, callback: Callable[[str], None], 
                                   use_wake_word=True):
        """
        Start continuous listening in background
        
        Args:
            callback: Function to call with recognized text
            use_wake_word: Require wake word before processing (default: True)
        """
        if self.listening:
            print("Already listening")
            return
        
        self.listening = True
        self.callback = callback
        self.use_wake_word = use_wake_word
        
        self.listen_thread = threading.Thread(
            target=self._continuous_listen_loop,
            daemon=True
        )
        self.listen_thread.start()
        print(f"✓ Started continuous listening (wake word: '{self.wake_word}')")
    
    def _continuous_listen_loop(self):
        """Internal continuous listening loop"""
        while self.listening:
            try:
                with self.microphone as source:
                    # Adjust for ambient noise periodically
                    if not hasattr(self, '_noise_adjusted'):
                        self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                        self._noise_adjusted = True
                    
                    # Listen with longer timeout
                    audio = self.recognizer.listen(source, timeout=2, phrase_time_limit=10)
                
                text = self.recognizer.recognize_google(audio).lower()
                print(f"[Heard: '{text}']")
                
                if self.use_wake_word:
                    # Check for wake word in the text
                    if self.wake_word in text:
                        print(f"✓ Wake word detected!")
                        
                        # Remove wake word and process the rest as the question
                        # Handle variations: "assistant what's the weather", "hey assistant what's the weather"
                        question = text
                        for variant in [f"hey {self.wake_word}", self.wake_word, "hey"]:
                            question = question.replace(variant, "").strip()
                        
                        # Remove leading punctuation/connecting words
                        question = question.lstrip(",.!? ")
                        
                        if question:
                            print(f"[Question: '{question}']")
                            if self.callback:
                                self.callback(question)
                        else:
                            # Just wake word, wait for follow-up
                            print("(Waiting for question...)")
                else:
                    # No wake word required, process everything
                    if self.callback:
                        self.callback(text)
                        
            except sr.WaitTimeoutError:
                # Normal timeout, continue listening
                continue
            except sr.UnknownValueError:
                # Could not understand, continue silently
                continue
            except sr.RequestError as e:
                print(f"❌ Speech recognition service error: {e}")
                time.sleep(1)
            except Exception as e:
                print(f"❌ Unexpected error in listening loop: {e}")
                time.sleep(1)
    
    def stop_listening(self):
        """Stop continuous listening (alias for stop_continuous_listening)"""
        self.stop_continuous_listening()
    
    def stop_continuous_listening(self):
        """Stop continuous listening"""
        if not self.listening:
            return
        
        self.listening = False
        if self.listen_thread:
            self.listen_thread.join(timeout=2.0)
        print("✓ Stopped continuous listening")
    
    def get_microphone_list(self):
        """List available microphones"""
        mic_list = sr.Microphone.list_microphone_names()
        print("Available microphones:")
        for i, name in enumerate(mic_list):
            print(f"  {i}: {name}")
        return mic_list
    
    def set_microphone(self, device_index):
        """Set specific microphone by index"""
        self.microphone = sr.Microphone(device_index=device_index)
        print(f"✓ Microphone set to device {device_index}")


def main():
    """Demo: Interactive speech interface"""
    print("Speech Interface Demo\n")
    
    # Initialize
    speech = SpeechInterface(wake_word="assistant")
    
    # Test TTS
    speech.speak("Hello! I am your voice assistant.")
    
    print("\n=== Interactive Mode ===")
    print(f"Say '{speech.wake_word}' followed by your question")
    print("Or press 'q' + Enter to quit\n")
    
    def handle_speech(text):
        """Handle recognized speech"""
        print(f"\n[User said]: {text}")
        
        # Simple responses
        if "hello" in text or "hi" in text:
            speech.speak("Hello! How can I help you?")
        elif "how are you" in text:
            speech.speak("I'm doing great, thank you for asking!")
        elif "what can you do" in text:
            speech.speak("I can see objects with my camera, answer questions, and provide haptic feedback.")
        elif "thank you" in text or "thanks" in text:
            speech.speak("You're welcome!")
        else:
            speech.speak(f"You said: {text}")
    
    # Start listening
    speech.start_continuous_listening(callback=handle_speech, use_wake_word=True)
    
    try:
        # Keep running until user quits
        while True:
            user_input = input()
            if user_input.lower() == 'q':
                break
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        speech.stop_continuous_listening()
        print("✓ Demo complete")


if __name__ == "__main__":
    main()

