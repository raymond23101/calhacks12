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
        Speak text through speaker
        
        Args:
            text: Text to speak
            blocking: Wait for speech to complete (default: True)
        """
        if blocking:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        else:
            # Non-blocking speech
            thread = threading.Thread(target=self._speak_async, args=(text,))
            thread.daemon = True
            thread.start()
    
    def _speak_async(self, text):
        """Internal method for non-blocking speech"""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
    
    def listen_once(self, timeout=5, phrase_time_limit=10) -> Optional[str]:
        """
        Listen for a single speech input
        
        Args:
            timeout: Max seconds to wait for speech to start
            phrase_time_limit: Max seconds for the phrase
            
        Returns:
            Recognized text or None if failed
        """
        try:
            with self.microphone as source:
                print("Listening...")
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )
            
            print("Processing speech...")
            text = self.recognizer.recognize_google(audio)
            print(f"Recognized: {text}")
            return text
            
        except sr.WaitTimeoutError:
            print("No speech detected (timeout)")
            return None
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
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
        wake_word_active = False
        
        while self.listening:
            try:
                with self.microphone as source:
                    # Short timeout for responsiveness
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=10)
                
                text = self.recognizer.recognize_google(audio).lower()
                
                if self.use_wake_word:
                    # Check for wake word
                    if self.wake_word in text:
                        wake_word_active = True
                        self.speak("Yes?", blocking=False)
                        print(f"Wake word detected: '{self.wake_word}'")
                        continue
                    
                    # Only process if wake word was said recently
                    if wake_word_active:
                        wake_word_active = False
                        if self.callback:
                            self.callback(text)
                else:
                    # No wake word required
                    if self.callback:
                        self.callback(text)
                        
            except sr.WaitTimeoutError:
                # Normal timeout, continue listening
                continue
            except sr.UnknownValueError:
                # Could not understand, continue
                continue
            except sr.RequestError as e:
                print(f"Speech recognition error: {e}")
                time.sleep(1)
            except Exception as e:
                print(f"Unexpected error in listening loop: {e}")
                time.sleep(1)
    
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

