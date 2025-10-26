#!/usr/bin/env python3
"""
Continuous Voice Q&A - Always listening, wake word activated
"""

from multi_ai import MultiAI
from speech_interface import SpeechInterface
import time
import threading

# Global state
ai = None
speech = None
processing = False

def handle_speech(text):
    """Callback for when speech is detected"""
    global ai, speech, processing
    
    # Ignore if already processing
    if processing:
        print("⏳ Still processing previous request...")
        return
    
    processing = True
    
    try:
        print(f"\n📝 You said: '{text}'")
        
        # Speak acknowledgment
        speech.speak("Thinking")
        
        # Get AI response with SHORT prompt
        print("🤖 Getting AI response...")
        try:
            response = ai.ask_with_context(
                question=f"Answer this briefly in 1-2 sentences: {text}",
                image=None,
                depth_grid=None,
                use_history=True
            )
            
            if response:
                # Limit response length
                if len(response) > 200:
                    response = response[:200].rsplit('.', 1)[0] + '.'
                
                print(f"\n💬 AI: {response}\n")
                speech.speak(response)
            else:
                print("❌ No response from AI")
                speech.speak("I couldn't generate a response.")
                
        except Exception as e:
            print(f"❌ AI Error: {e}")
            speech.speak("Sorry, I encountered an error.")
    
    finally:
        processing = False
        print("\n✅ Ready for next question (say 'assistant' to activate)")


def main():
    global ai, speech
    
    print("=" * 70)
    print("CONTINUOUS VOICE ASSISTANT")
    print("=" * 70)
    print()
    
    # Initialize AI
    print("Initializing AI...")
    try:
        models_to_try = [
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "anthropic/claude-3-haiku",
        ]
        
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
    
    # Initialize Speech Interface
    print("\nInitializing speech interface...")
    try:
        speech = SpeechInterface()
        speech.set_microphone(device_index=1)  # USB microphone
        
        # Optimize for continuous listening
        speech.recognizer.pause_threshold = 0.8
        speech.recognizer.non_speaking_duration = 0.5
        speech.recognizer.dynamic_energy_threshold = True  # Adapt to noise
        
        print("✓ Speech interface ready")
    except Exception as e:
        print(f"❌ Speech initialization failed: {e}")
        return
    
    # Welcome
    speech.speak("Voice assistant ready. Say assistant to ask me anything.")
    
    print()
    print("=" * 70)
    print("🎤 CONTINUOUS LISTENING ACTIVE")
    print("=" * 70)
    print()
    print("Wake word: 'assistant' or 'hey assistant'")
    print("Example: 'Assistant, what is the weather today?'")
    print()
    print("Press Ctrl+C to quit")
    print()
    print("🎤 Listening for wake word...")
    print()
    
    try:
        # Start continuous listening with wake word
        speech.start_continuous_listening(
            callback=handle_speech,
            use_wake_word=True  # Will look for "assistant"
        )
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopping...")
        speech.stop_listening()
        speech.speak("Goodbye")
        print("✓ Voice assistant stopped")

if __name__ == "__main__":
    main()

