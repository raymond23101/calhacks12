#!/usr/bin/env python3
"""
Voice Q&A Demo - Ask questions using microphone, get spoken responses
"""

from multi_ai import MultiAI
from speech_interface import SpeechInterface
import time

def main():
    print("=" * 70)
    print("VOICE Q&A ASSISTANT")
    print("=" * 70)
    print()
    
    # Initialize AI
    print("Initializing AI...")
    try:
        # Try different models
        models_to_try = [
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "anthropic/claude-3-haiku",
        ]
        
        ai = None
        for model in models_to_try:
            try:
                print(f"  Trying model: {model}")
                ai = MultiAI(provider="openrouter", model=model)
                print(f"✓ AI ready (using {model})")
                break
            except Exception as e:
                print(f"  ✗ {model} failed: {str(e)[:50]}")
                continue
        
        if not ai:
            print("❌ No AI models available. Check your API keys.")
            return
            
    except Exception as e:
        print(f"❌ AI initialization failed: {e}")
        return
    
    # Initialize Speech Interface
    print("\nInitializing speech interface...")
    try:
        speech = SpeechInterface()
        # Set to USB microphone (device 1)
        speech.set_microphone(device_index=1)
        
        # Adjust recognizer for better pause detection
        speech.recognizer.pause_threshold = 0.8  # Seconds of silence to consider end of phrase
        speech.recognizer.non_speaking_duration = 0.5  # Seconds of silence before considering phrase complete
        
        print("✓ Speech interface ready (using USB Microphone)")
    except Exception as e:
        print(f"❌ Speech initialization failed: {e}")
        return
    
    # Welcome message
    speech.speak("Hello! I'm your voice assistant. Ask me anything!")
    
    print()
    print("=" * 70)
    print("READY - Listening for your questions")
    print("=" * 70)
    print()
    print("Instructions:")
    print("  1. Press Enter to start listening")
    print("  2. Speak your question clearly into the microphone")
    print("  3. The assistant will respond with speech")
    print("  4. Type 'q' to quit")
    print()
    
    conversation_history = []
    
    while True:
        # Wait for user to press Enter
        user_input = input("Press Enter to ask a question (or 'q' to quit): ").strip().lower()
        
        if user_input == 'q':
            print("\nExiting...")
            speech.speak("Goodbye!")
            break
        
        # Listen for question
        print("\n🎤 Listening... (speak now)")
        question = speech.listen_once(timeout=10)
        
        if not question:
            print("❌ No speech detected or couldn't understand")
            speech.speak("I didn't hear anything. Please try again.")
            continue
        
        print(f"\n📝 You asked: '{question}'")
        
        # Get AI response
        print("🤖 Thinking...")
        speech.speak("Let me think about that.")
        
        try:
            # Use the ask method with conversation history
            response = ai.ask_with_context(
                question=question,
                image=None,
                depth_grid=None,
                use_history=True
            )
            
            if response:
                print(f"\n💬 AI Response:\n{response}\n")
                
                # Speak the response
                print("🔊 Speaking response...")
                speech.speak(response)
                
                # Add to history
                conversation_history.append({
                    "question": question,
                    "response": response
                })
            else:
                print("❌ AI did not provide a response")
                speech.speak("I'm sorry, I couldn't generate a response.")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            speech.speak("Sorry, I encountered an error.")
        
        print("\n" + "-" * 70 + "\n")
    
    print("\n✓ Voice Q&A session ended")
    print(f"Total questions asked: {len(conversation_history)}")

if __name__ == "__main__":
    main()

