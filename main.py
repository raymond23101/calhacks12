#!/usr/bin/env python3
"""
Main Vision Assistant - Complete Integration
- Live camera feed with object detection
- Continuous voice interaction
- AI-powered responses
"""

from oak_camera import OAKCamera
from multi_ai import MultiAI
from speech_interface import SpeechInterface
import cv2
import numpy as np
import time
import threading

# Global state
camera = None
ai = None
speech = None
processing = False
last_analysis_time = 0
last_auto_announcement = 0
ANALYSIS_COOLDOWN = 3  # Seconds between auto-analyses
AUTO_ANNOUNCEMENT_INTERVAL = 5  # Seconds between automatic announcements

def create_status_overlay(frame, status_text, color=(0, 255, 0)):
    """Add status text overlay to frame"""
    overlay = frame.copy()
    height, width = frame.shape[:2]
    
    # Semi-transparent black bar at top
    cv2.rectangle(overlay, (0, 0), (width, 60), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    # Status text
    cv2.putText(frame, status_text, (10, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return frame

def create_depth_heatmap(depth_grid):
    """Create visual heatmap from 2x6 depth grid"""
    rows, cols = depth_grid.shape
    cell_height, cell_width = 100, 100
    
    heatmap = np.zeros((rows * cell_height, cols * cell_width, 3), dtype=np.uint8)
    
    # Normalize for visualization
    max_val = np.max(depth_grid) if np.max(depth_grid) > 0 else 1
    
    for i in range(rows):
        for j in range(cols):
            val = depth_grid[i, j]
            norm_val = int((val / max_val) * 255)
            
            # Color map: blue (far) -> green -> red (close)
            if norm_val < 128:
                b, g, r = 255 - norm_val * 2, norm_val * 2, 0
            else:
                b, g, r = 0, 255 - (norm_val - 128) * 2, (norm_val - 128) * 2
            
            y1, y2 = i * cell_height, (i + 1) * cell_height
            x1, x2 = j * cell_width, (j + 1) * cell_width
            cv2.rectangle(heatmap, (x1, y1), (x2, y2), (b, g, r), -1)
            
            # Grid lines
            cv2.rectangle(heatmap, (x1, y1), (x2, y2), (255, 255, 255), 2)
            
            # Value text
            text = f"{val/1000000:.1f}M"
            cv2.putText(heatmap, text, (x1 + 10, y1 + 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return heatmap

def auto_announce_objects():
    """Automatically announce detected objects every few seconds"""
    global camera, ai, speech, processing, last_auto_announcement
    
    current_time = time.time()
    
    # Check if it's time for auto-announcement and not processing
    if (current_time - last_auto_announcement >= AUTO_ANNOUNCEMENT_INTERVAL and 
        not processing and camera is not None and ai is not None and speech is not None):
        
        try:
            print("\n🔔 Auto-announcing objects...")
            
            # Capture current frame
            frames = camera.getFrames(["rgb"])
            rgb_frame = frames[0]
            
            if rgb_frame is not None:
                depth_grid = camera.getDepthGrid(rows=2, cols=6)
                
                # Get very concise object detection with distance
                response = ai.analyze_scene(
                    image=rgb_frame,
                    depth_grid=depth_grid,
                    prompt="Identify visible objects in the image. For each major object, list: {type of object}, {relative position in frame}, {distance} feet. Use format: 'object, position, X feet'. Do not number the objects. If image is unclear, describe what you can see. Keep under 25 words."
                )
                
                if response:
                    # Limit response length for auto-announcements
                    if len(response) > 50:
                        response = response[:50].rsplit('.', 1)[0] + '.'
                    
                    print(f"🔔 Auto-announcement: {response}")
                    speech.speak(response)
                    last_auto_announcement = current_time
                    
        except Exception as e:
            print(f"❌ Auto-announcement error: {e}")

def handle_voice_command(text):
    """Handle voice commands"""
    global camera, ai, speech, processing, last_analysis_time, wake_word_detected
    
    if processing:
        print("⏳ Still processing previous request...")
        return
    
    processing = True
    
    try:
        print(f"\n📝 You asked: '{text}'")
        
        # If camera is available, ALWAYS use vision for context
        if camera is not None:
            print("🎥 Using live camera feed...")
            speech.speak("Looking")
            
            # Capture current frame
            frames = camera.getFrames(["rgb"])
            rgb_frame = frames[0]
            
            if rgb_frame is not None:
                depth_grid = camera.getDepthGrid(rows=2, cols=6)
                
                print("🤖 Analyzing with vision context...")
                
                # Check if user wants detailed description
                detail_keywords = ['describe', 'detailed', 'tell me about', 'explain', 'what do you see']
                wants_detail = any(keyword in text.lower() for keyword in detail_keywords)
                
                # Always include object detection, location, and depth in responses
                if wants_detail:
                    prompt = f"Based on the image and depth data, provide a detailed description: {text}"
                else:
                    prompt = f"Based on the image and depth data, answer very briefly (under 15 words): {text}"
                
                response = ai.analyze_scene(
                    image=rgb_frame,
                    depth_grid=depth_grid,
                    prompt=prompt
                )
                
                if response:
                    # Limit response length
                    if len(response) > 200:
                        response = response[:200].rsplit('.', 1)[0] + '.'
                    
                    print(f"\n💬 AI: {response}\n")
                    print(f"[DEBUG] About to speak: '{response}'")
                    speech.speak(response)
                    print(f"[DEBUG] Speech call completed")
                else:
                    speech.speak("I couldn't analyze the scene")
            else:
                speech.speak("Camera feed not available")
        else:
            # No camera - general question without vision
            print("🤖 Thinking (no camera)...")
            speech.speak("Thinking")
            
            # Check if user wants detailed response
            detail_keywords = ['describe', 'detailed', 'tell me about', 'explain']
            wants_detail = any(keyword in text.lower() for keyword in detail_keywords)
            
            if wants_detail:
                question = f"Provide a detailed response: {text}"
            else:
                question = f"Answer very briefly (under 15 words): {text}"
            
            response = ai.ask_with_context(
                question=question,
                image=None,
                depth_grid=None,
                use_history=True
            )
            
            if response:
                # Limit response length
                if len(response) > 200:
                    response = response[:200].rsplit('.', 1)[0] + '.'
                
                print(f"\n💬 AI: {response}\n")
                print(f"[DEBUG] About to speak (no camera): '{response}'")
                speech.speak(response)
                print(f"[DEBUG] Speech call completed (no camera)")
            else:
                speech.speak("I couldn't generate a response")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        speech.speak("Sorry, I encountered an error")
    
    finally:
        processing = False
        last_auto_announcement = time.time()  # Reset timer to resume auto-announcements
        print("✅ Ready for next command")

def main():
    global camera, ai, speech
    
    print("=" * 70)
    print("COMPLETE VISION ASSISTANT")
    print("=" * 70)
    print()
    
    # Initialize Camera
    print("Initializing OAK-D Lite Camera...")
    camera_available = False
    try:
        camera = OAKCamera(res="480", median="5x5", min_depth=300, max_depth=5000)
        camera.startDevice()
        time.sleep(1)
        print("✓ Camera ready")
        camera_available = True
    except Exception as e:
        print(f"⚠️  Camera not available: {e}")
        print("   Continuing in voice-only mode...")
        camera = None
    
    # Initialize AI
    print("\nInitializing AI...")
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
            if camera:
                camera.stop()
            return
        
    except Exception as e:
        print(f"❌ AI initialization failed: {e}")
        if camera:
            camera.stop()
        return
    
    # Initialize Speech
    print("\nInitializing speech interface...")
    try:
        speech = SpeechInterface()
        speech.set_microphone(device_index=1)  # USB microphone
        
        # Optimize for continuous listening
        speech.recognizer.pause_threshold = 0.8
        speech.recognizer.non_speaking_duration = 0.5
        speech.recognizer.dynamic_energy_threshold = True
        
        print("✓ Speech interface ready")
    except Exception as e:
        print(f"❌ Speech initialization failed: {e}")
        if camera:
            camera.stop()
        return
    
    # Welcome
    speech.speak("Vision assistant ready. Say assistant to interact with me.")
    
    print()
    print("=" * 70)
    if camera_available:
        print("🎥 CAMERA ACTIVE (Vision ON) | 🎤 LISTENING FOR 'ASSISTANT'")
        print("=" * 70)
        print()
        print("📹 ALL questions will use live camera feed for context!")
        print("🔔 Objects will be automatically announced every 5 seconds!")
        print()
        print("Controls:")
        print("  - Say 'assistant' + your question")
        print("  - Press 'q' in camera window to quit")
        print()
        print("Examples:")
        print("  'Assistant, what do you see?' → 'Person, center, 4 feet. Cup, right, 2 feet'")
        print("  'Assistant, how many objects?' → 'Person, center, 4 feet. Table, left, 6 feet. Phone, right, 1 feet'")
        print("  'Assistant, describe what you see' → Detailed description")
        print("  'Assistant, what's closest?' → 'Phone, right, 1 feet'")
    else:
        print("🎤 VOICE MODE (No Camera) | LISTENING FOR 'ASSISTANT'")
        print("=" * 70)
        print()
        print("Controls:")
        print("  - Say 'assistant' + your question")
        print("  - Press Ctrl+C to quit")
        print()
        print("Examples:")
        print("  'Assistant, what is 2 plus 2?' → '4'")
        print("  'Assistant, tell me a joke' → Short joke")
        print("  'Assistant, describe quantum physics' → Detailed explanation")
    print()
    
    # Start continuous voice listening in background
    speech.start_continuous_listening(
        callback=handle_voice_command,
        use_wake_word=True
    )
    
    # Main loop - display camera feed or just keep listening
    try:
        if camera_available:
            # Camera mode - show visual feed
            while True:
                # Get camera frames
                frames = camera.getFrames(["rgb"])
                rgb_frame = frames[0]
                
                if rgb_frame is None:
                    print("❌ No frame from camera")
                    time.sleep(0.1)
                    continue
                
                # Get depth grid
                depth_grid = camera.getDepthGrid(rows=2, cols=6)
                
                # Check for automatic object announcements
                auto_announce_objects()
                
                # Create depth visualization
                depth_viz = create_depth_heatmap(depth_grid)
                
                # Add status overlay
                status = "🎤 Listening... (say 'assistant')" if not processing else "⏳ Processing..."
                rgb_display = create_status_overlay(rgb_frame, status, 
                                                   (0, 255, 0) if not processing else (0, 165, 255))
                
                # Display windows
                cv2.imshow("Vision Assistant - Camera", rgb_display)
                cv2.imshow("Vision Assistant - Depth (2x6 Grid)", depth_viz)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            # Voice-only mode - just keep alive
            print("Running in voice-only mode (no camera)")
            print("Press Ctrl+C to quit")
            while True:
                time.sleep(1)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupted by user")
    
    finally:
        print("\nCleaning up...")
        speech.stop_listening()
        if camera:
            camera.stop()
        cv2.destroyAllWindows()
        speech.speak("Vision assistant shutting down")
        print("✓ Shutdown complete")

if __name__ == "__main__":
    main()

