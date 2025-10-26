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
from motor_controller import MotorController
import cv2
import numpy as np
import time
import threading

# Global state
camera = None
ai = None
speech = None
motors = None
processing = False
last_analysis_time = 0
last_auto_announcement = 0
ANALYSIS_COOLDOWN = 3  # Seconds between auto-analyses
AUTO_ANNOUNCEMENT_INTERVAL = 5  # Seconds between automatic announcements

def clean_text_for_speech(text):
    """Remove lexicographic symbols and formatting from text for speech"""
    if text is None:
        return ""
    
    # Remove common symbols that shouldn't be spoken
    symbols_to_remove = ['*', '#', '_', '~', '`', '^', '{', '}', '[', ']', '<', '>', '|', '\\']
    for symbol in symbols_to_remove:
        text = text.replace(symbol, '')
    
    # Remove multiple spaces
    text = ' '.join(text.split())
    
    return text.strip()

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
            
            # Value text (in feet)
            text = f"{val:.1f}ft"
            cv2.putText(heatmap, text, (x1 + 10, y1 + 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return heatmap

def auto_announce_objects_async():
    """Asynchronously announce detected objects in background thread"""
    global camera, ai, speech, last_auto_announcement
    
    try:
        print("\n🔔 Auto-announcing objects...")
        
        # Capture current frame
        frames = camera.getFrames(["rgb"])
        rgb_frame = frames[0]
        
        if rgb_frame is not None:
            depth_grid = camera.getDepthGrid(rows=2, cols=6)
            
            # Get very concise object detection with distance (~25 words)
            response = ai.analyze_scene(
                image=rgb_frame,
                depth_grid=depth_grid,
                prompt="Identify important objects for navigation (people, obstacles, vehicles, animals, moving objects, hazards). For each object: name, position (left/center/right), distance in feet. Use depth grid values in FEET. Format: object position distance feet. No symbols, no numbering, plain text only. Try to limit the amount of objects to 3 or less, preferably less"
            )
            
            if response:
                clean_response = clean_text_for_speech(response)
                print(f"🔔 Auto-announcement: {clean_response}")
                speech.speak(clean_response)
                last_auto_announcement = time.time()
                
    except Exception as e:
        print(f"❌ Auto-announcement error: {e}")

def auto_announce_objects():
    """Check if it's time for auto-announcement and trigger async if needed"""
    global camera, ai, speech, processing, last_auto_announcement
    
    current_time = time.time()
    
    # Check if it's time for auto-announcement and not processing
    if (current_time - last_auto_announcement >= AUTO_ANNOUNCEMENT_INTERVAL and 
        not processing and camera is not None and ai is not None and speech is not None):
        
        # Update timestamp immediately to prevent multiple triggers
        last_auto_announcement = current_time
        
        # Run announcement in background thread (non-blocking)
        announcement_thread = threading.Thread(target=auto_announce_objects_async, daemon=True)
        announcement_thread.start()

def handle_voice_command_async(text):
    """Process voice command asynchronously"""
    global camera, ai, speech, processing, last_analysis_time
    
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
                
                # Answer the question directly, only describe scene if asked
                if wants_detail:
                    prompt = f"Answer this question: {text}. If the question asks about the scene or surroundings, describe navigation objects with positions and distances in feet. Keep it concise and to the point."
                else:
                    prompt = f"Answer this question directly and briefly: {text}. Only describe the scene if the question specifically asks about what you see or surroundings. Keep it concise and to the point."
                
                response = ai.analyze_scene(
                    image=rgb_frame,
                    depth_grid=depth_grid,
                    prompt=prompt
                )
                
                if response:
                    clean_response = clean_text_for_speech(response)
                    print(f"\n💬 AI: {clean_response}\n")
                    print(f"[DEBUG] About to speak: '{clean_response}'")
                    speech.speak(clean_response)
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
                question = f"{text}"
            else:
                question = f"Answer briefly and naturally: {text}"
            
            response = ai.ask_with_context(
                question=question,
                image=None,
                depth_grid=None,
                use_history=True
            )
            
            if response:
                clean_response = clean_text_for_speech(response)
                print(f"\n💬 AI: {clean_response}\n")
                print(f"[DEBUG] About to speak (no camera): '{clean_response}'")
                speech.speak(clean_response)
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

def handle_voice_command(text):
    """Handle voice command by starting async processing"""
    global processing
    
    if processing:
        print("⏳ Still processing previous request...")
        return
    
    processing = True
    
    # Run in background thread to avoid blocking camera loop
    command_thread = threading.Thread(target=handle_voice_command_async, args=(text,), daemon=True)
    command_thread.start()

def main():
    global camera, ai, speech, motors
    
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
        # Use the preset model configured in multi_ai.py (defaults to @preset/calhacks-12)
        ai = MultiAI(provider="openrouter")
        print(f"✓ AI ready (using {ai.model})")
        
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
    
    # Initialize Motor Controller
    print("\nInitializing haptic motors...")
    try:
        motors = MotorController(
            grid_shape=(2, 6),
            min_distance=1.6,  # 1.6 feet = max intensity (~0.5m)
            max_distance=10.0,  # 10 feet = no activation (~3m)
            area_threshold=0.2  # 20% coverage required
        )
        print("✓ Motor controller ready (simulation mode)")
    except Exception as e:
        print(f"❌ Motor initialization failed: {e}")
        print("  Continuing without haptic feedback...")
        motors = None
    
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
        if motors:
            print("🎮 Haptic motors active - intensity based on object proximity!")
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
    
    # Give the listening thread a moment to start
    time.sleep(1)
    print()
    print("=" * 70)
    print("🎤 LISTENING FOR 'ASSISTANT' - Speech will ONLY stop for this word")
    print("=" * 70)
    
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
                
                # Update haptic motors based on depth grid
                if motors is not None:
                    motors.update_from_depth_grid(depth_grid, verbose=False)
                
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
        else:
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
        if motors:
            motors.disable_all_motors()
            print("✓ Motors disabled")
        cv2.destroyAllWindows()
        speech.speak("Vision assistant shutting down")
        print("✓ Shutdown complete")

if __name__ == "__main__":
    main()

