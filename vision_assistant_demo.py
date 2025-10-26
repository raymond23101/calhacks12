#!/usr/bin/env python3
"""
Vision Assistant Demo with Live View
Shows RGB camera, depth heatmap, and AI status in real-time
"""

from oak_camera import OAKCamera
from multi_ai import MultiAI
from speech_interface import SpeechInterface
import time
import cv2
import numpy as np

def create_status_overlay(frame, status_text, color=(0, 255, 0)):
    """Add status text overlay to frame"""
    overlay = frame.copy()
    height, width = frame.shape[:2]
    
    # Add semi-transparent black bar at top
    cv2.rectangle(overlay, (0, 0), (width, 60), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    # Add status text
    cv2.putText(frame, status_text, (10, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return frame

def create_depth_heatmap(depth_grid):
    """Create a visual heatmap from depth grid (2x6)"""
    # Create a larger visualization (each cell is 100x80 pixels)
    cell_width = 100
    cell_height = 80
    rows, cols = depth_grid.shape
    heatmap = np.zeros((rows * cell_height, cols * cell_width, 3), dtype=np.uint8)
    
    # Normalize depth values for visualization
    valid_depths = depth_grid[depth_grid > 0]
    if len(valid_depths) > 0:
        min_val = valid_depths.min()
        max_val = valid_depths.max()
    else:
        min_val, max_val = 0, 1
    
    for i in range(rows):
        for j in range(cols):
            depth_val = depth_grid[i, j]
            
            # Normalize to 0-1 range
            if max_val > min_val and depth_val > 0:
                normalized = (depth_val - min_val) / (max_val - min_val)
            else:
                normalized = 0
            
            # Convert to color (blue=close, red=far)
            color_val = int(normalized * 255)
            color = cv2.applyColorMap(np.array([[color_val]], dtype=np.uint8), 
                                     cv2.COLORMAP_JET)[0][0]
            
            # Fill cell
            y1, y2 = i * cell_height, (i + 1) * cell_height
            x1, x2 = j * cell_width, (j + 1) * cell_width
            cv2.rectangle(heatmap, (x1, y1), (x2, y2), color.tolist(), -1)
            
            # Draw grid lines
            cv2.rectangle(heatmap, (x1, y1), (x2, y2), (255, 255, 255), 2)
            
            # Add depth value text (convert to millimeters)
            if depth_val > 0:
                text = f"{int(depth_val)}mm"
                cv2.putText(heatmap, text, (x1 + 10, y1 + 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return heatmap

def wrap_text(text, max_chars=70):
    """Wrap text into lines"""
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        test_line = current_line + word + " "
        if len(test_line) > max_chars:
            if current_line:
                lines.append(current_line.strip())
            current_line = word + " "
        else:
            current_line = test_line
    
    if current_line:
        lines.append(current_line.strip())
    
    return lines

def create_text_display(text, width=800, height=400):
    """Create a display for AI response text"""
    display = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add title
    cv2.putText(display, "AI RESPONSE:", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Wrap and display text
    lines = wrap_text(text, max_chars=80)
    y_offset = 70
    
    for line in lines[:15]:  # Max 15 lines
        cv2.putText(display, line, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
    
    return display

def main():
    print("="*70)
    print("VISION ASSISTANT - Live View with Object Detection")
    print("="*70)
    print()
    
    # Initialize camera
    print("Initializing camera...")
    cam = OAKCamera(res="480", min_depth=300, max_depth=5000, DEBUG_MODE=False)
    cam.startDevice()
    time.sleep(2)
    print("✓ Camera ready")
    
    # Initialize AI with OpenRouter + Vision Models
    print("Initializing AI (OpenRouter with vision model)...")
    ai_available = False
    try:
        # Try different vision-capable models available on OpenRouter
        # OpenAI models require credits but are most reliable
        models_to_try = [
            "openai/gpt-4o-mini",  # Fast, reliable, requires credits
            "openai/gpt-4o",  # High quality, requires credits
            "anthropic/claude-3-haiku",  # Good alternative
            "meta-llama/llama-3.2-90b-vision-instruct",  # Free but may be rate-limited
        ]
        
        ai = None
        for model in models_to_try:
            try:
                print(f"  Trying model: {model}")
                ai = MultiAI(provider="openrouter", model=model)
                print(f"✓ AI ready (using {model})")
                ai_available = True
                break
            except Exception as e:
                print(f"  ✗ {model} failed: {str(e)[:50]}")
                continue
        
        if not ai:
            raise Exception("No Gemini models available on OpenRouter")
    except Exception as e:
        print(f"❌ AI initialization failed: {e}")
        print("Continuing without AI - camera display only")
    
    # Initialize speech
    print("Initializing speech...")
    speech_available = False
    try:
        speech = SpeechInterface()
        speech.set_microphone(device_index=1)  # USB microphone
        print("✓ Speech ready")
        speech_available = True
        if ai_available:
            speech.speak("Vision assistant ready. Press space to analyze objects.")
    except Exception as e:
        print(f"⚠️  Speech initialization failed: {e}")
        print("Continuing without speech")
    
    print()
    print("="*70)
    print("CONTROLS:")
    print("  SPACE - Capture and analyze scene with AI")
    print("  'q'   - Quit")
    print("="*70)
    print()
    print("📺 Camera windows will open now...")
    print("   Look for 3 OpenCV windows:")
    print("   1. 'Vision Assistant - RGB Camera' (live video)")
    print("   2. 'Vision Assistant - Depth Heatmap (2x6 Grid)' (color depth)")
    print("   3. 'Vision Assistant - AI Response' (text window)")
    print()
    print("⚠️  If windows don't appear, check behind other windows!")
    print()
    
    # Test audio output
    if speech_available:
        print("🔊 Testing audio output...")
        speech.speak("Audio test")
        print("✓ Did you hear 'Audio test'? If not, check your volume!")
        print()
    
    last_response = "Waiting for analysis... (Press SPACE)"
    analyzing = False
    windows_created = False
    
    try:
        while True:
            # Get RGB frame and depth grid
            frames = cam.getFrames(["rgb"])
            rgb_frame = frames[0]
            
            if rgb_frame is None:
                print("❌ Failed to capture frame")
                time.sleep(0.1)
                continue
            
            depth_grid = cam.getDepthGrid(rows=2, cols=6)
            
            # Create depth heatmap visualization
            depth_viz = create_depth_heatmap(depth_grid)
            
            # Add status overlay to RGB
            if analyzing:
                status_text = "🤖 AI ANALYZING... Please wait"
                status_color = (0, 165, 255)  # Orange
            elif ai_available:
                status_text = "✓ READY - Press SPACE to analyze"
                status_color = (0, 255, 0)  # Green
            else:
                status_text = "⚠️  AI NOT AVAILABLE - Display only"
                status_color = (0, 255, 255)  # Yellow
            
            rgb_display = create_status_overlay(rgb_frame.copy(), status_text, status_color)
            
            # Create text display for AI response
            text_display = create_text_display(last_response)
            
            # Display all windows (MUST be called every frame to stay visible)
            cv2.imshow("Vision Assistant - RGB Camera", rgb_display)
            cv2.imshow("Vision Assistant - Depth Heatmap (2x6 Grid)", depth_viz)
            cv2.imshow("Vision Assistant - AI Response", text_display)
            
            # First time confirmation
            if not windows_created:
                print("✓ 3 camera windows now visible!")
                print("   (If you don't see them, check behind other windows)")
                print()
                windows_created = True
            
            # Position windows (only works on some systems)
            try:
                cv2.moveWindow("Vision Assistant - RGB Camera", 0, 0)
                cv2.moveWindow("Vision Assistant - Depth Heatmap (2x6 Grid)", 650, 0)
                cv2.moveWindow("Vision Assistant - AI Response", 0, 520)
            except:
                pass  # Window positioning may not work on all systems
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            
            elif key == ord(' ') and ai_available and not analyzing:
                print("\n" + "="*70)
                print("📸 CAPTURING AND ANALYZING SCENE...")
                print("="*70)
                analyzing = True
                
                # Update display immediately
                rgb_display = create_status_overlay(rgb_frame.copy(), 
                                                   "🤖 AI ANALYZING... Please wait", 
                                                   (0, 165, 255))
                cv2.imshow("Vision Assistant - RGB Camera", rgb_display)
                cv2.waitKey(1)
                
                if speech_available:
                    speech.speak("Analyzing scene")
                
                try:
                    # VERY short and concise prompt
                    prompt = "In ONE brief sentence, name the main objects you see and their positions (left/right/center)."
                    
                    print("🤖 Sending to AI via OpenRouter...")
                    description = ai.analyze_scene(
                        image=rgb_frame,
                        depth_grid=depth_grid,
                        prompt=prompt
                    )
                    
                    if description:
                        print(f"\n✓ AI Response:\n{description}\n")
                        last_response = description
                        
                        if speech_available:
                            print("🔊 Speaking results...")
                            
                            # Update display to show speaking status
                            temp_display = create_status_overlay(rgb_frame.copy(), 
                                                                 "🔊 SPEAKING RESULTS...", 
                                                                 (255, 0, 255))
                            cv2.imshow("Vision Assistant - RGB Camera", temp_display)
                            cv2.waitKey(1)
                            
                            # Speak the AI response
                            speech.speak(description)
                            print("✓ Speech completed")
                        else:
                            print("⚠️  Speech not available")
                    else:
                        print("❌ AI did not provide a description")
                        last_response = "No description available from AI."
                        if speech_available:
                            try:
                                speech.speak("I could not analyze the scene.")
                            except Exception as e:
                                print(f"⚠️  Speech failed: {e}")
                
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    print(f"❌ {error_msg}")
                    last_response = f"Error during analysis: {error_msg}"
                    if speech_available:
                        speech.speak("Sorry, I encountered an error.")
                
                analyzing = False
                print("="*70)
                print("Ready for next analysis (press SPACE)")
                print("="*70 + "\n")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        if speech_available:
            speech.speak("Vision assistant shutting down.")
        cam.stop()
        cv2.destroyAllWindows()
        print("✓ Done")


if __name__ == "__main__":
    main()
