#!/usr/bin/env python3
"""
Claude AI Integration
Uses Anthropic's Claude API for intelligent scene understanding and Q&A
"""

import os
import base64
import anthropic
from typing import Optional, Dict, Any
import cv2
import numpy as np
from dotenv import load_dotenv


class ClaudeVisionAI:
    """
    Claude AI integration for vision-based Q&A
    - Scene analysis from camera images
    - Object detection descriptions
    - Natural conversational Q&A
    """
    
    def __init__(self, api_key: Optional[str] = None, model="claude-3-5-sonnet-20241022"):
        """
        Initialize Claude AI client
        
        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            model: Claude model to use (default: claude-3-5-sonnet-20241022)
        """
        # Load API key from env if not provided
        if api_key is None:
            load_dotenv()
            api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not api_key:
            raise ValueError(
                "API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter"
            )
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.conversation_history = []
        
        print(f"✓ Claude AI initialized (model: {model})")
    
    def encode_image(self, image: np.ndarray, quality=85) -> str:
        """
        Encode OpenCV image to base64 for Claude
        
        Args:
            image: OpenCV image (BGR format)
            quality: JPEG quality 0-100
            
        Returns:
            Base64 encoded image string
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Encode as JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buffer = cv2.imencode('.jpg', image_rgb, encode_param)
        
        # Base64 encode
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64
    
    def analyze_scene(self, image: np.ndarray, depth_grid: Optional[np.ndarray] = None,
                     prompt: str = "Describe what you see in this image.") -> str:
        """
        Analyze a scene from camera image
        
        Args:
            image: OpenCV image from camera
            depth_grid: Optional depth grid data (2x6 array)
            prompt: Question or instruction for Claude
            
        Returns:
            Claude's response
        """
        # Encode image
        image_b64 = self.encode_image(image)
        
        # Build message with depth context if provided
        full_prompt = prompt
        if depth_grid is not None:
            depth_info = f"\n\nDepth grid (2 rows x 6 columns, values in mm):\n{depth_grid}"
            full_prompt += depth_info
        
        # Create message
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64
                        }
                    },
                    {
                        "type": "text",
                        "text": full_prompt
                    }
                ]
            }]
        )
        
        response = message.content[0].text
        return response
    
    def ask_with_context(self, question: str, image: Optional[np.ndarray] = None,
                        depth_grid: Optional[np.ndarray] = None,
                        use_history: bool = True) -> str:
        """
        Ask a question with optional visual context
        
        Args:
            question: User's question
            image: Optional camera image for visual context
            depth_grid: Optional depth grid data
            use_history: Include conversation history
            
        Returns:
            Claude's response
        """
        # Build message content
        content = []
        
        # Add image if provided
        if image is not None:
            image_b64 = self.encode_image(image)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_b64
                }
            })
        
        # Add depth context if provided
        if depth_grid is not None:
            depth_info = f"Depth grid (2x6, values in mm):\n{depth_grid}\n\n"
            question = depth_info + question
        
        # Add question
        content.append({
            "type": "text",
            "text": question
        })
        
        # Build messages with history if requested
        messages = []
        if use_history:
            messages.extend(self.conversation_history)
        
        messages.append({
            "role": "user",
            "content": content
        })
        
        # Get response
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=messages
        )
        
        response = message.content[0].text
        
        # Update history
        if use_history:
            self.conversation_history.append({
                "role": "user",
                "content": question  # Store text only in history
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })
        
        return response
    
    def describe_objects(self, image: np.ndarray) -> str:
        """
        Get detailed description of objects in the image
        
        Args:
            image: Camera image
            
        Returns:
            Description of detected objects
        """
        prompt = (
            "List all objects you can identify in this image. "
            "For each object, describe its location (left, center, right, "
            "top, bottom) and any relevant details."
        )
        return self.analyze_scene(image, prompt=prompt)
    
    def get_spatial_info(self, image: np.ndarray, depth_grid: np.ndarray,
                        object_name: str) -> str:
        """
        Get spatial information about a specific object
        
        Args:
            image: Camera image
            depth_grid: Depth grid data
            object_name: Name of object to locate
            
        Returns:
            Spatial description of the object
        """
        prompt = (
            f"Where is the {object_name} located in this image? "
            f"Describe its position and approximate distance based on the depth grid."
        )
        return self.analyze_scene(image, depth_grid, prompt)
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("✓ Conversation history cleared")


def main():
    """Demo: Claude AI with vision"""
    print("Claude AI Vision Demo\n")
    
    # Check for API key
    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set")
        print("Create a .env file with: ANTHROPIC_API_KEY=your_key_here")
        return
    
    # Initialize
    try:
        ai = ClaudeVisionAI()
    except Exception as e:
        print(f"Failed to initialize Claude AI: {e}")
        return
    
    # Test without image (text-only Q&A)
    print("\n=== Text-only Q&A ===")
    response = ai.ask_with_context("What is the OAK-D Lite camera?")
    print(f"Q: What is the OAK-D Lite camera?")
    print(f"A: {response}\n")
    
    # Test with camera if available
    print("\n=== Camera + Vision Q&A ===")
    print("Attempting to capture image from camera...")
    
    try:
        from oak_camera import OAKCamera
        import time
        
        # Initialize camera
        cam = OAKCamera(res="480", DEBUG_MODE=False)
        cam.startDevice()
        time.sleep(2)
        
        # Get frame and depth
        frames = cam.getFrames(["rgb"])
        rgb = frames[0]
        depth_grid = cam.getDepthGrid(2, 6)
        
        if rgb is not None:
            # Analyze scene
            print("\nAnalyzing scene...")
            description = ai.describe_objects(rgb)
            print(f"Scene: {description}\n")
            
            # Ask follow-up question with context
            print("Follow-up question...")
            response = ai.ask_with_context(
                "What is the closest object to the camera?",
                image=rgb,
                depth_grid=depth_grid
            )
            print(f"A: {response}")
        
        cam.stop()
        
    except ImportError:
        print("Camera not available for demo")
    except Exception as e:
        print(f"Camera demo error: {e}")
    
    print("\n✓ Demo complete")


if __name__ == "__main__":
    main()

