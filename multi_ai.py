#!/usr/bin/env python3
"""
Multi-Provider AI Integration
Supports: OpenRouter, Claude, Gemini, OpenAI (ChatGPT)
"""

import os
import base64
import cv2
import numpy as np
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv


class MultiAI:
    """
    Multi-provider AI for vision-based Q&A
    Supports: OpenRouter, Claude (Anthropic), Gemini (Google), OpenAI (ChatGPT)
    """
    
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize AI client with specified provider
        
        Args:
            provider: "openrouter" | "claude" | "gemini" | "openai" (or from .env AI_PROVIDER)
            model: Model name (optional, uses defaults)
        """
        load_dotenv()
        
        # Determine provider
        self.provider = provider or os.getenv("AI_PROVIDER", "claude")
        self.conversation_history = []
        
        # Initialize the appropriate client
        if self.provider == "openrouter":
            self._init_openrouter(model)
        elif self.provider == "claude":
            self._init_claude(model)
        elif self.provider == "gemini":
            self._init_gemini(model)
        elif self.provider == "openai":
            self._init_openai(model)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}. Use: openrouter, claude, gemini, openai")
        
        print(f"✓ AI initialized: {self.provider} ({self.model})")
    
    def _init_openrouter(self, model: Optional[str]):
        """Initialize OpenRouter (can access Claude, GPT, Gemini, etc.)"""
        import openai
        
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set in .env")
        
        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        
        # Default to Claude 3.5 Sonnet via OpenRouter
        self.model = model or "anthropic/claude-3.5-sonnet"
        self.max_tokens = 1024
    
    def _init_claude(self, model: Optional[str]):
        """Initialize Anthropic Claude"""
        import anthropic
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in .env")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model or "claude-3-5-sonnet-20241022"
        self.max_tokens = 1024
    
    def _init_gemini(self, model: Optional[str]):
        """Initialize Google Gemini"""
        import google.generativeai as genai
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set in .env")
        
        genai.configure(api_key=api_key)
        self.model = model or "gemini-1.5-flash"
        self.client = genai.GenerativeModel(self.model)
    
    def _init_openai(self, model: Optional[str]):
        """Initialize OpenAI (ChatGPT)"""
        import openai
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in .env")
        
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model or "gpt-4o"
        self.max_tokens = 1024
    
    def encode_image(self, image: np.ndarray, quality=85) -> str:
        """
        Encode OpenCV image to base64
        
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
            prompt: Question or instruction
            
        Returns:
            AI response
        """
        # Add depth context if provided
        full_prompt = prompt
        if depth_grid is not None:
            depth_info = f"\n\nDepth grid (2 rows x 6 columns, values in mm):\n{depth_grid}"
            full_prompt += depth_info
        
        # Route to appropriate provider
        if self.provider == "openrouter":
            return self._analyze_openrouter(image, full_prompt)
        elif self.provider == "claude":
            return self._analyze_claude(image, full_prompt)
        elif self.provider == "gemini":
            return self._analyze_gemini(image, full_prompt)
        elif self.provider == "openai":
            return self._analyze_openai(image, full_prompt)
    
    def _analyze_openrouter(self, image: np.ndarray, prompt: str) -> str:
        """OpenRouter vision analysis"""
        image_b64 = self.encode_image(image)
        
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            }]
        )
        
        return response.choices[0].message.content
    
    def _analyze_claude(self, image: np.ndarray, prompt: str) -> str:
        """Claude vision analysis"""
        image_b64 = self.encode_image(image)
        
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
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
                        "text": prompt
                    }
                ]
            }]
        )
        
        return message.content[0].text
    
    def _analyze_gemini(self, image: np.ndarray, prompt: str) -> str:
        """Gemini vision analysis"""
        from PIL import Image
        
        # Convert OpenCV image to PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Generate response
        response = self.client.generate_content([prompt, pil_image])
        return response.text
    
    def _analyze_openai(self, image: np.ndarray, prompt: str) -> str:
        """OpenAI vision analysis"""
        image_b64 = self.encode_image(image)
        
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            }]
        )
        
        return response.choices[0].message.content
    
    def ask_with_context(self, question: str, image: Optional[np.ndarray] = None,
                        depth_grid: Optional[np.ndarray] = None,
                        use_history: bool = True) -> str:
        """
        Ask a question with optional visual context
        
        Args:
            question: User's question
            image: Optional camera image for visual context
            depth_grid: Optional depth grid data
            use_history: Include conversation history (currently only for text)
            
        Returns:
            AI response
        """
        # Add depth context if provided
        if depth_grid is not None:
            depth_info = f"Depth grid (2x6, values in mm):\n{depth_grid}\n\n"
            question = depth_info + question
        
        # With image, use vision analysis
        if image is not None:
            return self.analyze_scene(image, None, question)
        
        # Text-only conversation
        if self.provider == "openrouter" or self.provider == "openai":
            # OpenRouter/OpenAI use OpenAI-compatible API
            messages = []
            if use_history:
                messages.extend(self.conversation_history)
            messages.append({"role": "user", "content": question})
            
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=messages
            )
            answer = response.choices[0].message.content
            
        elif self.provider == "claude":
            messages = []
            if use_history:
                messages.extend(self.conversation_history)
            messages.append({"role": "user", "content": question})
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=messages
            )
            answer = response.content[0].text
            
        elif self.provider == "gemini":
            # Gemini uses chat sessions for history
            if use_history and len(self.conversation_history) > 0:
                chat = self.client.start_chat(history=self.conversation_history)
                response = chat.send_message(question)
            else:
                response = self.client.generate_content(question)
            answer = response.text
        
        # Update history
        if use_history:
            if self.provider == "gemini":
                self.conversation_history.append({"role": "user", "parts": [question]})
                self.conversation_history.append({"role": "model", "parts": [answer]})
            else:
                self.conversation_history.append({"role": "user", "content": question})
                self.conversation_history.append({"role": "assistant", "content": answer})
        
        return answer
    
    def describe_objects(self, image: np.ndarray) -> str:
        """Get detailed description of objects in the image"""
        prompt = (
            "List all objects you can identify in this image. "
            "For each object, describe its location (left, center, right, "
            "top, bottom) and any relevant details."
        )
        return self.analyze_scene(image, prompt=prompt)
    
    def get_spatial_info(self, image: np.ndarray, depth_grid: np.ndarray,
                        object_name: str) -> str:
        """Get spatial information about a specific object"""
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
    """Demo: Multi-provider AI"""
    print("Multi-Provider AI Demo\n")
    
    # Check for API keys
    load_dotenv()
    provider = os.getenv("AI_PROVIDER", "claude")
    
    print(f"Selected provider: {provider}")
    print("Make sure you have set up your .env file with API keys\n")
    
    # Initialize
    try:
        ai = MultiAI(provider=provider)
    except Exception as e:
        print(f"Failed to initialize AI: {e}")
        print("\nMake sure to:")
        print("1. Copy env.example to .env")
        print("2. Add your API key for the selected provider")
        return
    
    # Test text-only Q&A
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

