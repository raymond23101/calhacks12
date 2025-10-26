# OAK-D Lite Vision Assistant

Complete vision assistant system with camera, AI, and voice interaction for accessibility and navigation.

## Features

### 🎥 Camera (`oak_camera.py`)
- **High FPS:** 60-70 FPS real-time performance
- **Optimized Depth Processing:**
  - Advanced noise reduction (temporal + spatial + speckle filtering)
  - Configurable depth range filtering (300mm - 5000mm)
  - Edge-preserving smoothing
  - High confidence thresholding
- **2x6 Depth Grid Analysis:** Real-time depth summation across grid regions
- **Thread-Safe API:** Background frame processing with mutex-protected access
- **Full Field of View:** ISP scaling preserves complete sensor view (no cropping)

### 🤖 AI Integration (`multi_ai.py`)
- **Multi-Provider Support:** Switch between OpenRouter, Claude, Gemini, or ChatGPT
- **Scene Understanding:** AI analyzes camera images with depth context
- **Object Detection:** Describes objects and their locations
- **Spatial Awareness:** Combines depth grid data with vision
- **Conversational Q&A:** Natural language interaction with history

### 🎙️ Voice Interface (`speech_interface.py`)
- **Text-to-Speech (TTS):** Google Text-to-Speech (gTTS) with pygame audio playback
- **Cross-Platform Audio:** Works on macOS, Linux, and Windows
- **Speech-to-Text (STT):** Listens to user through microphone
- **Wake Word Detection:** Activates on "assistant" or custom word
- **Continuous Listening:** Background voice interaction

## Requirements

```bash
# Camera
depthai>=2.24.0,<3.0.0
opencv-python>=4.8.0
numpy>=1.24.0
blobconverter>=1.2.0

# AI - Multi-provider support
anthropic>=0.18.0          # Claude (Anthropic)
openai>=1.12.0             # OpenAI (ChatGPT) and OpenRouter
google-generativeai>=0.3.0 # Google Gemini
python-dotenv>=1.0.0
Pillow>=10.0.0

# Speech
gTTS>=2.5.0                # Google Text-to-Speech
pygame>=2.6.0              # Audio playback
SpeechRecognition>=3.10.0
PyAudio>=0.2.13
```

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up API keys for AI providers
cp env.example .env
# Edit .env and set:
# - AI_PROVIDER (openrouter, claude, gemini, or openai)
# - Add your API key for the selected provider
```

## Quick Start

### Camera Only

```python
from oak_camera import OAKCamera
import time

# Initialize camera
cam = OAKCamera(
    res="480",           # Resolution: "480" | "720" | "800"
    median="5x5",        # Median filter: "OFF" | "3x3" | "5x5" | "7x7"
    min_depth=300,       # Min depth in mm (30cm)
    max_depth=5000,      # Max depth in mm (5m)
)

# Start device
cam.startDevice()
time.sleep(1.0)

# Get frames and depth data
frames = cam.getFrames(["rgb", "depth"])
rgb, depth = frames
grid = cam.getDepthGrid(rows=2, cols=6)

cam.stop()
```

### Speech Interface

```python
from speech_interface import SpeechInterface

# Initialize speech
speech = SpeechInterface(wake_word="assistant")

# Speak something
speech.speak("Hello! I am your vision assistant.")

# Listen once
text = speech.listen_once()
if text:
    print(f"You said: {text}")

# Continuous listening with callback
def handle_speech(text):
    speech.speak(f"You said: {text}")

speech.start_continuous_listening(callback=handle_speech)
# ... do other work ...
speech.stop_continuous_listening()
```

### AI Vision + Speech (Full System)

```python
from oak_camera import OAKCamera
from multi_ai import MultiAI
from speech_interface import SpeechInterface
import time

# Initialize all components
cam = OAKCamera(res="480")
cam.startDevice()
time.sleep(1)

# AI will use provider from .env (openrouter, claude, gemini, or openai)
ai = MultiAI()
speech = SpeechInterface()

# Capture and analyze scene
frames = cam.getFrames(["rgb"])
rgb = frames[0]
depth_grid = cam.getDepthGrid(2, 6)

# Get AI description
description = ai.describe_objects(rgb)
print(f"AI sees: {description}")

# Speak the description
speech.speak(description)

# Interactive Q&A
speech.speak("What would you like to know?")
question = speech.listen_once()

if question:
    answer = ai.ask_with_context(question, image=rgb, depth_grid=depth_grid)
    speech.speak(answer)

cam.stop()
```

### Switching AI Providers

```python
from multi_ai import MultiAI

# Use Claude (Anthropic)
ai = MultiAI(provider="claude")

# Use OpenAI (ChatGPT)
ai = MultiAI(provider="openai", model="gpt-4o")

# Use Google Gemini
ai = MultiAI(provider="gemini", model="gemini-1.5-flash")

# Use OpenRouter (access to many models)
ai = MultiAI(provider="openrouter", model="anthropic/claude-3.5-sonnet")

# Ask questions
response = ai.ask_with_context("What is computer vision?")
```

## API Reference

### OAKCamera Class

#### Constructor
```python
OAKCamera(res="480", median="7x7", lrcheck=True, extended=False, 
          subpixel=True, min_depth=300, max_depth=5000, DEBUG_MODE=True)
```

**Parameters:**
- `res`: Camera resolution - `"480"`, `"720"`, or `"800"`
- `median`: Median filter size - `"OFF"`, `"3x3"`, `"5x5"`, or `"7x7"`
- `lrcheck`: Enable left-right consistency check (reduces occlusion errors)
- `extended`: Enable extended disparity (closer minimum depth, doubles range)
- `subpixel`: Enable subpixel mode (better accuracy at distance)
- `min_depth`: Minimum valid depth in millimeters (default: 300mm / 30cm)
- `max_depth`: Maximum valid depth in millimeters (default: 5000mm / 5m)
- `DEBUG_MODE`: Enable debug output

#### Methods

**`startDevice()`**
Starts the camera device and begins background frame processing.

**`getFrames(names)`**
Thread-safe method to retrieve frames.
```python
frames = cam.getFrames(["rgb", "depth", "disparity", "left", "right"])
```
Returns list of frames in the same order as requested.

**`getDepth(x, y)`**
Get depth value at specific pixel coordinates.
```python
depth_cm = cam.getDepth(320, 240)  # Returns depth in centimeters
```

**`getDepthGrid(rows=2, cols=6)`**
Get summed depth values across a grid.
```python
grid = cam.getDepthGrid(rows=2, cols=6)  # Returns numpy array (2, 6)
```

**`halt()`**
Wait for any ongoing frame read operations to complete.

**`stop()`**
Stop the device and cleanup resources.

## Running the Demo

The `oak_camera.py` file includes a built-in demo that displays:
- RGB camera feed (left)
- Depth heatmap with 2x6 grid overlay (right)
- Real-time FPS counter
- Depth grid values printed to console

```bash
python oak_camera.py
```

**Controls:**
- `q` - Quit
- `d` - Toggle depth overlay on/off
- `g` - Toggle grid overlay on/off

## Performance

- **FPS:** 60-70 FPS on USB 3.0 (30-40 FPS on USB 2.0)
- **Resolution:** 640x480 default (configurable)
- **Latency:** <20ms end-to-end
- **Depth Range:** 30cm - 5m (configurable)

## Depth Processing Pipeline

The system applies advanced post-processing for clean, stable depth:

1. **Threshold Filter:** Removes readings outside min/max range
2. **Temporal Filter:** Smooths depth over time (alpha=0.4)
3. **Spatial Filter:** Edge-preserving smoothing (removes graininess)
4. **Speckle Filter:** Removes isolated noise pixels
5. **Confidence Threshold:** Only shows high-quality depth (>230/255)

## USB Speed Warning

The system automatically detects USB connection speed. For best performance, connect to a **USB 3.0** (blue) port. USB 2.0 connections will show a warning and may have reduced FPS.

## Test Utilities

### Check Environment Setup
```bash
python check_env.py
```
Verifies your `.env` file configuration and API keys.

### Test Speaker/TTS
```bash
python test_speaker.py
```
Tests text-to-speech with various phrases to verify audio output.

### Test USB Microphone
```bash
python test_usb_mic.py
```
Tests USB microphone input with speech recognition.

## Troubleshooting

**Camera not detected:**
- Ensure OAK-D Lite is plugged in
- Try different USB port (prefer USB 3.0)
- Check `lsusb` (Linux) or System Information (macOS)

**Low FPS:**
- Use USB 3.0 port instead of USB 2.0
- Reduce resolution (`res="480"`)
- Use lighter median filter (`median="3x3"`)

**Noisy/grainy depth:**
- Already optimized with temporal + spatial + speckle filtering
- Adjust `min_depth` / `max_depth` to focus on target range
- Ensure good lighting conditions

**Speech not working:**
- Run `python test_speaker.py` to verify audio output
- Check system volume and output device settings
- Ensure internet connection (gTTS requires online access)
- Test microphone with `python test_usb_mic.py`

## License

MIT License - See LICENSE file for details

## Credits

Built with [DepthAI](https://github.com/luxonis/depthai) SDK by Luxonis.
