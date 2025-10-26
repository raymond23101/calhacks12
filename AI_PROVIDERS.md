# AI Provider Setup Guide

This project supports **4 AI providers** for vision and Q&A:
- **OpenRouter** - Access to 100+ models (recommended for flexibility)
- **Claude** (Anthropic) - Best for vision and reasoning
- **Gemini** (Google) - Fast and free tier available
- **ChatGPT** (OpenAI) - GPT-4o with vision

## Quick Setup

1. **Copy the environment template:**
   ```bash
   cp env.example .env
   ```

2. **Edit `.env` and set your provider:**
   ```bash
   # Choose your provider
   AI_PROVIDER=openrouter  # or "claude" | "gemini" | "openai"
   
   # Add the corresponding API key
   OPENROUTER_API_KEY=sk-or-v1-...
   ```

3. **Use in your code:**
   ```python
   from multi_ai import MultiAI
   
   # Will automatically use provider from .env
   ai = MultiAI()
   ```

## Provider Comparison

| Provider | Pros | Cons | Best For |
|----------|------|------|----------|
| **OpenRouter** | Access to 100+ models, one API key | Slight cost markup | Flexibility, trying different models |
| **Claude** | Excellent vision, great reasoning | Requires Anthropic account | Production, complex tasks |
| **Gemini** | Fast, generous free tier | Less sophisticated | Prototyping, high volume |
| **ChatGPT** | Good all-around, popular | Vision can be slower | General purpose |

## Getting API Keys

### OpenRouter (Recommended)
1. Visit: https://openrouter.ai/keys
2. Sign up with Google/GitHub
3. Create an API key
4. **Benefits:** One key for Claude, GPT, Gemini, and 100+ other models

**Available models on OpenRouter:**
- `anthropic/claude-3.5-sonnet` - Best overall
- `openai/gpt-4o` - OpenAI's best
- `google/gemini-flash-1.5` - Fast and cheap
- `meta-llama/llama-3.2-90b-vision` - Open source vision
- Many more at https://openrouter.ai/models

### Claude (Anthropic)
1. Visit: https://console.anthropic.com/
2. Sign up
3. Add credits (requires payment)
4. Create API key under "API Keys"

**Best models:**
- `claude-3-5-sonnet-20241022` - Latest, best for vision
- `claude-3-opus-20240229` - Most capable (more expensive)

### Gemini (Google)
1. Visit: https://makersuite.google.com/app/apikey
2. Sign in with Google account
3. Click "Get API Key"
4. **Free tier:** 15 requests/min, 1500 requests/day

**Best models:**
- `gemini-1.5-flash` - Fast, good for prototyping
- `gemini-1.5-pro` - More capable

### ChatGPT (OpenAI)
1. Visit: https://platform.openai.com/api-keys
2. Sign up
3. Add payment method
4. Create API key

**Best models:**
- `gpt-4o` - Best for vision (GPT-4 Omni)
- `gpt-4-turbo` - Faster, slightly cheaper

## Usage Examples

### Basic Usage (Uses .env settings)
```python
from multi_ai import MultiAI

# Automatically uses AI_PROVIDER from .env
ai = MultiAI()

# Text Q&A
response = ai.ask_with_context("What is depth sensing?")
print(response)
```

### Override Provider
```python
# Force specific provider
ai = MultiAI(provider="gemini")
```

### Custom Model
```python
# Use specific model
ai = MultiAI(provider="openrouter", model="anthropic/claude-3.5-sonnet")
```

### Vision with Camera
```python
from oak_camera import OAKCamera
from multi_ai import MultiAI
import time

# Initialize
cam = OAKCamera(res="480")
cam.startDevice()
time.sleep(1)

ai = MultiAI()  # Uses .env settings

# Capture and analyze
frames = cam.getFrames(["rgb"])
rgb = frames[0]
depth_grid = cam.getDepthGrid(2, 6)

# Ask about the scene
description = ai.describe_objects(rgb)
print(f"I see: {description}")

# Follow-up with depth context
distance = ai.ask_with_context(
    "What is the closest object?",
    image=rgb,
    depth_grid=depth_grid
)
print(distance)

cam.stop()
```

### Full System with Voice
```python
from oak_camera import OAKCamera
from multi_ai import MultiAI
from speech_interface import SpeechInterface
import time

# Initialize
cam = OAKCamera(res="480")
cam.startDevice()
time.sleep(1)

ai = MultiAI()
speech = SpeechInterface()

# Greet user
speech.speak("Vision assistant ready. What would you like to know?")

# Listen for question
question = speech.listen_once(timeout=10)

if question:
    # Capture scene
    frames = cam.getFrames(["rgb"])
    rgb = frames[0]
    depth_grid = cam.getDepthGrid(2, 6)
    
    # Get AI answer with visual context
    answer = ai.ask_with_context(question, image=rgb, depth_grid=depth_grid)
    
    # Speak answer
    speech.speak(answer)
    print(f"Q: {question}")
    print(f"A: {answer}")

cam.stop()
```

## Switching Providers

You can easily switch providers without changing code:

**Method 1: Edit .env file**
```bash
# .env
AI_PROVIDER=gemini  # Change this to switch providers
GOOGLE_API_KEY=your_key_here
```

**Method 2: Override in code**
```python
# Try different providers
providers = ["claude", "gemini", "openai"]

for provider in providers:
    ai = MultiAI(provider=provider)
    response = ai.ask_with_context("Describe the image", image=rgb)
    print(f"{provider}: {response}\n")
```

## Cost Comparison

| Provider | Input (per 1M tokens) | Output (per 1M tokens) | Vision (per image) |
|----------|----------------------|------------------------|-------------------|
| **Gemini Flash** | $0.075 | $0.30 | ~$0.0003 |
| **GPT-4o** | $2.50 | $10.00 | ~$0.0025 |
| **Claude Sonnet 3.5** | $3.00 | $15.00 | ~$0.0048 |
| **OpenRouter** | +10-30% markup | +10-30% markup | +10-30% markup |

**Note:** Prices are approximate and subject to change. OpenRouter adds a small markup but provides access to all models.

## Tips

1. **For prototyping:** Use Gemini (free tier) or OpenRouter with cheaper models
2. **For production:** Use Claude or GPT-4o directly for best quality
3. **For flexibility:** Use OpenRouter to easily switch between models
4. **For cost optimization:** Start with Gemini Flash, upgrade to Claude/GPT when needed

## Error Handling

```python
from multi_ai import MultiAI

try:
    ai = MultiAI(provider="claude")
    response = ai.ask_with_context("Hello!")
    print(response)
except ValueError as e:
    print(f"Configuration error: {e}")
    print("Make sure API key is set in .env file")
except Exception as e:
    print(f"API error: {e}")
```

## Environment File Template

Your `.env` should look like this:
```bash
# AI Provider Selection
AI_PROVIDER=claude

# OpenRouter (access to 100+ models)
OPENROUTER_API_KEY=sk-or-v1-...

# Claude (Anthropic)
ANTHROPIC_API_KEY=sk-ant-...

# Gemini (Google)
GOOGLE_API_KEY=AIza...

# ChatGPT (OpenAI)
OPENAI_API_KEY=sk-proj-...
```

Only the key for your selected `AI_PROVIDER` needs to be set.

