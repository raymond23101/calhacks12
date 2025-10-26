#!/usr/bin/env python3
"""Check .env configuration"""

from dotenv import load_dotenv
import os

load_dotenv()

print("="*60)
print("ENV CONFIGURATION CHECK")
print("="*60)
print()

# Check provider
provider = os.getenv("AI_PROVIDER")
print(f"AI_PROVIDER: {provider}")
print()

# Check API keys
keys = {
    "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
    "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
    "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
}

print("API Keys Status:")
for key, value in keys.items():
    if value and not value.startswith("your_"):
        # Mask the key for security
        masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
        print(f"  ✓ {key}: {masked}")
    else:
        print(f"  ✗ {key}: Not set")

print()
print("="*60)

# Check if correct key is set for provider
if provider == "openrouter" and keys["OPENROUTER_API_KEY"]:
    print("✓ Configuration looks good for OpenRouter!")
elif provider == "claude" and keys["ANTHROPIC_API_KEY"]:
    print("✓ Configuration looks good for Claude!")
elif provider == "gemini" and keys["GOOGLE_API_KEY"]:
    print("✓ Configuration looks good for Gemini!")
elif provider == "openai" and keys["OPENAI_API_KEY"]:
    print("✓ Configuration looks good for OpenAI!")
else:
    print(f"⚠️  Mismatch: AI_PROVIDER is '{provider}' but required API key may not be set")
    print(f"   Make sure the corresponding API key is configured in .env")

print("="*60)

