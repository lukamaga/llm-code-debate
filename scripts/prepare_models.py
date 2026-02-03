#!/usr/bin/env python3
"""
Prepare Ollama models for the debate system.

This script:
1. Checks Ollama availability
2. Lists available models
3. Pulls recommended models if not present
4. Tests models with a simple prompt

Usage:
    python scripts/prepare_models.py
    python scripts/prepare_models.py --models qwen2.5-coder:7b deepseek-coder:6.7b
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm import OllamaClient, MultiModelClient


# Recommended models for code generation debates
RECOMMENDED_MODELS = [
    "qwen2.5-coder:7b",     # Best overall for code
    "deepseek-coder:6.7b",  # Strong code generation
    "codellama:7b",         # Meta's code model
    "mistral:7b",           # Good general model
    # "llama3:8b",          # Newer, but larger
]

# Minimum models needed
MINIMUM_MODELS = [
    "qwen2.5-coder:7b",
    "deepseek-coder:6.7b",
]


async def check_ollama(base_url: str) -> bool:
    """Check if Ollama is running."""
    client = OllamaClient(base_url=base_url, model="dummy")
    try:
        return await client.is_available()
    finally:
        await client.close()


async def list_models(base_url: str) -> list[str]:
    """List available models."""
    client = OllamaClient(base_url=base_url, model="dummy")
    try:
        return await client.list_models()
    finally:
        await client.close()


async def pull_model(base_url: str, model: str) -> bool:
    """Pull a model."""
    client = OllamaClient(base_url=base_url, model=model)
    try:
        print(f"   Pulling {model}... (this may take several minutes)")
        return await client.pull_model(model)
    finally:
        await client.close()


async def test_model(base_url: str, model: str) -> bool:
    """Test a model with a simple prompt."""
    from src.llm import LLMRequest
    
    client = OllamaClient(base_url=base_url, model=model)
    try:
        request = LLMRequest(
            prompt="Write a Python function that adds two numbers. Respond with only the code.",
            temperature=0.1,
            max_tokens=100,
        )
        response = await client.generate(request)
        return "def" in response.content and "return" in response.content
    except Exception as e:
        print(f"   Error testing {model}: {e}")
        return False
    finally:
        await client.close()


async def main():
    parser = argparse.ArgumentParser(description="Prepare Ollama models")
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="Ollama API URL",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MINIMUM_MODELS,
        help="Models to prepare",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Prepare all recommended models",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only test existing models, don't pull new ones",
    )
    
    args = parser.parse_args()
    
    models_to_prepare = RECOMMENDED_MODELS if args.all else args.models
    
    print("🤖 LLM Code Debate - Model Preparation")
    print("=" * 50)
    
    # Check Ollama
    print("\n1. Checking Ollama availability...")
    if not await check_ollama(args.base_url):
        print("   ❌ Ollama is not running!")
        print(f"   Please start Ollama:")
        print(f"      ollama serve")
        print(f"   Or check if it's running on a different URL.")
        sys.exit(1)
    print("   ✅ Ollama is running")
    
    # List existing models
    print("\n2. Checking existing models...")
    existing_models = await list_models(args.base_url)
    print(f"   Found {len(existing_models)} models:")
    for m in existing_models:
        print(f"      - {m}")
    
    # Determine which models to pull
    models_to_pull = []
    for model in models_to_prepare:
        # Check if model exists (handle tag variations)
        model_base = model.split(":")[0]
        exists = any(model_base in m for m in existing_models)
        if not exists:
            models_to_pull.append(model)
    
    # Pull missing models
    if models_to_pull and not args.test_only:
        print(f"\n3. Pulling {len(models_to_pull)} missing models...")
        for model in models_to_pull:
            success = await pull_model(args.base_url, model)
            if success:
                print(f"   ✅ {model} pulled successfully")
            else:
                print(f"   ❌ Failed to pull {model}")
    elif args.test_only:
        print("\n3. Skipping model pulling (test-only mode)")
    else:
        print("\n3. All required models are already available")
    
    # Test models
    print(f"\n4. Testing models...")
    available_for_debate = []
    
    # Refresh model list after pulling
    existing_models = await list_models(args.base_url)
    
    for model in models_to_prepare:
        model_base = model.split(":")[0]
        matching = [m for m in existing_models if model_base in m]
        
        if matching:
            actual_model = matching[0]
            print(f"   Testing {actual_model}...")
            success = await test_model(args.base_url, actual_model)
            if success:
                print(f"   ✅ {actual_model} works correctly")
                available_for_debate.append(actual_model)
            else:
                print(f"   ⚠️ {actual_model} returned unexpected output")
        else:
            print(f"   ⏭️ {model} not available, skipping test")
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    if len(available_for_debate) >= 2:
        print(f"✅ Ready for debate with {len(available_for_debate)} models:")
        for m in available_for_debate:
            print(f"   - {m}")
        print(f"\nRun a quick test:")
        print(f"   python scripts/quick_run.py")
    elif len(available_for_debate) == 1:
        print(f"⚠️ Only 1 model available. Need at least 2 for debate.")
        print(f"   Run: ollama pull deepseek-coder:6.7b")
    else:
        print(f"❌ No models available for debate.")
        print(f"   Run: ollama pull qwen2.5-coder:7b deepseek-coder:6.7b")


if __name__ == "__main__":
    asyncio.run(main())
