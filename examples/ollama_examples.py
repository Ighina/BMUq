"""
Examples demonstrating Ollama integration with BMUq.

This script shows how to:
1. Set up Ollama models
2. Configure BMUq to use Ollama
3. Run uncertainty quantification with local models
4. Compare different local models

Prerequisites:
- Ollama installed and running (https://ollama.ai/)
- At least one model pulled (e.g., ollama pull llama2)
"""

import os
import sys
from pathlib import Path

# Add bmuq to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

from bmuq.config.settings import BMUqConfig, LLMConfig
from bmuq.benchmarks.benchmark import BMUqBenchmark
from bmuq.models.ollama_llm import OllamaLLM, list_popular_ollama_models, get_ollama_model_requirements

def test_ollama_connection():
    """Test basic Ollama connection and model availability."""
    print("=" * 60)
    print("Testing Ollama Connection")
    print("=" * 60)

    try:
        # Test with default settings
        llm = OllamaLLM(model="llama2")

        # Check health
        health = llm.health_check()
        print(f"Server connected: {health['server_connected']}")
        print(f"Model available: {health['model_available']}")
        print(f"Available models: {', '.join(health.get('available_models', []))}")

        if health['server_connected'] and health['model_available']:
            # Test basic generation
            print("\nTesting basic generation...")
            response = llm.generate("What is 2 + 2?", max_tokens=50)
            print(f"Response: {response}")

            # Show usage stats
            stats = llm.get_usage_stats()
            print(f"\nUsage stats:")
            print(f"  Requests: {stats.total_requests}")
            print(f"  Tokens: {stats.total_tokens}")
            print(f"  Estimated cost: ${stats.estimated_cost_usd:.6f}")

        return health['server_connected'] and health['model_available']

    except Exception as e:
        print(f"Error: {e}")
        return False


def show_popular_models():
    """Display information about popular Ollama models."""
    print("=" * 60)
    print("Popular Ollama Models")
    print("=" * 60)

    models = list_popular_ollama_models()

    for category, model_dict in models.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for model_name, info in model_dict.items():
            print(f"  {model_name:25} - {info['size']:>4} - {info['use_case']}")


def show_model_requirements():
    """Display system requirements for different model sizes."""
    print("=" * 60)
    print("Model System Requirements")
    print("=" * 60)

    test_models = ["llama2", "llama2:13b", "codellama", "mistral", "tinyllama"]

    for model in test_models:
        requirements = get_ollama_model_requirements(model)
        print(f"\n{model}:")
        print(f"  Model size: {requirements['model_size_b']:.1f}B parameters")
        print(f"  Min RAM: {requirements['minimum_ram_gb']:.1f} GB")
        print(f"  Recommended RAM: {requirements['recommended_ram_gb']:.1f} GB")
        print(f"  Disk space: {requirements['disk_space_gb']:.1f} GB")
        print(f"  CPU cores: {requirements['cpu_cores']}")
        if requirements['gpu_recommended']:
            print(f"  GPU recommended: Yes ({requirements.get('gpu_memory_gb', 'N/A')} GB VRAM)")
        else:
            print(f"  GPU recommended: No")


def test_different_models():
    """Test generation with different Ollama models."""
    print("=" * 60)
    print("Testing Different Models")
    print("=" * 60)

    # Models to test (only if available)
    models_to_test = ["llama2", "mistral", "codellama", "tinyllama"]

    test_prompt = "Solve this step by step: What is 15 * 23?"

    for model_name in models_to_test:
        print(f"\nTesting {model_name}...")
        try:
            llm = OllamaLLM(model=model_name)

            # Check if model is available
            health = llm.health_check()
            if not health['model_available']:
                print(f"  Model {model_name} not available locally")
                continue

            # Generate response
            response = llm.generate(test_prompt, max_tokens=150)
            print(f"  Response: {response[:200]}...")  # Truncate for display

            # Show model info
            info = llm.get_model_info()
            if 'model_details' in info:
                details = info['model_details']
                print(f"  Size: {details.get('size', 'unknown')} bytes")
                print(f"  Format: {details.get('format', 'unknown')}")

        except Exception as e:
            print(f"  Error testing {model_name}: {e}")


def demo_bmuq_with_ollama():
    """Demonstrate BMUq uncertainty quantification with Ollama."""
    print("=" * 60)
    print("BMUq with Ollama Demo")
    print("=" * 60)

    try:
        # Create configuration for Ollama
        config = BMUqConfig()
        config.llm = LLMConfig(
            provider="ollama",
            model="llama2",  # Change to your preferred model
            temperature=0.7,
            max_tokens=200,
            ollama_base_url="http://localhost:11434",
            ollama_system_prompt="You are a helpful assistant that provides step-by-step mathematical reasoning."
        )

        config.experiment_name = "ollama_demo"
        config.benchmark.num_questions = 3  # Small demo
        config.benchmark.verbose = True

        print("Configuration:")
        print(f"  Provider: {config.llm.provider}")
        print(f"  Model: {config.llm.model}")
        print(f"  Ollama URL: {config.llm.ollama_base_url}")

        # Initialize benchmark
        benchmark = BMUqBenchmark(config)

        # Test the LLM directly
        print(f"\nTesting LLM generation...")
        response = benchmark.llm.generate("What is 7 * 8?", max_tokens=100)
        print(f"Direct response: {response}")

        # Show LLM info
        llm_info = benchmark.llm.get_model_info()
        print(f"\nLLM Info:")
        print(f"  Provider: {llm_info['provider']}")
        print(f"  Model: {llm_info['model']}")
        print(f"  Available models: {len(llm_info.get('available_models', []))}")

        print("\nOllama integration test completed successfully!")

    except Exception as e:
        print(f"Error in BMUq demo: {e}")
        import traceback
        traceback.print_exc()


def test_streaming():
    """Test Ollama streaming generation."""
    print("=" * 60)
    print("Testing Streaming Generation")
    print("=" * 60)

    try:
        llm = OllamaLLM(model="llama2")

        print("Streaming response to: 'Explain how photosynthesis works'")
        print("Response: ", end="", flush=True)

        for chunk in llm.stream_generate("Explain how photosynthesis works", max_tokens=200):
            print(chunk, end="", flush=True)

        print("\n\nStreaming test completed!")

    except Exception as e:
        print(f"Streaming error: {e}")


def main():
    """Run all Ollama examples."""
    print("BMUq Ollama Integration Examples")
    print("=" * 60)
    print("This script demonstrates how to use Ollama models with BMUq.")
    print("Make sure Ollama is running and you have at least one model downloaded.")
    print("Example: ollama pull llama2")
    print()

    # Test connection first
    if not test_ollama_connection():
        print("\n❌ Ollama connection failed!")
        print("Please ensure:")
        print("1. Ollama is installed and running")
        print("2. At least one model is downloaded (e.g., 'ollama pull llama2')")
        print("3. Ollama server is accessible at http://localhost:11434")
        return

    print("\n✅ Ollama connection successful!")

    # Show available information
    show_popular_models()
    show_model_requirements()

    # Test different models
    test_different_models()

    # Test streaming
    test_streaming()

    # Demo BMUq integration
    demo_bmuq_with_ollama()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("\nNext steps:")
    print("1. Try different models with 'ollama pull <model_name>'")
    print("2. Adjust system prompts for your use case")
    print("3. Run full BMUq benchmarks with your local models")
    print("4. Compare local vs cloud model performance")


if __name__ == "__main__":
    main()