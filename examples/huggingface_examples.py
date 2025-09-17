"""
Examples using HuggingFace models with BMUq.

This script demonstrates how to use local HuggingFace models with CUDA support
for uncertainty quantification in mathematical reasoning.
"""

import torch
from bmuq.config import get_preset_config, BMUqConfig, LLMConfig
from bmuq.benchmarks import create_dataset_from_questions, BMUqBenchmark
from bmuq.models import list_recommended_models, get_model_memory_requirements

try:
    from bmuq.models import HuggingFaceLLM
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    print("HuggingFace transformers not installed. Install with:")
    print("pip install 'bmuq[huggingface]'")


def check_system_requirements():
    """Check system requirements for HuggingFace models."""
    print("System Requirements Check")
    print("=" * 40)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    Memory: {props.total_memory / 1e9:.1f} GB")
            print(f"    Compute Capability: {props.major}.{props.minor}")
    else:
        print("  Running on CPU (will be slower)")
    
    # Check PyTorch version
    print(f"PyTorch Version: {torch.__version__}")
    
    return cuda_available


def explore_recommended_models():
    """Explore recommended HuggingFace models for different use cases."""
    
    if not HUGGINGFACE_AVAILABLE:
        print("HuggingFace transformers not available. Skipping model exploration.")
        return
    
    print("\nRecommended HuggingFace Models")
    print("=" * 40)
    
    models = list_recommended_models()
    
    for category, model_dict in models.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for model_name, info in model_dict.items():
            print(f"  {model_name}")
            print(f"    Parameters: {info['params']}")
            print(f"    Use case: {info['use_case']}")
            
            # Get memory requirements
            memory_req = get_model_memory_requirements(model_name, "fp16")
            print(f"    GPU Memory (FP16): {memory_req['recommended_gpu_memory_gb']:.1f} GB")


def basic_huggingface_example():
    """Basic example using HuggingFace model."""
    
    if not HUGGINGFACE_AVAILABLE:
        print("Skipping HuggingFace example - transformers not installed")
        return
    
    print("\nBasic HuggingFace Example")
    print("=" * 40)
    
    # Use preset configuration
    config = get_preset_config("huggingface_local")
    
    # Override with a smaller model for demo
    config.llm.model = "microsoft/DialoGPT-small"  # Smaller for demo
    config.llm.extra_params["device"] = "cpu"  # Force CPU for demo
    
    print(f"Using model: {config.llm.model}")
    print("Loading model... (this may take a moment)")
    
    # Create simple dataset
    questions = [
        "What is 5 + 3?",
        "Solve for x: x + 2 = 7",
        "Calculate 10% of 50"
    ]
    answers = ["8", "5", "5"]
    
    dataset = create_dataset_from_questions(questions, answers, name="simple_math")
    
    try:
        # Run benchmark
        benchmark = BMUqBenchmark(config)
        results = benchmark.run(
            dataset=dataset,
            num_questions=len(questions),
            save_results=True,
            output_dir="huggingface_results"
        )
        
        print(f"\nResults:")
        print(f"  Success rate: {results.success_rate:.3f}")
        print(f"  Average confidence: {results.average_confidence:.3f}")
        
        # Show model usage stats
        if hasattr(results, 'llm_stats'):
            print(f"  Model usage:")
            for key, value in results.llm_stats.items():
                print(f"    {key}: {value}")
        
    except Exception as e:
        print(f"Error running benchmark: {e}")
        print("This might be due to model loading issues or insufficient memory")


def quantized_model_example():
    """Example using quantized model for memory efficiency."""
    
    if not HUGGINGFACE_AVAILABLE:
        print("Skipping quantized model example")
        return
    
    print("\nQuantized Model Example")
    print("=" * 40)
    
    # Check if we can use quantization
    try:
        import bitsandbytes
        quantization_available = True
    except ImportError:
        quantization_available = False
        print("bitsandbytes not installed. Install with: pip install bitsandbytes")
    
    if not quantization_available:
        print("Skipping quantization example")
        return
    
    # Use quantized preset
    config = get_preset_config("huggingface_quantized")
    
    # Use a smaller model for demo
    config.llm.model = "microsoft/DialoGPT-medium"
    config.llm.extra_params.update({
        "device": "auto",
        "load_in_8bit": True,
        "load_in_4bit": False
    })
    
    print(f"Using quantized model: {config.llm.model}")
    print("Model will be loaded in 8-bit precision for memory efficiency")
    
    # Simple test
    questions = ["What is 2 × 3?", "If y = 2x and x = 3, what is y?"]
    answers = ["6", "6"]
    
    dataset = create_dataset_from_questions(questions, answers)
    
    try:
        benchmark = BMUqBenchmark(config)
        results = benchmark.run(dataset, num_questions=2, save_results=False)
        
        print(f"\nQuantized Model Results:")
        print(f"  Success rate: {results.success_rate:.3f}")
        print(f"  Average confidence: {results.average_confidence:.3f}")
        
    except Exception as e:
        print(f"Error with quantized model: {e}")


def custom_huggingface_config():
    """Example of creating custom HuggingFace configuration."""
    
    print("\nCustom HuggingFace Configuration")
    print("=" * 40)
    
    # Create custom configuration
    custom_llm_config = LLMConfig(
        provider="huggingface",
        model="gpt2",  # Simple, widely available model
        temperature=0.6,
        max_tokens=150,
        extra_params={
            "device": "auto",
            "use_quantization": False,
            "load_in_8bit": False,
            "trust_remote_code": False,
            "cache_dir": "./model_cache"  # Custom cache directory
        }
    )
    
    config = BMUqConfig(
        llm=custom_llm_config,
        experiment_name="custom_huggingface",
        description="Custom HuggingFace configuration example"
    )
    
    # Validate configuration
    issues = config.validate()
    if issues:
        print("Configuration issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Configuration is valid!")
    
    # Show configuration details
    print(f"\nConfiguration Details:")
    print(f"  Provider: {config.llm.provider}")
    print(f"  Model: {config.llm.model}")
    print(f"  Device: {config.llm.extra_params.get('device', 'auto')}")
    print(f"  Quantization: {config.llm.extra_params.get('load_in_8bit', False)}")
    
    return config


def direct_model_usage():
    """Example of using HuggingFace model directly."""
    
    if not HUGGINGFACE_AVAILABLE:
        print("Skipping direct model usage example")
        return
    
    print("\nDirect HuggingFace Model Usage")
    print("=" * 40)
    
    try:
        # Initialize model directly
        llm = HuggingFaceLLM(
            model_name="gpt2",
            device="cpu",  # Use CPU for demo
            temperature=0.7,
            max_new_tokens=100
        )
        
        print(f"Model loaded: {llm.model_name}")
        print(f"Device: {llm.device}")
        
        # Get model info
        info = llm.get_model_info()
        print(f"\nModel Information:")
        print(f"  Vocab Size: {info.get('vocab_size', 'unknown')}")
        print(f"  Hidden Size: {info.get('hidden_size', 'unknown')}")
        print(f"  Parameters: {info.get('num_parameters', 'unknown')}")
        
        # Test generation
        print(f"\nTesting generation:")
        test_prompts = [
            "What is 4 + 4?",
            "Solve the equation: 2x = 10"
        ]
        
        for prompt in test_prompts:
            response = llm.generate(prompt, max_tokens=50)
            print(f"  Q: {prompt}")
            print(f"  A: {response[:100]}{'...' if len(response) > 100 else ''}")
            print()
        
        # Show usage statistics
        stats = llm.get_usage_stats()
        print(f"Usage Statistics:")
        print(f"  Total requests: {stats.total_requests}")
        print(f"  Total tokens: {stats.total_tokens}")
        print(f"  Estimated cost: ${stats.estimated_cost_usd:.4f}")
        
    except Exception as e:
        print(f"Error with direct model usage: {e}")


def memory_usage_monitoring():
    """Monitor GPU memory usage during model operations."""
    
    print("\nMemory Usage Monitoring")
    print("=" * 40)
    
    if not torch.cuda.is_available():
        print("CUDA not available - cannot monitor GPU memory")
        return
    
    if not HUGGINGFACE_AVAILABLE:
        print("HuggingFace not available")
        return
    
    try:
        print("Initial GPU memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        
        # Load model
        print(f"\nLoading model...")
        llm = HuggingFaceLLM(
            model_name="microsoft/DialoGPT-small",
            device="cuda",
            max_new_tokens=50
        )
        
        print("After model loading:")
        memory_info = llm.get_memory_usage()
        print(f"  Allocated: {memory_info['allocated_gb']:.2f} GB")
        print(f"  Reserved: {memory_info['reserved_gb']:.2f} GB")
        print(f"  Utilization: {memory_info['utilization']:.1%}")
        
        # Test generation
        print(f"\nGenerating responses...")
        for i in range(3):
            response = llm.generate(f"What is {i+1} + {i+2}?")
            memory_info = llm.get_memory_usage()
            print(f"  Generation {i+1} - Memory: {memory_info['allocated_gb']:.2f} GB")
        
        # Clean up
        llm.clear_cuda_cache()
        print(f"\nAfter cache clear:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        
    except Exception as e:
        print(f"Error monitoring memory: {e}")


def compare_model_sizes():
    """Compare different model sizes and their performance."""
    
    if not HUGGINGFACE_AVAILABLE:
        print("Skipping model comparison")
        return
    
    print("\nModel Size Comparison")
    print("=" * 40)
    
    models_to_test = [
        "gpt2",  # Small
        "microsoft/DialoGPT-small",  # Small conversational
    ]
    
    test_question = "What is 6 + 7?"
    
    for model_name in models_to_test:
        print(f"\nTesting {model_name}:")
        
        try:
            # Get memory requirements
            memory_req = get_model_memory_requirements(model_name, "fp16")
            print(f"  Estimated memory: {memory_req['total_memory_gb']:.1f} GB")
            
            # Test generation (CPU only to avoid GPU memory issues)
            llm = HuggingFaceLLM(
                model_name=model_name,
                device="cpu",
                max_new_tokens=30
            )
            
            import time
            start_time = time.time()
            response = llm.generate(test_question, max_tokens=30)
            generation_time = time.time() - start_time
            
            print(f"  Generation time: {generation_time:.2f} seconds")
            print(f"  Response: {response[:80]}{'...' if len(response) > 80 else ''}")
            
            # Clean up
            del llm
            
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    # Run all examples
    check_system_requirements()
    
    if HUGGINGFACE_AVAILABLE:
        explore_recommended_models()
        custom_huggingface_config()
        direct_model_usage()
        basic_huggingface_example()
        compare_model_sizes()
        
        # Only run GPU-specific examples if CUDA is available
        if torch.cuda.is_available():
            print("\n" + "="*60)
            print("CUDA-SPECIFIC EXAMPLES")
            print("="*60)
            memory_usage_monitoring()
            quantized_model_example()
        else:
            print("\n" + "="*60)
            print("CUDA not available - skipping GPU-specific examples")
            print("For GPU acceleration, ensure CUDA is properly installed")
    
    print("\n" + "="*60)
    print("HuggingFace Examples Completed!")
    print("\nInstallation notes:")
    print("- For GPU support: pip install 'bmuq[huggingface]'")
    print("- For CPU only: pip install 'bmuq[huggingface-cpu]'")
    print("- Ensure CUDA is installed for GPU acceleration")
    print("\nModel recommendations:")
    print("- Small/Fast: gpt2, microsoft/DialoGPT-small")
    print("- Medium/Balanced: microsoft/DialoGPT-medium")  
    print("- Large/Quality: microsoft/DialoGPT-large (requires GPU)")
    print("\nMemory optimization:")
    print("- Use 8-bit quantization: load_in_8bit=True")
    print("- Use 4-bit quantization: load_in_4bit=True (experimental)")
    print("- Monitor GPU memory with get_memory_usage()")