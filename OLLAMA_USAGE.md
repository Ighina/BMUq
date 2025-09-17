# Ollama Integration with BMUq

BMUq now supports [Ollama](https://ollama.ai/) for running large language models locally. This integration allows you to use uncertainty quantification methods with local models, providing privacy, cost control, and customization benefits.

## Quick Start

### 1. Install and Setup Ollama

```bash
# Install Ollama (visit https://ollama.ai for platform-specific instructions)
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama server
ollama serve

# Pull a model (in another terminal)
ollama pull llama2
```

### 2. Configure BMUq for Ollama

```python
from bmuq.config.settings import BMUqConfig, LLMConfig

# Create configuration
config = BMUqConfig()
config.llm = LLMConfig(
    provider="ollama",
    model="llama2",
    temperature=0.7,
    ollama_base_url="http://localhost:11434",
    ollama_system_prompt="You are a helpful assistant that provides clear, step-by-step mathematical reasoning."
)
```

### 3. Run BMUq with Local Models

```python
from bmuq.benchmarks.benchmark import BMUqBenchmark

# Initialize and run benchmark
benchmark = BMUqBenchmark(config)
result = benchmark.run()
```

## Supported Models

BMUq's Ollama integration works with any model available through Ollama. Here are some recommended models:

### Code Models
- `codellama` - 7B parameters, great for code generation
- `codellama:13b` - Better quality, more memory required
- `deepseek-coder` - Advanced code understanding
- `starcoder` - Multi-language code generation

### Chat Models
- `llama2` - General purpose, good balance of speed/quality
- `llama2:13b` - Better reasoning, requires more memory
- `mistral` - Fast and efficient
- `mixtral` - High quality mixture of experts model
- `neural-chat` - Optimized for conversations

### Lightweight Models
- `tinyllama` - Very fast, 1.1B parameters
- `orca-mini` - Small but capable, 3B parameters
- `phi` - Microsoft's efficient 2.7B model

### Specialized Models
- `wizardmath` - Optimized for mathematical problems
- `llava` - Vision and language understanding
- `meditron` - Medical and healthcare applications

## Configuration Options

### Basic Configuration

```python
from bmuq.models.ollama_llm import OllamaLLM

llm = OllamaLLM(
    model="llama2",                           # Model name
    base_url="http://localhost:11434",       # Ollama server URL
    temperature=0.7,                         # Generation temperature
    timeout=120,                             # Request timeout in seconds
    system_prompt="Custom system prompt"     # Optional system prompt
)
```

### Environment Variables

You can configure Ollama through environment variables:

```bash
export BMUQ_LLM_PROVIDER=ollama
export BMUQ_LLM_MODEL=llama2
export BMUQ_LLM_OLLAMA_BASE_URL=http://localhost:11434
export BMUQ_LLM_OLLAMA_SYSTEM_PROMPT="Your custom prompt"
```

### Configuration File (YAML)

```yaml
llm:
  provider: ollama
  model: llama2
  temperature: 0.7
  max_tokens: 150
  ollama_base_url: http://localhost:11434
  ollama_system_prompt: "You are a helpful assistant..."

uncertainty:
  method: selfcheck

search:
  algorithm: tree_search
  beam_width: 3
```

## Usage Examples

### Basic Generation

```python
from bmuq.models.ollama_llm import OllamaLLM

llm = OllamaLLM(model="llama2")

# Generate response
response = llm.generate("What is 15 * 23?", max_tokens=100)
print(response)

# Check usage statistics
stats = llm.get_usage_stats()
print(f"Total requests: {stats.total_requests}")
print(f"Total tokens: {stats.total_tokens}")
```

### Streaming Generation

```python
# Stream response as it's generated
for chunk in llm.stream_generate("Explain quantum computing", max_tokens=200):
    print(chunk, end="", flush=True)
```

### Batch Generation

```python
prompts = [
    "What is 2 + 2?",
    "Explain photosynthesis",
    "Solve: x² + 5x + 6 = 0"
]

responses = llm.batch_generate(prompts, max_tokens=100)
for prompt, response in zip(prompts, responses):
    print(f"Q: {prompt}")
    print(f"A: {response}\n")
```

### Model Management

```python
# List available models
models = llm.list_available_models()
print("Available models:", [m['name'] for m in models])

# Get model information
info = llm.get_model_info()
print(f"Model: {info['model']}")
print(f"Provider: {info['provider']}")

# Health check
health = llm.health_check()
print(f"Server connected: {health['server_connected']}")
print(f"Model available: {health['model_available']}")
```

## System Requirements

Use the built-in utilities to check system requirements:

```python
from bmuq.models.ollama_llm import get_ollama_model_requirements

# Check requirements for a model
reqs = get_ollama_model_requirements("llama2:13b")
print(f"Minimum RAM: {reqs['minimum_ram_gb']} GB")
print(f"Recommended RAM: {reqs['recommended_ram_gb']} GB")
print(f"GPU recommended: {reqs['gpu_recommended']}")
```

## Running Examples

The repository includes comprehensive examples:

```bash
# Run the complete Ollama demo
python examples/ollama_examples.py

# Test just the imports and basic functionality
python test_ollama_import.py
```

## Comparison with Other Providers

| Feature | Ollama | OpenAI | HuggingFace |
|---------|--------|---------|-------------|
| Cost | Free (local compute) | Pay per token | Free (local compute) |
| Privacy | Complete privacy | Data sent to API | Complete privacy |
| Speed | Depends on hardware | Fast API | Depends on hardware |
| Models | Curated selection | Latest GPT models | Thousands available |
| Setup | Easy with Ollama CLI | API key needed | More complex setup |

## Troubleshooting

### Common Issues

**Connection Failed**
```
Error: Failed to connect to Ollama server at http://localhost:11434
```
- Ensure Ollama is running: `ollama serve`
- Check if port 11434 is available
- Try different base URL if needed

**Model Not Available**
```
Model 'llama2' not found locally
```
- Pull the model: `ollama pull llama2`
- Check available models: `ollama list`

**Memory Issues**
- Use smaller models (tinyllama, orca-mini)
- Close other applications
- Consider using quantized models

**Slow Generation**
- Use GPU if available
- Try smaller models
- Reduce max_tokens parameter

### Performance Tips

1. **Choose the right model size** for your hardware
2. **Use streaming** for better user experience with long responses
3. **Batch requests** when processing multiple prompts
4. **Monitor memory usage** and adjust model size accordingly
5. **Cache frequently used models** locally

## Integration with BMUq Features

### Uncertainty Quantification

All BMUq uncertainty methods work with Ollama models:

```python
config = BMUqConfig()
config.llm.provider = "ollama"
config.llm.model = "llama2"

# Use different uncertainty methods
config.uncertainty.method = "selfcheck"     # SelfCheck
config.uncertainty.method = "entropy_based" # Entropy-based
config.uncertainty.method = "consistency_based" # Consistency-based
```

### Search Algorithms

Tree search and beam search work with local models:

```python
config.search.algorithm = "tree_search"
config.search.beam_width = 3
config.search.max_depth = 8
```

### Benchmarking

Run benchmarks with local models:

```python
benchmark = BMUqBenchmark(config)
result = benchmark.run(num_questions=10)

# Compare multiple local models
results = benchmark.run_comparison([
    "selfcheck", "entropy_based", "consistency_based"
])
```

## Contributing

To extend Ollama support:

1. Models are defined in `bmuq/models/ollama_llm.py`
2. Configuration in `bmuq/config/settings.py`
3. Integration logic in `bmuq/benchmarks/benchmark.py`
4. Add new models to `list_popular_ollama_models()`
5. Update system requirements in `get_ollama_model_requirements()`

## Resources

- [Ollama Documentation](https://ollama.ai/docs)
- [Available Models](https://ollama.ai/library)
- [BMUq Documentation](README.md)
- [Configuration Guide](bmuq/config/README.md)