# BMUq: Bayesian Methods for Uncertainty Quantification in Large Language Models

A modular framework for implementing and evaluating uncertainty quantification methods in LLM-based reasoning, with a focus on SelfCheck and tree search approaches.

## Features

- **Multiple Uncertainty Quantification Methods**:
  - SelfCheck: Four-stage verification process from the paper
  - Entropy-based: Uncertainty from response diversity
  - Consistency-based: Logical and mathematical consistency checks
  - Random baseline: For comparison purposes

- **Advanced Search Algorithms**:
  - Tree search with uncertainty-guided exploration
  - Beam search with diversity penalties
  - Configurable depth, beam width, and pruning

- **Comprehensive Benchmarking**:
  - Built-in datasets (GSM8K, Math problems)
  - Custom dataset support (JSON/CSV)
  - Multiple evaluation metrics
  - Detailed result analysis

- **Flexible Configuration**:
  - Preset configurations for common scenarios
  - YAML/JSON configuration files
  - Environment variable support
  - Extensible architecture

## Quick Start

### Installation

```bash
git clone https://github.com/your-org/BMUq.git
cd BMUq

# Basic installation
pip install -e .

# With HuggingFace support (GPU)
pip install -e .[huggingface]

# With HuggingFace support (CPU only)
pip install -e .[huggingface-cpu]

# Install all optional dependencies
pip install -e .[all]
```

### Basic Usage

```python
from bmuq import BMUqBenchmark
from bmuq.config import get_preset_config
from bmuq.benchmarks import create_dataset_from_questions

# Create a simple dataset
questions = [
    "Solve for x: 2x + 3 = 7",
    "What is the area of a circle with radius 5?",
    "If John has 15 apples and gives away 6, how many are left?"
]
answers = ["2", "25π", "9"]

dataset = create_dataset_from_questions(questions, answers)

# Use preset configuration
config = get_preset_config("selfcheck_tree")
config.llm.provider = "openai"  # or "huggingface" for local models
config.llm.api_key = "your-api-key"  # Set your API key (OpenAI only)

# Run benchmark
benchmark = BMUqBenchmark(config)
results = benchmark.run(dataset)

print(f"Success rate: {results.success_rate:.3f}")
print(f"Average confidence: {results.average_confidence:.3f}")
```

### Command Line Usage

```bash
# Run with preset configuration
python scripts/run_experiments.py --preset selfcheck_tree

# Compare multiple uncertainty methods
python scripts/run_experiments.py --compare selfcheck entropy_based consistency_based

# Use custom dataset
python scripts/run_experiments.py --dataset custom --dataset-path data/my_questions.json

# List available options
python scripts/run_experiments.py --list
```

## Architecture

BMUq is organized into several key modules:

### Core Components
- **`bmuq.core`**: Data structures for reasoning steps and paths
- **`bmuq.models`**: LLM interfaces (OpenAI, HuggingFace, Mock, extensible)
- **`bmuq.uncertainty`**: Uncertainty quantification methods
- **`bmuq.search`**: Search algorithms for reasoning exploration

### Supporting Systems
- **`bmuq.config`**: Configuration management and presets
- **`bmuq.benchmarks`**: Evaluation framework and datasets
- **`bmuq.utils`**: Utility functions and helpers

## Uncertainty Quantification Methods

### SelfCheck
Four-stage verification process:
1. **Extract Target**: Identify what the step aims to achieve
2. **Collect Information**: Find dependencies from previous steps
3. **Regenerate Step**: Independently recreate the step
4. **Compare Results**: Assess consistency between original and regenerated steps

```python
from bmuq.uncertainty import SelfCheck
from bmuq.models import OpenAILLM

llm = OpenAILLM(api_key="your-key")
selfcheck = SelfCheck(llm, lambda_neg1=1.0, lambda_0=0.3)
```

### Entropy-Based
Measures uncertainty through response diversity:

```python
from bmuq.uncertainty import EntropyBasedUQ

entropy_uq = EntropyBasedUQ(llm, num_samples=5, temperature=0.8)
```

### Consistency-Based
Evaluates mathematical and logical consistency:

```python
from bmuq.uncertainty import ConsistencyBasedUQ

consistency_uq = ConsistencyBasedUQ(llm)
```

## Search Algorithms

### Tree Search
Explores reasoning paths with uncertainty-guided pruning:

```python
from bmuq.search import TreeSearchCoT

tree_search = TreeSearchCoT(
    llm=llm,
    uncertainty_method=selfcheck,
    beam_width=3,
    max_depth=8,
    confidence_threshold=0.2
)
```

### Beam Search
Maintains fixed number of best paths:

```python
from bmuq.search import BeamSearchCoT

beam_search = BeamSearchCoT(
    llm=llm,
    uncertainty_method=selfcheck,
    beam_width=3,
    diversity_penalty=0.1
)
```

### HuggingFace Models (Local Inference)

BMUq supports local HuggingFace models with CUDA acceleration:

```python
from bmuq.models import HuggingFaceLLM
from bmuq.config import get_preset_config

# Use HuggingFace preset
config = get_preset_config("huggingface_local")

# Or create custom HuggingFace configuration
from bmuq.config import LLMConfig

llm_config = LLMConfig(
    provider="huggingface",
    model="microsoft/DialoGPT-medium",
    temperature=0.7,
    extra_params={
        "device": "auto",  # Auto-detect CUDA
        "load_in_8bit": True,  # Memory-efficient quantization
        "max_new_tokens": 200
    }
)

# Direct model usage
hf_llm = HuggingFaceLLM(
    model_name="microsoft/DialoGPT-medium",
    device="cuda",  # or "cpu"
    temperature=0.7,
    load_in_8bit=True  # For memory efficiency
)

response = hf_llm.generate("Solve for x: 2x + 3 = 7")
print(response)

# Monitor GPU memory usage
memory_info = hf_llm.get_memory_usage()
print(f"GPU utilization: {memory_info['utilization']:.1%}")
```

**Recommended Models:**
- **Small/Fast**: `gpt2`, `microsoft/DialoGPT-small` (< 2GB GPU)
- **Medium/Balanced**: `microsoft/DialoGPT-medium` (< 4GB GPU) 
- **Large/Quality**: `microsoft/DialoGPT-large` (< 8GB GPU)

**Memory Optimization:**
- Use `load_in_8bit=True` for 8-bit quantization
- Use `load_in_4bit=True` for 4-bit quantization (experimental)
- Set `device="cpu"` for CPU-only inference
```

## Configuration

### Preset Configurations

BMUq includes several preset configurations:

```python
from bmuq.config import get_preset_config, list_available_presets

# List all presets
presets = list_available_presets()
for preset in presets:
    print(f"{preset['name']}: {preset['description']}")

# Use a preset
config = get_preset_config("selfcheck_tree")
```

Available presets:
- `default`: Basic SelfCheck with tree search
- `selfcheck_tree`: Balanced SelfCheck with tree search
- `selfcheck_beam`: Faster SelfCheck with beam search
- `baseline_comparison`: For comparing multiple methods
- `fast_development`: Quick setup with mock LLM
- `thorough_evaluation`: Comprehensive evaluation settings
- `math_problems`: Optimized for mathematical reasoning
- `huggingface_local`: Local HuggingFace model with CUDA support
- `huggingface_quantized`: Memory-efficient quantized HuggingFace model

### Custom Configuration

```python
from bmuq.config import BMUqConfig, LLMConfig, UncertaintyConfig, SearchConfig

config = BMUqConfig(
    llm=LLMConfig(
        provider="openai",
        model="gpt-4-turbo-preview",
        temperature=0.7
    ),
    uncertainty=UncertaintyConfig(
        method="selfcheck",
        lambda_neg1=1.2,
        lambda_0=0.3
    ),
    search=SearchConfig(
        algorithm="tree_search",
        beam_width=4,
        max_depth=10
    )
)
```

### Configuration Files

Save and load configurations in YAML or JSON:

```python
from bmuq.config import save_config, load_config

# Save configuration
save_config(config, "my_config.yaml")

# Load configuration
config = load_config("my_config.yaml")
```

## Datasets

### Built-in Datasets

```python
from bmuq.benchmarks import load_dataset

# Grade School Math 8K
gsm8k = load_dataset("gsm8k")

# Mathematical reasoning problems
math_dataset = load_dataset("math")
```

### Custom Datasets

```python
from bmuq.benchmarks import create_dataset_from_questions, CustomDataset

# From question-answer pairs
questions = ["What is 2+2?", "Solve: x+1=3"]
answers = ["4", "2"]
dataset = create_dataset_from_questions(questions, answers)

# From JSON/CSV file
dataset = load_dataset("custom", data_path="my_data.json")

# From structured data
data = [
    {"question": "What is 2+2?", "answer": "4", "category": "arithmetic"},
    {"question": "What is 3×3?", "answer": "9", "category": "arithmetic"}
]
dataset = CustomDataset(data, name="arithmetic_problems")
```

## Evaluation Metrics

BMUq provides comprehensive evaluation metrics:

- **Accuracy**: Fraction of correct answers
- **Confidence Correlation**: Correlation between confidence and correctness
- **Uncertainty Quality**: AUPR for uncertainty estimates
- **Path Diversity**: Diversity of reasoning approaches
- **Search Efficiency**: Accuracy per unit time

```python
# Run evaluation with specific metrics
config.benchmark.metrics = [
    "accuracy", 
    "confidence_correlation", 
    "uncertainty_quality"
]

results = benchmark.run(dataset)
for name, metric in results.metrics.items():
    print(f"{name}: {metric.value:.3f}")
```

## Examples

The `examples/` directory contains comprehensive usage examples:

- **`basic_usage.py`**: Getting started with BMUq
- **`custom_config.py`**: Configuration customization
- **`dataset_examples.py`**: Working with different datasets
- **`method_comparison.py`**: Comparing uncertainty methods

Run examples:

```bash
cd examples
python basic_usage.py
python custom_config.py
```

## Development

### Project Structure

```
BMUq/
├── bmuq/                   # Main package
│   ├── core/              # Data structures and interfaces
│   ├── models/            # LLM implementations
│   ├── uncertainty/       # Uncertainty quantification methods
│   ├── search/            # Search algorithms
│   ├── config/            # Configuration management
│   ├── benchmarks/        # Evaluation framework
│   └── utils/             # Utility functions
├── examples/              # Usage examples
├── scripts/               # Command-line tools
├── tests/                 # Test suite
└── docs/                  # Documentation
```

### Running Tests

```bash
pip install -e .[dev]
pytest tests/
```

### Adding New Methods

1. **Uncertainty Method**: Inherit from `UncertaintyMethod`
2. **Search Algorithm**: Inherit from `SearchAlgorithm`
3. **LLM Provider**: Inherit from `BaseLLM`
4. **Dataset**: Inherit from `Dataset`

See existing implementations for examples.

## Citation

If you use BMUq in your research, please cite:

```bibtex
@misc{bmuq2024,
  title={BMUq: Bayesian Methods for Uncertainty Quantification in Large Language Models},
  author={BMUq Contributors},
  year={2024},
  howpublished={\\url{https://github.com/your-org/BMUq}}
}
```

## License

MIT License. See `LICENSE` file for details.

## Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for guidelines.

## Support

- **Issues**: Report bugs and request features on GitHub
- **Discussions**: Join our community discussions
- **Documentation**: Full documentation at [docs link]

## Acknowledgments

This work builds upon the SelfCheck paper and incorporates ideas from the uncertainty quantification and reasoning communities. We thank all contributors and the broader research community for their insights and feedback.