# Process Reward Models (PRM) Module

This module provides tools for training and using Process Reward Models (PRMs) for uncertainty quantification in LLM reasoning. PRMs are models that evaluate individual reasoning steps as positive, neutral, or negative.

## Overview

The PRM module consists of three main components:

1. **`featurizer.py`**: Converts text into features suitable for model training
2. **`train.py`**: Implements the training pipeline for BERT-like models
3. **`example_usage.py`**: Demonstrates various usage patterns

## Features

- **Flexible Input Types**: Supports both token-based features (from pre-trained tokenizers) and sentence embeddings (from sentence-transformers)
- **Pre-trained or Random Initialization**: Train from pre-trained BERT models or initialize randomly
- **PRM800K Integration**: Built-in support for loading the standard PRM800K dataset
- **Comprehensive Training**: Full training pipeline with evaluation, checkpointing, and metrics
- **Sequence Classification**: Token-level classification for evaluating reasoning steps

## Installation

Required dependencies:

```bash
pip install torch transformers datasets sentence-transformers scikit-learn
```

## Quick Start

### 1. Token-Based Training (Pre-trained BERT)

```python
from bmuq.prm import PRMTrainer
from bmuq.prm.train import PRMTrainingConfig

# Configure training
config = PRMTrainingConfig(
    model_name_or_path="bert-base-uncased",
    use_pretrained=True,
    featurizer_type="tokens",
    num_labels=3,
    output_dir="./prm_output",
    num_train_epochs=3,
    per_device_train_batch_size=8,
)

# Initialize trainer
trainer = PRMTrainer(config)

# Load PRM800K dataset
datasets = trainer.load_prm800k_dataset()
train_dataset = trainer.prepare_dataset(datasets["train"])
eval_dataset = trainer.prepare_dataset(datasets["test"])

# Train
trainer.train(train_dataset, eval_dataset)

# Evaluate
metrics = trainer.evaluate(eval_dataset)
print(f"Metrics: {metrics}")
```

### 2. Embedding-Based Training (Sentence-Transformers)

```python
from bmuq.prm import PRMTrainer
from bmuq.prm.train import PRMTrainingConfig

# Configure training with embeddings
config = PRMTrainingConfig(
    model_name_or_path="bert-base-uncased",  # Base architecture
    use_pretrained=False,  # Random init (embeddings from sentence-transformers)
    featurizer_type="embeddings",
    featurizer_model="all-MiniLM-L6-v2",  # Sentence-transformers model
    num_labels=3,
    output_dir="./prm_output_embeddings",
    num_train_epochs=5,
)

# Initialize and train
trainer = PRMTrainer(config)
datasets = trainer.load_prm800k_dataset()
train_dataset = trainer.prepare_dataset(datasets["train"])
eval_dataset = trainer.prepare_dataset(datasets["test"])
trainer.train(train_dataset, eval_dataset)
```

### 3. Custom Dataset Training

```python
from datasets import Dataset
from bmuq.prm import PRMTrainer
from bmuq.prm.train import PRMTrainingConfig

# Create custom dataset
custom_data = {
    "text": [
        "Step 1: Calculate 2 + 2 = 4. This is correct.",
        "Step 2: The reasoning here is unclear.",
        "Step 3: This claims 2 + 2 = 5, which is wrong.",
    ],
    "label": ["positive", "neutral", "negative"],
}
custom_dataset = Dataset.from_dict(custom_data)

# Configure and train
config = PRMTrainingConfig(
    model_name_or_path="bert-base-uncased",
    output_dir="./prm_custom",
)
trainer = PRMTrainer(config)
train_dataset = trainer.prepare_dataset(custom_dataset)
trainer.train(train_dataset)
```

## Architecture

### Featurizer Types

#### 1. Token-Based Features (`featurizer_type='tokens'`)
- Uses a pre-trained tokenizer (e.g., BERT tokenizer)
- Converts text to token IDs, attention masks, and token type IDs
- Standard approach for fine-tuning pre-trained models

```python
from bmuq.prm import PRMFeaturizer

featurizer = PRMFeaturizer(
    featurizer_type='tokens',
    model_name_or_path='bert-base-uncased',
    max_length=512,
)

features = featurizer("This is a reasoning step.")
# Returns: {'input_ids': ..., 'attention_mask': ..., 'token_type_ids': ...}
```

#### 2. Embedding-Based Features (`featurizer_type='embeddings'`)
- Uses sentence-transformers to generate embeddings
- Bypasses the embedding layer of BERT
- Useful for custom embeddings or transfer learning

```python
from bmuq.prm import PRMFeaturizer

featurizer = PRMFeaturizer(
    featurizer_type='embeddings',
    model_name_or_path='all-MiniLM-L6-v2',
)

embeddings = featurizer("This is a reasoning step.")
# Returns: torch.Tensor of shape (embedding_dim,)
```

### Model Architecture

The module supports two model types:

1. **Standard BERT for Sequence Classification** (`featurizer_type='tokens'`)
   - Uses `AutoModelForSequenceClassification` from transformers
   - Can be initialized from pre-trained weights or randomly

2. **Custom BERT with Embedding Inputs** (`featurizer_type='embeddings'`)
   - Custom `BertForTokenClassificationWithEmbeddings` class
   - Accepts pre-computed embeddings via `inputs_embeds` parameter
   - Skips the embedding layer of BERT

## Configuration

### PRMTrainingConfig

Complete configuration options:

```python
from bmuq.prm.train import PRMTrainingConfig

config = PRMTrainingConfig(
    # Model configuration
    model_name_or_path="bert-base-uncased",  # Base model
    use_pretrained=True,  # Use pre-trained weights
    featurizer_type="tokens",  # 'tokens' or 'embeddings'
    featurizer_model=None,  # Optional separate featurizer model
    num_labels=3,  # negative, neutral, positive

    # Training parameters
    output_dir="./prm_output",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=500,

    # Logging and checkpointing
    logging_steps=100,
    eval_steps=500,
    save_steps=500,

    # Data parameters
    max_length=512,

    # Performance
    fp16=False,  # Mixed precision training
    gradient_accumulation_steps=1,
    dataloader_num_workers=4,

    # Reproducibility
    seed=42,
)
```

## PRM800K Dataset

The module includes built-in support for the PRM800K dataset from OpenAI:

```python
# Load the entire dataset
datasets = trainer.load_prm800k_dataset()
# Returns: {'train': Dataset, 'test': Dataset}

# Load a specific split
train_data = trainer.load_prm800k_dataset(split='train')
# Returns: {'train': Dataset}

# Stream the dataset (for large datasets)
datasets = trainer.load_prm800k_dataset(streaming=True)
```

### Dataset Format

The PRM800K dataset contains reasoning steps labeled as:
- **Positive**: Correct reasoning step
- **Neutral**: Neither clearly correct nor incorrect
- **Negative**: Incorrect reasoning step

### Custom Label Mapping

You can provide custom label mappings when preparing datasets:

```python
custom_label_map = {
    "correct": 0,
    "unclear": 1,
    "incorrect": 2,
}

train_dataset = trainer.prepare_dataset(
    datasets["train"],
    label_map=custom_label_map
)
```

## Training and Evaluation

### Training

```python
# Basic training
trainer.train(train_dataset)

# Training with evaluation
trainer.train(train_dataset, eval_dataset)
```

### Evaluation Metrics

The trainer computes the following metrics:
- **Accuracy**: Overall classification accuracy
- **F1 Score**: Weighted F1 score
- **Precision**: Weighted precision
- **Recall**: Weighted recall

```python
metrics = trainer.evaluate(eval_dataset)
print(metrics)
# {'accuracy': 0.85, 'f1': 0.84, 'precision': 0.86, 'recall': 0.85}
```

### Making Predictions

```python
sample_texts = [
    "The calculation is correct.",
    "This step is unclear.",
    "The logic is wrong here.",
]

predictions = trainer.predict(sample_texts)
# Returns: array of predicted labels (0, 1, or 2)

label_names = ["negative", "neutral", "positive"]
for text, pred in zip(sample_texts, predictions):
    print(f"{text} -> {label_names[pred[0]]}")
```

## Saving and Loading Models

### Saving

```python
# Save at the end of training (automatic)
trainer.train(train_dataset)  # Saves to config.output_dir/final_model

# Manual save
trainer.save_model("./my_prm_model")
```

### Loading

```python
# Load a saved model
trainer.load_model("./my_prm_model")

# Make predictions with loaded model
predictions = trainer.predict(["Test reasoning step."])
```

## Advanced Usage

### Random Initialization

Train a model from scratch without pre-trained weights:

```python
config = PRMTrainingConfig(
    model_name_or_path="bert-base-uncased",
    use_pretrained=False,  # Random initialization
    num_train_epochs=5,  # More epochs needed
    learning_rate=5e-5,  # Higher learning rate
)
```

### Mixed Precision Training

Enable FP16 for faster training on compatible GPUs:

```python
config = PRMTrainingConfig(
    fp16=True,
    per_device_train_batch_size=16,  # Can use larger batch size
)
```

### Gradient Accumulation

Simulate larger batch sizes:

```python
config = PRMTrainingConfig(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size = 16
)
```

## Integration with BMUq

The PRM module is designed to integrate with the BMUq uncertainty quantification framework:

```python
from bmuq.prm import PRMTrainer
from bmuq.uncertainty import UncertaintyMethod  # Future integration

# Train a PRM
prm_trainer = PRMTrainer(config)
prm_trainer.train(train_dataset)

# Use in uncertainty quantification
# uncertainty_method = PRMBasedUncertainty(prm_trainer.model)
# uncertainty_scores = uncertainty_method.quantify(reasoning_paths)
```

## Examples

See `example_usage.py` for complete working examples:

1. **Token-based training with pre-trained BERT**
2. **Token-based training with random initialization**
3. **Embedding-based training with sentence-transformers**
4. **Training on custom datasets**
5. **Saving and loading models**

Run examples:

```bash
python -m bmuq.prm.example_usage
```

## Best Practices

1. **Choose the Right Featurizer Type**:
   - Use `tokens` for standard fine-tuning of pre-trained models
   - Use `embeddings` when you have custom embeddings or want to leverage sentence-transformers

2. **Adjust Hyperparameters**:
   - Pre-trained models: Lower learning rate (2e-5), fewer epochs (3-5)
   - Random initialization: Higher learning rate (5e-5), more epochs (5-10)

3. **Dataset Size**:
   - Small datasets (<1000 examples): Use pre-trained models, more epochs
   - Large datasets (>10k examples): Can train from scratch if needed

4. **Evaluation Strategy**:
   - Always use a held-out test set
   - Monitor F1 score for imbalanced datasets
   - Use early stopping to prevent overfitting

5. **Hardware Considerations**:
   - Enable FP16 on modern GPUs for 2x speedup
   - Adjust batch size based on available memory
   - Use gradient accumulation if memory is limited

## Troubleshooting

### Out of Memory Errors
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Reduce `max_length`
- Enable `fp16=True`

### Poor Performance
- Increase `num_train_epochs`
- Adjust `learning_rate` (try 1e-5 to 5e-5)
- Check label distribution in dataset
- Ensure proper data preprocessing

### Slow Training
- Enable `fp16=True`
- Increase `per_device_train_batch_size`
- Reduce `logging_steps` and `eval_steps`
- Use fewer `dataloader_num_workers` if CPU-bound

## Citation

If you use the PRM800K dataset, please cite:

```bibtex
@article{lightman2023lets,
  title={Let's Verify Step by Step},
  author={Lightman, Hunter and Kosaraju, Vineet and Burda, Yura and Edwards, Harri and Baker, Bowen and Lee, Teddy and Leike, Jan and Schulman, John and Sutskever, Ilya and Cobbe, Karl},
  journal={arXiv preprint arXiv:2305.20050},
  year={2023}
}
```

## License

This module is part of the BMUq framework. See the main repository for license information.

## Contributing

Contributions are welcome! Please see the main BMUq repository for contribution guidelines.

## Future Work

- Support for other architectures (RoBERTa, DeBERTa, etc.)
- Multi-task learning with additional objectives
- Integration with active learning frameworks
- Support for few-shot learning scenarios
- Distributed training support
