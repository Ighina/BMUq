"""
Example usage of the PRM (Process Reward Models) module.

This script demonstrates how to:
1. Initialize a PRM trainer with different configurations
2. Load the PRM800K dataset
3. Train a PRM model (with tokens or embeddings)
4. Evaluate and make predictions
"""

import os
from bmuq.prm import PRMTrainer, PRMFeaturizer
from bmuq.prm.train import PRMTrainingConfig


def example_token_based_training():
    """
    Example: Train a PRM using token-based features with a pre-trained BERT model.
    """
    print("=" * 80)
    print("Example 1: Token-based training with pre-trained BERT")
    print("=" * 80)

    # Configure training
    config = PRMTrainingConfig(
        model_name_or_path="bert-base-uncased",
        use_pretrained=True,
        featurizer_type="tokens",
        num_labels=3,
        output_dir="./prm_output_tokens_pretrained",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        max_length=512,
        logging_steps=100,
        eval_steps=500,
        save_steps=500,
    )

    # Initialize trainer
    trainer = PRMTrainer(config)

    # Load PRM800K dataset
    print("\nLoading PRM800K dataset...")
    datasets = trainer.load_prm800k_dataset()

    # Prepare datasets
    print("Preparing datasets...")
    train_dataset = trainer.prepare_dataset(datasets["train"])
    eval_dataset = trainer.prepare_dataset(datasets["test"])

    # Train
    print("\nStarting training...")
    trainer.train(train_dataset, eval_dataset)

    # Evaluate
    print("\nEvaluating model...")
    metrics = trainer.evaluate(eval_dataset)
    print(f"Evaluation metrics: {metrics}")

    # Make predictions
    print("\nMaking predictions on sample texts...")
    sample_texts = [
        "The answer is correct because 2 + 2 = 4.",
        "This step is neutral and doesn't contribute much.",
        "This reasoning is incorrect because 2 + 2 = 5.",
    ]
    predictions = trainer.predict(sample_texts)
    print(f"Predictions: {predictions}")


def example_token_based_training_random_init():
    """
    Example: Train a PRM using token-based features with a randomly initialized BERT model.
    """
    print("\n" + "=" * 80)
    print("Example 2: Token-based training with randomly initialized BERT")
    print("=" * 80)

    # Configure training
    config = PRMTrainingConfig(
        model_name_or_path="bert-base-uncased",
        use_pretrained=False,  # Random initialization
        featurizer_type="tokens",
        num_labels=3,
        output_dir="./prm_output_tokens_random",
        num_train_epochs=5,  # More epochs for random init
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,  # Higher learning rate for random init
        max_length=512,
    )

    # Initialize trainer
    trainer = PRMTrainer(config)

    # Load and prepare datasets
    print("\nLoading and preparing datasets...")
    datasets = trainer.load_prm800k_dataset()
    train_dataset = trainer.prepare_dataset(datasets["train"])
    eval_dataset = trainer.prepare_dataset(datasets["test"])

    # Train
    print("\nStarting training...")
    trainer.train(train_dataset, eval_dataset)

    # Evaluate
    print("\nEvaluating model...")
    metrics = trainer.evaluate(eval_dataset)
    print(f"Evaluation metrics: {metrics}")


def example_embedding_based_training():
    """
    Example: Train a PRM using sentence embeddings from sentence-transformers.
    """
    print("\n" + "=" * 80)
    print("Example 3: Embedding-based training with sentence-transformers")
    print("=" * 80)

    # Configure training
    config = PRMTrainingConfig(
        model_name_or_path="bert-base-uncased",  # Base BERT architecture
        use_pretrained=False,  # Random init (embeddings come from sentence-transformers)
        featurizer_type="embeddings",
        featurizer_model="all-MiniLM-L6-v2",  # Sentence-transformers model for embeddings
        num_labels=3,
        output_dir="./prm_output_embeddings",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=1e-4,
        max_length=512,
    )

    # Initialize trainer
    trainer = PRMTrainer(config)

    # Load and prepare datasets
    print("\nLoading and preparing datasets...")
    datasets = trainer.load_prm800k_dataset()
    train_dataset = trainer.prepare_dataset(datasets["train"])
    eval_dataset = trainer.prepare_dataset(datasets["test"])

    # Train
    print("\nStarting training...")
    trainer.train(train_dataset, eval_dataset)

    # Evaluate
    print("\nEvaluating model...")
    metrics = trainer.evaluate(eval_dataset)
    print(f"Evaluation metrics: {metrics}")


def example_custom_dataset():
    """
    Example: Train a PRM on a custom dataset.
    """
    print("\n" + "=" * 80)
    print("Example 4: Training on a custom dataset")
    print("=" * 80)

    from datasets import Dataset as HFDataset

    # Create a custom dataset
    custom_data = {
        "text": [
            "Step 1: Calculate 2 + 2 = 4. This is correct.",
            "Step 2: The reasoning here is unclear.",
            "Step 3: This claims 2 + 2 = 5, which is wrong.",
            "Step 4: Following the algorithm correctly.",
        ],
        "label": ["positive", "neutral", "negative", "positive"],
    }
    custom_dataset = HFDataset.from_dict(custom_data)

    # Configure training
    config = PRMTrainingConfig(
        model_name_or_path="bert-base-uncased",
        use_pretrained=True,
        featurizer_type="tokens",
        num_labels=3,
        output_dir="./prm_output_custom",
        num_train_epochs=10,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=2e-5,
    )

    # Initialize trainer
    trainer = PRMTrainer(config)

    # Prepare dataset
    print("\nPreparing custom dataset...")
    train_dataset = trainer.prepare_dataset(custom_dataset)

    # Train
    print("\nStarting training...")
    trainer.train(train_dataset)

    # Make predictions
    print("\nMaking predictions...")
    sample_texts = [
        "The calculation is correct.",
        "This step needs more explanation.",
        "The logic is flawed here.",
    ]
    predictions = trainer.predict(sample_texts)
    label_names = ["negative", "neutral", "positive"]
    for text, pred in zip(sample_texts, predictions):
        print(f"Text: {text}")
        print(f"Predicted label: {label_names[pred[0]]}\n")


def example_save_and_load():
    """
    Example: Save and load a trained PRM model.
    """
    print("\n" + "=" * 80)
    print("Example 5: Saving and loading a trained model")
    print("=" * 80)

    # Train a simple model
    config = PRMTrainingConfig(
        model_name_or_path="bert-base-uncased",
        use_pretrained=True,
        featurizer_type="tokens",
        output_dir="./prm_output_save_load",
        num_train_epochs=1,
        per_device_train_batch_size=8,
    )

    trainer = PRMTrainer(config)

    # Create a small dataset for quick training
    from datasets import Dataset as HFDataset
    small_data = {
        "text": ["Correct reasoning."] * 10,
        "label": ["positive"] * 10,
    }
    small_dataset = HFDataset.from_dict(small_data)
    train_dataset = trainer.prepare_dataset(small_dataset)

    print("\nTraining a small model...")
    trainer.train(train_dataset)

    # Save the model
    save_path = "./prm_saved_model"
    print(f"\nSaving model to {save_path}...")
    trainer.save_model(save_path)

    # Load the model
    print("\nLoading model from disk...")
    trainer.load_model(save_path)

    # Make a prediction to verify
    print("\nVerifying loaded model with prediction...")
    prediction = trainer.predict(["Test reasoning step."])
    print(f"Prediction: {prediction}")


if __name__ == "__main__":
    # Run examples (uncomment the ones you want to try)

    # Example 1: Standard token-based training with pre-trained BERT
    # example_token_based_training()

    # Example 2: Token-based training with random initialization
    # example_token_based_training_random_init()

    # Example 3: Embedding-based training with sentence-transformers
    # example_embedding_based_training()

    # Example 4: Training on custom dataset
    example_custom_dataset()

    # Example 5: Save and load model
    # example_save_and_load()

    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)
