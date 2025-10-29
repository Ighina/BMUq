"""
Training module for Process Reward Models.

This module provides functionality to train BERT-like models as Process Reward Models
using the PRM800K dataset or custom datasets. Supports both pre-trained and randomly
initialized models with token-based or embedding-based inputs.
"""

from typing import Optional, Dict, Any, Literal, List, Union
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    PreTrainedModel,
    BertConfig,
    ModernBertConfig,
    BertModel,
    ModernBertModel,
)
from datasets import load_dataset, Dataset as HFDataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .featurizer import PRMFeaturizer


@dataclass
class PRMCollator:
    """
    Collate function for PRM dataset that handles both token and embedding inputs.

    Args:
        pad_token_id: Token ID to use for padding (required for token featurization)
        featurizer_type: Either "tokens" or "embeddings"
        embedding_dim: Dimension of embeddings (required for embeddings featurization)
    """

    pad_token_id: Optional[int] = None
    featurizer_type: str = "tokens"
    embedding_dim: Optional[int] = None

    def __post_init__(self):
        if self.featurizer_type == "tokens" and self.pad_token_id is None:
            raise ValueError(
                "pad_token_id must be provided when featurizer_type='tokens'"
            )
        if self.featurizer_type == "embeddings" and self.embedding_dim is None:
            raise ValueError(
                "embedding_dim must be provided when featurizer_type='embeddings'"
            )
        if self.featurizer_type not in ["tokens", "embeddings"]:
            raise ValueError(
                f"featurizer_type must be 'tokens' or 'embeddings', got {self.featurizer_type}"
            )

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples.

        Args:
            batch: List of dictionaries from PRMDataset.__getitem__

        Returns:
            Dictionary with padded tensors
        """
        if self.featurizer_type == "tokens":
            return self._collate_tokens(batch)
        else:
            return self._collate_embeddings(batch)

    def _collate_tokens(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate token-based inputs.
        Pads input_ids with pad_token_id and labels with -100.
        """
        # Extract sequences
        input_ids_list = torch.concatenate([item["input_ids"] for item in batch])
        labels_list = torch.concatenate([item["labels"] for item in batch])

        # Pad input_ids with pad_token_id
        # input_ids_padded = pad_sequence(
        #     input_ids_list, batch_first=True, padding_value=self.pad_token_id
        # )

        # # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids_list != self.pad_token_id).long()

        # # Stack labels (they're already single values, not sequences)
        # labels_stacked = pad_sequence(labels_list, batch_first=True, padding_value=-100)

        result = {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask,
            "labels": labels_list,
        }

        # Include token_type_ids if present
        if "token_type_ids" in batch[0]:
            token_type_ids_list = [item["token_type_ids"] for item in batch]
            token_type_ids_padded = pad_sequence(
                token_type_ids_list, batch_first=True, padding_value=0
            )
            result["token_type_ids"] = token_type_ids_padded

        return result

    def _collate_embeddings(
        self, batch: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate embedding-based inputs.
        Pads embeddings with zeros and creates attention mask.
        """
        # Extract sequences
        embeddings_list = [item["inputs_embeds"] for item in batch]
        labels_list = [item["labels"] for item in batch]

        # Truncate sequences that are too long
        max_allowed_seq_len = 512
        embeddings_list = [
            emb[:max_allowed_seq_len] if emb.shape[0] > max_allowed_seq_len else emb
            for emb in embeddings_list
        ]

        max_seq_len = max(emb.shape[0] for emb in embeddings_list)
        batch_size = len(embeddings_list)

        # Initialize padded tensors
        embeddings_padded = torch.zeros(batch_size, max_seq_len, self.embedding_dim)
        attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long)

        # Fill in the actual embeddings and attention mask
        for i, emb in enumerate(embeddings_list):
            seq_len = emb.shape[0]
            embeddings_padded[i, :seq_len, :] = emb
            attention_mask[i, :seq_len] = 1

        # Stack labels (they're already single values, not sequences)
        labels_stacked = pad_sequence(labels_list, batch_first=True, padding_value=-100)

        return {
            "inputs_embeds": embeddings_padded,
            "attention_mask": attention_mask,
            "labels": labels_stacked,
        }


@dataclass
class PRMTrainingConfig:
    """Configuration for PRM training.

    Args:
        model_name_or_path: Name or path of the base model
        use_pretrained: Whether to use pre-trained weights (default: True)
        featurizer_type: Type of featurization ('tokens' or 'embeddings')
        featurizer_model: Model to use for featurization (if different from base model)
        num_labels: Number of output labels (default: 3 for negative/neutral/positive)
        output_dir: Directory to save model checkpoints
        num_train_epochs: Number of training epochs (default: 3)
        per_device_train_batch_size: Training batch size per device (default: 8)
        per_device_eval_batch_size: Evaluation batch size per device (default: 8)
        learning_rate: Learning rate (default: 2e-5)
        weight_decay: Weight decay (default: 0.01)
        warmup_steps: Number of warmup steps (default: 500)
        logging_steps: Number of steps between logging (default: 100)
        eval_steps: Number of steps between evaluations (default: 500)
        save_steps: Number of steps between checkpoints (default: 500)
        max_length: Maximum sequence length (default: 512)
        seed: Random seed for reproducibility (default: 42)
        fp16: Whether to use mixed precision training (default: False)
        gradient_accumulation_steps: Number of gradient accumulation steps (default: 1)
        dataloader_num_workers: Number of dataloader workers (default: 4)
    """

    model_name_or_path: str = "bert-base-uncased"
    use_pretrained: bool = True
    featurizer_type: Literal["tokens", "embeddings"] = "tokens"
    featurizer_model: Optional[str] = None
    num_labels: int = 3
    output_dir: str = "./prm_output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 5
    max_length: int = 512
    seed: int = 42
    fp16: bool = False
    gradient_accumulation_steps: int = 1
    dataloader_num_workers: int = 4


class BertForTokenClassificationWithEmbeddings(nn.Module):
    """
    BERT model for token-level classification that accepts pre-computed embeddings.

    This model is designed to work with sentence embeddings from sentence-transformers
    and skips the embedding layer of BERT.
    """

    def __init__(
        self, config: Union[BertConfig, ModernBertConfig], num_labels: int = 3
    ):
        super().__init__()
        self.num_labels = num_labels
        self.config = config

        # Initialize BERT encoder without embeddings
        if isinstance(config, ModernBertConfig):
            self.bert = ModernBertModel(config)
        else:
            self.bert = BertModel(config, add_pooling_layer=False)

        # Classification head
        try:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
        except AttributeError:
            self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        # Initialize weights
        self.classifier.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass using pre-computed embeddings.

        Args:
            inputs_embeds: Pre-computed embeddings of shape (batch_size, seq_length, hidden_size)
            attention_mask: Attention mask of shape (batch_size, seq_length)
            labels: Labels for computing the loss (batch_size, seq_length)

        Returns:
            Dictionary containing loss (if labels provided) and logits
        """
        # Pass through BERT encoder
        outputs = self.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        sequence_output = outputs[0]  # (batch_size, seq_length, hidden_size)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(
            sequence_output
        )  # (batch_size, seq_length, num_labels)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # Flatten for loss computation
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {
            "loss": loss,
            "logits": logits,
        }


class PRMDataset(Dataset):
    """
    PyTorch Dataset for Process Reward Model training.

    Args:
        data: HuggingFace dataset or list of examples
        featurizer: PRMFeaturizer instance for converting text to features
        label_map: Mapping from label strings to integers
    """

    def __init__(
        self,
        data: HFDataset,
        featurizer: PRMFeaturizer,
        label_map: Dict[str, int] = None,
    ):
        self.data = data
        self.featurizer = featurizer
        self.label_map = label_map or {"negative": 0, "neutral": 1, "positive": 2}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example from the dataset.

        Returns:
            Dictionary containing features and labels
        """
        example = self.data[idx]["labeled_steps"]

        # Extract text and label
        text = example.get("steps", example.get("input", ""))
        label = example.get("labels", example.get("rating", "neutral"))

        # Convert label to integer
        if isinstance(label, str):
            label = self.label_map.get(label, 1)  # Default to neutral

        # Featurize the text
        if self.featurizer.featurizer_type == "tokens":
            features = self.featurizer(text, return_tensors="pt")
            # Remove batch dimension
            features = {k: v.squeeze(0) for k, v in features.items()}
            features["labels"] = torch.tensor(label, dtype=torch.long)
        else:  # embeddings
            embeddings = self.featurizer(text)

            # CRITICAL: Ensure embeddings are always 2D [seq_len, embedding_dim]
            if embeddings.dim() == 1:
                embeddings = embeddings.unsqueeze(0)  # Add sequence dimension
            elif embeddings.dim() == 3:
                embeddings = embeddings.squeeze(0)  # Remove batch dimension

            features = {
                "inputs_embeds": (embeddings),
                "labels": torch.tensor(label, dtype=torch.long),
            }

        return features


class PRMTrainer:
    """
    Trainer for Process Reward Models.

    This class handles the complete training pipeline including:
    - Model initialization (pre-trained or random)
    - Dataset loading and preprocessing
    - Training with transformers Trainer
    - Evaluation and metrics computation

    Args:
        config: PRMTrainingConfig instance with training configuration
    """

    def __init__(self, config: PRMTrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize featurizer
        featurizer_model = config.featurizer_model or config.model_name_or_path
        self.featurizer = PRMFeaturizer(
            featurizer_type=config.featurizer_type,
            model_name_or_path=featurizer_model,
            max_length=config.max_length,
        )

        # Initialize model
        self.model = self._initialize_model()

        # Training arguments
        self.training_args = TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_steps=config.warmup_steps,
            logging_dir=os.path.join(config.output_dir, "logs"),
            logging_steps=config.logging_steps,
            eval_steps=config.eval_steps,
            save_steps=config.save_steps,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            seed=config.seed,
            fp16=config.fp16,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            dataloader_num_workers=config.dataloader_num_workers,
            remove_unused_columns=False,  # Important for custom datasets
        )

    def _initialize_model(self) -> PreTrainedModel:
        """
        Initialize the PRM model (pre-trained or random).

        Returns:
            Initialized model
        """
        if self.config.featurizer_type == "tokens":
            if self.config.use_pretrained:
                # Load pre-trained model
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.model_name_or_path,
                    num_labels=self.config.num_labels,
                )
            else:
                # Initialize with random weights
                config = AutoConfig.from_pretrained(
                    self.config.model_name_or_path,
                    num_labels=self.config.num_labels,
                )
                model = AutoModelForSequenceClassification.from_config(config)

        else:  # embeddings
            # For embeddings, we need a custom model that accepts inputs_embeds
            if self.config.use_pretrained:
                if isinstance(
                    AutoConfig.from_pretrained(self.config.model_name_or_path),
                    ModernBertConfig,
                ):
                    config = ModernBertConfig.from_pretrained(
                        self.config.model_name_or_path,
                        num_labels=self.config.num_labels,
                    )
                else:
                    config = BertConfig.from_pretrained(self.config.model_name_or_path)
            else:
                if self.config.model_name_or_path.startswith("modern-bert"):
                    config = ModernBertConfig()  # Use default config for ModernBert
                else:
                    config = BertConfig(
                        vocab_size=30522,  # Default BERT vocab size
                        hidden_size=768,
                        num_hidden_layers=12,
                        num_attention_heads=12,
                        intermediate_size=3072,
                    )

            # Set hidden_size to match embedding dimension if available
            embedding_dim = self.featurizer.get_embedding_dim()
            if embedding_dim is not None:
                config.hidden_size = embedding_dim

            model = BertForTokenClassificationWithEmbeddings(
                config, self.config.num_labels
            )

        return model.to(self.device)

    def _process_prm800k_labels_batch(
        self,
        problems: list,
        labels_batch: list,
    ) -> dict:
        """
        Process a batch of labels.

        Args:
            problems: List of problem dicts
            labels_batch: List of label dicts
        Returns:
            Dict with lists of steps and labels for each example
        """
        label_mapping = {-1: 0, 0: 1, 1: 2}

        all_labeled_steps = []

        for problem, labels in zip(problems, labels_batch):
            steps = [problem["problem"]]
            example_labels = [-100]

            for label in labels["steps"]:
                for step in label["completions"]:
                    mapped_label = label_mapping.get(step["rating"], 1)
                    example_labels.append(mapped_label)
                    steps.append(step["text"])

            # Create a dict for each example
            all_labeled_steps.append(
                {
                    "steps": steps,
                    "labels": example_labels,  # Keep as list, not tensor yet
                }
            )

        return all_labeled_steps

    def load_prm800k_dataset(
        self,
        split: Optional[str] = None,
        streaming: bool = False,
    ) -> Dict[str, HFDataset]:
        """
        Load the PRM800K dataset from HuggingFace Hub.

        Args:
            split: Specific split to load ('train', 'test'), or None for all
            streaming: Whether to stream the dataset

        Returns:
            Dictionary with train and/or test datasets
        """
        dataset = load_dataset(
            "data/prm800k",
            split=split,
            streaming=streaming,
        )

        if split is None:
            return {
                "train": dataset["train"].map(
                    lambda batch: {
                        "labeled_steps": self._process_prm800k_labels_batch(
                            batch["question"], batch["label"]
                        )
                    },
                    batched=True,
                ),
                "test": dataset["test"].map(
                    lambda batch: {
                        "labeled_steps": self._process_prm800k_labels_batch(
                            batch["question"], batch["label"]
                        )
                    },
                    batched=True,
                ),
            }
        else:
            return {
                split: dataset.map(
                    lambda batch: {
                        "labeled_steps": self._process_prm800k_labels_batch(
                            batch["question"], batch["label"]
                        )
                    },
                    batched=True,
                )
            }

    def prepare_dataset(
        self,
        dataset: HFDataset,
        label_map: Optional[Dict[str, int]] = None,
    ) -> PRMDataset:
        """
        Prepare a dataset for training.

        Args:
            dataset: HuggingFace dataset
            label_map: Mapping from label strings to integers

        Returns:
            PRMDataset instance
        """
        return PRMDataset(dataset, self.featurizer, label_map)

    def compute_metrics(self, pred: EvalPrediction) -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            pred: Predictions from the model

        Returns:
            Dictionary of metrics
        """
        logits, labels = pred.predictions, pred.label_ids
        predictions = np.argmax(logits, axis=-1)

        # Flatten if needed (for token classification)
        if predictions.ndim > 1:
            predictions = predictions.reshape(-1)
            labels = labels.reshape(-1)

        # Remove padding tokens (label = -100)
        mask = labels != -100
        predictions = predictions[mask]
        labels = labels[mask]

        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average="weighted"),
            "precision": precision_score(
                labels, predictions, average="weighted", zero_division=0
            ),
            "recall": recall_score(
                labels, predictions, average="weighted", zero_division=0
            ),
        }

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
    ) -> None:
        """
        Train the PRM model.

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
        """
        # Create the collator based on your featurizer type
        if self.featurizer.featurizer_type == "tokens":
            collator = PRMCollator(
                pad_token_id=self.featurizer.tokenizer.pad_token_id,
                featurizer_type="tokens",
            )
        else:  # embeddings
            # Get embedding dimension from the featurizer or infer it from a sample
            if hasattr(self.featurizer, "embedding_dim"):
                embedding_dim = self.featurizer.embedding_dim
            elif hasattr(self.featurizer, "model"):
                # Try to infer from the model's config
                embedding_dim = self.featurizer.model.config.hidden_size
            else:
                # As a last resort, get it from a sample embedding
                sample_text = "test"
                sample_embedding = self.featurizer(sample_text)
                embedding_dim = sample_embedding.shape[-1]

            collator = PRMCollator(
                featurizer_type="embeddings", embedding_dim=embedding_dim
            )

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            data_collator=collator,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
        )

        # Train the model
        trainer.train()

        # Save the final model
        trainer.save_model(os.path.join(self.config.output_dir, "final_model"))

        # Save the featurizer configuration
        if self.featurizer.tokenizer is not None:
            self.featurizer.tokenizer.save_pretrained(
                os.path.join(self.config.output_dir, "final_model")
            )

    def evaluate(
        self,
        eval_dataset: Dataset,
    ) -> Dict[str, float]:
        """
        Evaluate the PRM model.

        Args:
            eval_dataset: Evaluation dataset

        Returns:
            Dictionary of evaluation metrics
        """
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
        )

        return trainer.evaluate()

    def predict(
        self,
        texts: list,
    ) -> np.ndarray:
        """
        Make predictions on new texts.

        Args:
            texts: List of texts to predict on

        Returns:
            Array of predictions
        """
        self.model.eval()

        predictions = []
        with torch.no_grad():
            for text in texts:
                if self.config.featurizer_type == "tokens":
                    features = self.featurizer(text, return_tensors="pt")
                    features = {k: v.to(self.device) for k, v in features.items()}
                    outputs = self.model(**features)
                else:
                    embeddings = self.featurizer(text)
                    embeddings = embeddings.unsqueeze(0).to(self.device)
                    outputs = self.model(inputs_embeds=embeddings)

                logits = (
                    outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
                )
                pred = torch.argmax(logits, dim=-1).cpu().numpy()
                predictions.append(pred)

        return np.array(predictions)

    def save_model(self, output_dir: str) -> None:
        """
        Save the trained model.

        Args:
            output_dir: Directory to save the model
        """
        os.makedirs(output_dir, exist_ok=True)

        if isinstance(self.model, PreTrainedModel):
            self.model.save_pretrained(output_dir)
        else:
            torch.save(self.model.state_dict(), os.path.join(output_dir, "model.pt"))

        if self.featurizer.tokenizer is not None:
            self.featurizer.tokenizer.save_pretrained(output_dir)

    def load_model(self, model_path: str) -> None:
        """
        Load a trained model.

        Args:
            model_path: Path to the saved model
        """
        if self.config.featurizer_type == "tokens":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=self.config.num_labels,
            )
        else:
            # Load custom model
            config = BertConfig.from_pretrained(model_path)
            self.model = BertForTokenClassificationWithEmbeddings(
                config, self.config.num_labels
            )
            state_dict = torch.load(os.path.join(model_path, "model.pt"))
            self.model.load_state_dict(state_dict)

        self.model = self.model.to(self.device)
