"""
Featurizer module for Process Reward Models.

This module provides functionality to convert text into features suitable for PRM training,
supporting both token-based features (from pre-trained tokenizers) and sentence embeddings
(from sentence-transformers models).
"""

from typing import Dict, List, Union, Optional, Literal
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
from sentence_transformers import SentenceTransformer


class PRMFeaturizer:
    """
    Featurizer for Process Reward Models that supports two types of input features:
    1. Token-based features using a pre-trained tokenizer
    2. Sentence embeddings from sentence-transformers models

    Args:
        featurizer_type: Type of featurization to use ('tokens' or 'embeddings')
        model_name_or_path: Path or name of the model to use for featurization
        max_length: Maximum sequence length for tokenization (default: 512)
        padding: Padding strategy for tokenization (default: 'max_length')
        truncation: Whether to truncate sequences (default: True)
    """

    def __init__(
        self,
        featurizer_type: Literal["tokens", "embeddings"] = "tokens",
        model_name_or_path: str = "bert-base-uncased",
        max_length: int = 512,
        padding: Union[bool, str] = "max_length",
        truncation: bool = True,
    ):
        self.featurizer_type = featurizer_type
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

        # Initialize the appropriate featurizer
        if featurizer_type == "tokens":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.sentence_transformer = None
        elif featurizer_type == "embeddings":
            self.tokenizer = None
            self.sentence_transformer = SentenceTransformer(model_name_or_path)
            self.embedding_dim = (
                self.sentence_transformer.get_sentence_embedding_dimension()
            )
        else:
            raise ValueError(
                f"Invalid featurizer_type: {featurizer_type}. Must be 'tokens' or 'embeddings'"
            )

    def featurize_tokens(
        self, texts: Union[str, List[str]], return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """
        Convert texts to token-based features using the tokenizer.

        Args:
            texts: Single text or list of texts to featurize
            return_tensors: Type of tensors to return ('pt' for PyTorch)

        Returns:
            Dictionary containing input_ids, attention_mask, and token_type_ids
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Use featurizer_type='tokens'")

        # Tokenize the input texts
        features = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors=return_tensors,
        )

        return features

    def featurize_embeddings(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> torch.Tensor:
        """
        Convert texts to sentence embeddings using sentence-transformers.

        Args:
            texts: Single text or list of texts to featurize
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar during encoding

        Returns:
            Tensor of shape (batch_size, embedding_dim) containing sentence embeddings
        """
        if self.sentence_transformer is None:
            raise ValueError(
                "SentenceTransformer not initialized. Use featurizer_type='embeddings'"
            )

        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]

        # Encode texts to embeddings
        embeddings = self.sentence_transformer.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=True,
        )

        return embeddings

    def __call__(
        self, texts: Union[str, List[str]], **kwargs
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Featurize texts based on the configured featurizer type.

        Args:
            texts: Single text or list of texts to featurize
            **kwargs: Additional arguments passed to the specific featurization method

        Returns:
            Features appropriate for the configured featurizer type
        """
        if self.featurizer_type == "tokens":
            return self.featurize_tokens(texts, **kwargs)
        elif self.featurizer_type == "embeddings":
            return self.featurize_embeddings(texts, **kwargs)
        else:
            raise ValueError(f"Invalid featurizer_type: {self.featurizer_type}")

    def get_embedding_dim(self) -> Optional[int]:
        """
        Get the embedding dimension for sentence embeddings.

        Returns:
            Embedding dimension if using embeddings, None otherwise
        """
        if (
            self.featurizer_type == "embeddings"
            and self.sentence_transformer is not None
        ):
            return self.sentence_transformer.get_sentence_embedding_dimension()
        return None

    def get_vocab_size(self) -> Optional[int]:
        """
        Get the vocabulary size for token-based featurization.

        Returns:
            Vocabulary size if using tokens, None otherwise
        """
        if self.featurizer_type == "tokens" and self.tokenizer is not None:
            return len(self.tokenizer)
        return None
