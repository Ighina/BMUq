"""
Coherence-based uncertainty quantification using embedding similarity.

This module implements uncertainty quantification based on semantic coherence
between reasoning steps using sentence embeddings and cosine similarity.
"""

from typing import List, Optional, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer, util

from ..core.interfaces import UncertaintyMethod
from ..core.data_structures import ReasoningStep, ReasoningPath, UncertaintyScore


class CoherenceBasedUQ(UncertaintyMethod):
    """
    Coherence-based uncertainty quantification using embedding similarity.
    
    Measures uncertainty by evaluating the semantic coherence of each reasoning step
    with respect to previous steps in the reasoning chain. Higher coherence indicates
    lower uncertainty and higher confidence.
    
    The method uses sentence transformers to embed reasoning steps and computes
    various coherence metrics based on cosine similarity.
    """

    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 coherence_method: str = "mean_cosine_similarity",
                 decay: float = 0.9):
        """
        Initialize coherence-based uncertainty quantification.
        
        Args:
            model_name: Name of the sentence transformer model to use
            coherence_method: Method for computing coherence scores
                - "mean_cosine_similarity": Average similarity with all previous steps
                - "max_cosine_similarity": Maximum similarity with any previous step
                - "weighted_cosine_similarity": Weighted average with recency decay
                - "cluster_centroid": Similarity with centroid of previous embeddings
            decay: Decay factor for weighted similarity (0 < decay <= 1)
        """
        super().__init__("coherence_based")
        self.model = SentenceTransformer(model_name)
        self.coherence_method = coherence_method
        self.decay = decay
        
        # Validate coherence method
        valid_methods = {
            "mean_cosine_similarity",
            "max_cosine_similarity", 
            "weighted_cosine_similarity",
            "cluster_centroid"
        }
        if coherence_method not in valid_methods:
            raise ValueError(f"Invalid coherence method: {coherence_method}. "
                           f"Must be one of {valid_methods}")

    def _embed_texts(self, texts: List[str]):
        """Embed texts using the sentence transformer model."""
        return self.model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)

    def _compute_coherence_score(self, current_text: str, previous_texts: List[str]) -> float:
        """
        Compute coherence score between current step and previous steps.
        
        Args:
            current_text: Content of the current reasoning step
            previous_texts: Contents of previous reasoning steps
            
        Returns:
            Coherence score between 0 and 1
        """
        if not previous_texts:
            return 1.0  # No previous context, assume full coherence
        
        # Embed all texts
        all_texts = previous_texts + [current_text]
        embeddings = self._embed_texts(all_texts)
        
        prev_embeddings = embeddings[:-1]
        curr_embedding = embeddings[-1]
        
        # Compute similarity scores
        similarities = util.cos_sim(curr_embedding, prev_embeddings)[0].cpu().numpy()
        
        # Apply coherence method
        if self.coherence_method == "mean_cosine_similarity":
            return float(np.mean(similarities))
            
        elif self.coherence_method == "max_cosine_similarity":
            return float(np.max(similarities))
            
        elif self.coherence_method == "weighted_cosine_similarity":
            weights = np.array([self.decay ** (len(previous_texts) - i - 1) 
                              for i in range(len(previous_texts))])
            weights /= np.sum(weights)
            return float(np.sum(similarities * weights))
            
        elif self.coherence_method == "cluster_centroid":
            centroid = prev_embeddings.mean(dim=0)
            similarity = util.cos_sim(curr_embedding, centroid)
            return float(similarity.item())
        
        else:
            # Fallback to mean similarity
            return float(np.mean(similarities))

    def evaluate_step(self, question: str, reasoning_path: List[ReasoningStep], 
                     step_to_evaluate: ReasoningStep) -> UncertaintyScore:
        """
        Evaluate coherence-based uncertainty for a single reasoning step.
        
        Args:
            question: The original question being solved
            reasoning_path: The reasoning path so far (context)
            step_to_evaluate: The step to evaluate uncertainty for
            
        Returns:
            UncertaintyScore with coherence-based confidence
        """
        try:
            # Extract text content from reasoning steps
            previous_texts = [step.content for step in reasoning_path]
            current_text = step_to_evaluate.content
            
            # Compute coherence score
            coherence_score = self._compute_coherence_score(current_text, previous_texts)
            
            # Coherence score is directly used as confidence
            # High coherence = high confidence = low uncertainty
            confidence = max(0.0, min(1.0, coherence_score))
            
            # Prepare metadata
            metadata = {
                "coherence_method": self.coherence_method,
                "coherence_score": coherence_score,
                "num_previous_steps": len(previous_texts),
                "model_name": self.model._modules['0'].auto_model.name_or_path if hasattr(self.model, '_modules') else "unknown"
            }
            
            if self.coherence_method == "weighted_cosine_similarity":
                metadata["decay"] = self.decay
            
            return UncertaintyScore(
                value=confidence,
                method=self.name,
                metadata=metadata
            )
            
        except Exception as e:
            # Return neutral confidence on error
            return UncertaintyScore(
                value=0.5,
                method=self.name,
                metadata={"error": str(e)}
            )

    def evaluate_path(self, question: str, reasoning_path: ReasoningPath) -> float:
        """
        Evaluate coherence-based uncertainty for an entire reasoning path.
        
        Args:
            question: The original question being solved
            reasoning_path: Complete reasoning path to evaluate
            
        Returns:
            Overall coherence-based confidence score for the path
        """
        if not reasoning_path.steps:
            return 0.0
        
        # If only one step, return high confidence
        if len(reasoning_path.steps) == 1:
            return 1.0
        
        confidences = []
        
        # Evaluate each step in context of previous steps
        for i, step in enumerate(reasoning_path.steps):
            if self.name in step.uncertainty_scores:
                # Use existing score if available
                confidences.append(step.uncertainty_scores[self.name].value)
            else:
                # Compute coherence score for this step
                previous_steps = reasoning_path.steps[:i]
                score = self.evaluate_step(question, previous_steps, step)
                step.uncertainty_scores[self.name] = score
                confidences.append(score.value)
        
        # Return average confidence across all steps
        return sum(confidences) / len(confidences)

    def batch_evaluate_steps(self, question: str, reasoning_path: List[ReasoningStep],
                            steps_to_evaluate: List[ReasoningStep]) -> List[UncertaintyScore]:
        """
        Efficiently batch evaluate multiple steps using vectorized embedding.
        
        Args:
            question: The original question being solved
            reasoning_path: The reasoning path so far (context)
            steps_to_evaluate: List of steps to evaluate
            
        Returns:
            List of UncertaintyScore objects for each step
        """
        if not steps_to_evaluate:
            return []
        
        try:
            # Extract previous texts for context
            previous_texts = [step.content for step in reasoning_path]
            
            # Extract texts from steps to evaluate
            current_texts = [step.content for step in steps_to_evaluate]
            
            # Batch embed all texts if we have previous context
            if previous_texts:
                all_texts = previous_texts + current_texts
                embeddings = self._embed_texts(all_texts)
                
                prev_embeddings = embeddings[:len(previous_texts)]
                curr_embeddings = embeddings[len(previous_texts):]
                
                # Compute batch similarities
                batch_similarities = util.cos_sim(curr_embeddings, prev_embeddings).cpu().numpy()
                
                results = []
                for i, (step, similarities) in enumerate(zip(steps_to_evaluate, batch_similarities)):
                    # Apply coherence method to similarities
                    if self.coherence_method == "mean_cosine_similarity":
                        coherence_score = float(np.mean(similarities))
                    elif self.coherence_method == "max_cosine_similarity":
                        coherence_score = float(np.max(similarities))
                    elif self.coherence_method == "weighted_cosine_similarity":
                        weights = np.array([self.decay ** (len(previous_texts) - j - 1) 
                                          for j in range(len(previous_texts))])
                        weights /= np.sum(weights)
                        coherence_score = float(np.sum(similarities * weights))
                    elif self.coherence_method == "cluster_centroid":
                        centroid = prev_embeddings.mean(dim=0)
                        similarity = util.cos_sim(curr_embeddings[i], centroid)
                        coherence_score = float(similarity.item())
                    else:
                        coherence_score = float(np.mean(similarities))
                    
                    confidence = max(0.0, min(1.0, coherence_score))
                    
                    metadata = {
                        "coherence_method": self.coherence_method,
                        "coherence_score": coherence_score,
                        "num_previous_steps": len(previous_texts),
                        "batch_processed": True
                    }
                    
                    results.append(UncertaintyScore(
                        value=confidence,
                        method=self.name,
                        metadata=metadata
                    ))
                
                return results
            else:
                # No previous context, return high confidence for all
                return [UncertaintyScore(
                    value=1.0,
                    method=self.name,
                    metadata={"coherence_method": self.coherence_method, "no_previous_context": True}
                ) for _ in steps_to_evaluate]
                
        except Exception as e:
            # Return neutral confidence for all on error
            return [UncertaintyScore(
                value=0.5,
                method=self.name,
                metadata={"error": str(e), "batch_processed": True}
            ) for _ in steps_to_evaluate]

    def get_method_info(self) -> Dict[str, Any]:
        """Get detailed information about this coherence-based method."""
        base_info = super().get_method_info()
        base_info.update({
            "coherence_method": self.coherence_method,
            "model_name": getattr(self.model, 'model_name', 'unknown'),
            "decay_factor": self.decay if self.coherence_method == "weighted_cosine_similarity" else None,
            "description": "Coherence-based uncertainty quantification using semantic similarity between reasoning steps"
        })
        return base_info


# Factory function for easy instantiation
def create_coherence_uq(coherence_method: str = "mean_cosine_similarity",
                       model_name: str = "all-MiniLM-L6-v2",
                       decay: float = 0.9) -> CoherenceBasedUQ:
    """
    Factory function to create a CoherenceBasedUQ instance.
    
    Args:
        coherence_method: Method for computing coherence scores
        model_name: Sentence transformer model name
        decay: Decay factor for weighted similarity
        
    Returns:
        Configured CoherenceBasedUQ instance
    """
    return CoherenceBasedUQ(
        model_name=model_name,
        coherence_method=coherence_method,
        decay=decay
    )