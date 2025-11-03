"""
Uncertainty quantification methods for reasoning chains.
"""

from .selfcheck import SelfCheck
from .base_methods import (
    EntropyBasedUQ,
    ConsistencyBasedUQ,
    RandomBaselineUQ,
    MajorityVoteUQ,
)
from .uq_methods import SemanticEntropyBasedUQ, SequentialConsistencyBasedUQ
from .coherence_uq import (
    CoherenceBasedUQ,
    create_coherence_uq,
    RelativeCoherenceBasedUQ,
    create_relative_coherence_uq,
)
from .prmbased_uq import PRMBasedUQ
from .token_probability_uq import (
    PerplexityBasedUQ,
    MeanLogProbUQ,
    TokenEntropyBasedUQ,
    create_perplexity_uq,
    create_mean_log_prob_uq,
    create_token_entropy_uq,
)
from .weighted_aggregation import (
    WeightedAnswerAggregator,
    WeightedUncertaintyMethod,
    MathAnswerExtractor,
    GenericAnswerExtractor,
    AnswerCandidate,
    create_math_weighted_method,
    create_generic_weighted_method,
)
from .adapters import (
    WeightedAnswerAdapter,
    add_weighted_aggregation,
    BulkEvaluator,
    create_coherence_with_aggregation,
    create_relative_coherence_with_aggregation,
    create_random_with_aggregation,
)

__all__ = [
    "SelfCheck",
    "EntropyBasedUQ",
    "ConsistencyBasedUQ",
    "RandomBaselineUQ",
    "MajorityVoteUQ",
    "SemanticEntropyBasedUQ",
    "SequentialConsistencyBasedUQ",
    "CoherenceBasedUQ",
    "create_coherence_uq",
    "RelativeCoherenceBasedUQ",
    "create_relative_coherence_uq",
    "PRMBasedUQ",
    # Token probability-based methods
    "PerplexityBasedUQ",
    "MeanLogProbUQ",
    "TokenEntropyBasedUQ",
    "create_perplexity_uq",
    "create_mean_log_prob_uq",
    "create_token_entropy_uq",
    # Weighted aggregation components
    "WeightedAnswerAggregator",
    "WeightedUncertaintyMethod",
    "MathAnswerExtractor",
    "GenericAnswerExtractor",
    "AnswerCandidate",
    "create_math_weighted_method",
    "create_generic_weighted_method",
    # Adapters
    "WeightedAnswerAdapter",
    "add_weighted_aggregation",
    "BulkEvaluator",
    "create_coherence_with_aggregation",
    "create_relative_coherence_with_aggregation",
    "create_random_with_aggregation",
]
