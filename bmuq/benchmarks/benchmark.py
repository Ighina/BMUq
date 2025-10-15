"""
Main benchmarking framework for BMUq.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union, Tuple

from ..config.settings import BMUqConfig
from ..models.base import BaseLLM
from ..uncertainty.selfcheck import SelfCheck
from ..uncertainty.base_methods import (
    EntropyBasedUQ,
    ConsistencyBasedUQ,
    RandomBaselineUQ,
    MajorityVoteUQ,
)
from ..uncertainty.uq_methods import (
    SemanticEntropyBasedUQ,
    SemanticEntropy,
    EntailmentChecker,
)
from ..uncertainty.coherence_uq import CoherenceBasedUQ, RelativeCoherenceBasedUQ
from ..uncertainty.adapters import add_weighted_aggregation
from ..search.tree_search import TreeSearchCoT
from ..search.beam_search import BeamSearchCoT
from ..search.best_of_n import BestNSearchCoT
from ..core.data_structures import ReasoningPath
from .datasets import load_dataset, Dataset
from .metrics import calculate_metrics, MetricResult
from .evaluator import Evaluator
from .utils import set_nested_attribute


@dataclass
class BenchmarkResult:
    """Results from running a benchmark."""

    experiment_name: str
    config: BMUqConfig
    dataset_name: str
    num_questions: int

    # Timing information
    start_time: datetime
    end_time: datetime
    total_runtime_seconds: float

    # Results per question
    question_results: List[Dict[str, Any]]

    # Aggregated metrics
    metrics: Dict[str, MetricResult]

    # Statistics
    success_rate: float  # Fraction of questions successfully processed
    average_confidence: float
    average_path_length: float

    # LLM usage statistics
    llm_stats: Dict[str, Any]

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert benchmark result to dictionary for serialization."""
        return {
            "experiment_name": self.experiment_name,
            "config": self.config.to_dict(),
            "dataset_name": self.dataset_name,
            "num_questions": self.num_questions,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "total_runtime_seconds": self.total_runtime_seconds,
            "question_results": self.question_results,
            "metrics": {
                name: result.to_dict() for name, result in self.metrics.items()
            },
            "success_rate": self.success_rate,
            "average_confidence": self.average_confidence,
            "average_path_length": self.average_path_length,
            "llm_stats": self.llm_stats,
            "metadata": self.metadata,
        }

    def save(self, output_path: Union[str, Path]) -> None:
        """Save benchmark result to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

        print(f"Benchmark results saved to: {output_path}")


class BMUqBenchmark:
    """
    Main benchmark class for evaluating uncertainty quantification methods.
    """

    def __init__(self, config: BMUqConfig):
        """
        Initialize benchmark with configuration.

        Args:
            config: BMUq configuration
        """
        self.config = config
        self.evaluator = Evaluator()

        # Initialize components based on config
        self.llm = self._create_llm()
        self.uncertainty_method = self._create_uncertainty_method()
        self.search_algorithm = self._create_search_algorithm()

    def run(
        self,
        dataset: Optional[Dataset] = None,
        num_questions: Optional[int] = None,
        save_results: bool = True,
        output_dir: Optional[str] = None,
        generated_outputs: Union[str, List[Dict[str, List[str]]]] = None,
        create_outputs: bool = False,
    ) -> BenchmarkResult:
        """
        Run benchmark evaluation.

        Args:
            dataset: Dataset to evaluate on (if None, loads from config)
            num_questions: Number of questions to evaluate (if None, uses config)
            save_results: Whether to save results to disk
            output_dir: Directory to save results (if None, uses config)
            generated_outputs: either the path to a json file or a list of dictionaries with the structure {id_of_dataset_instance:list_of_answers_to_the_dataset_instance}
            create_outputs: whether to create and save the options for best of n method and then pass them as precomputed options
        Returns:
            BenchmarkResult containing evaluation results
        """
        start_time = datetime.now()

        # Load dataset if not provided
        if dataset is None:
            dataset = load_dataset(
                self.config.benchmark.dataset, data_path=self.config.benchmark.data_path
            )

        # Determine number of questions
        if num_questions is None:
            num_questions = self.config.benchmark.num_questions
        if num_questions is None:
            num_questions = len(dataset)

        num_questions = min(num_questions, len(dataset))

        print(f"Starting benchmark: {self.config.experiment_name}")
        print(f"Dataset: {dataset.name} ({num_questions} questions)")
        print(f"Method: {self.config.uncertainty.method}")
        print(f"Search: {self.config.search.algorithm}")
        print("-" * 60)

        # Run evaluation on each question
        question_results = []
        successful_evaluations = 0

        if isinstance(generated_outputs, str):
            with open(generated_outputs) as f:
                generated_outputs = json.load(f)
        elif create_outputs:
            # TODO: test the precomputed outputs!
            assert (
                self.search_algorithm.name == "best_of_n_cot"
            ), "Only best of N method can be used with create_outputs option!"
            generated_outputs = []
            for i in range(num_questions):
                question = dataset[i]

                beam_width = self.config.search.beam_width

                current_beams = [
                    ReasoningPath(steps=[], path_id=f"beam_{idx}")
                    for idx in range(beam_width)
                ]

                candidates, answers = self.search_algorithm.generate_next_steps(
                    question["question"], current_beams, beam_width, return_text=True
                )

                generated_outputs.append({"reasonings": candidates, "answers": answers})

            with open(
                f"{self.config.benchmark.dataset}-generated-outputs-{beam_width}.json",
                "w",
            ) as f:
                json.dump(generated_outputs, f)

        for i in range(num_questions):
            question = dataset[i]

            if self.config.benchmark.verbose:
                print(f"Question {i+1}/{num_questions}: Processing...")

            try:
                # Run search to find reasoning paths
                # TODO: implement answers extraction for other search algorithms too
                if self.search_algorithm.name == "best_of_n_cot":

                    precomputed_dictionary = (
                        generated_outputs[i] if generated_outputs else None
                    )

                    paths, answers = self.search_algorithm.search(
                        question["question"],
                        num_solutions=3,
                        verbose=False,
                        generated_outputs=precomputed_dictionary,
                    )

                else:
                    paths = self.search_algorithm.search(
                        question["question"], num_solutions=3, verbose=False
                    )

                # Select best path
                best_path = paths[0] if paths else None

                try:
                    name = self.uncertainty_method.name
                except AttributeError:
                    self.uncertainty_method.name = (
                        self.uncertainty_method.base_method.name
                    )

                if best_path:
                    # Evaluate the path
                    result = self.evaluator.evaluate_question(
                        question=question,
                        predicted_path=(best_path, answers[0]),
                        uncertainty_method=self.uncertainty_method,
                        search_algorithm=self.search_algorithm,
                        structured_output=self.config.search.structured_output,
                    )

                    question_results.append(result)
                    successful_evaluations += 1

                    if self.config.benchmark.verbose:
                        print(f"  Confidence: {best_path.total_confidence:.3f}")
                        print(f"  Path length: {len(best_path.steps)} steps")
                else:
                    # No paths found
                    question_results.append(
                        {
                            "question_id": i,
                            "question": question["question"],
                            "error": "No reasoning paths found",
                            "success": False,
                        }
                    )

            except Exception as e:
                print(f"  Error processing question {i+1}: {e}")
                question_results.append(
                    {
                        "question_id": i,
                        "question": question["question"],
                        "error": str(e),
                        "success": False,
                    }
                )

        end_time = datetime.now()
        total_runtime = (end_time - start_time).total_seconds()

        # Calculate aggregated metrics
        metrics = calculate_metrics(question_results, self.config.benchmark.metrics)

        # Calculate summary statistics
        successful_results = [r for r in question_results if r.get("success", False)]

        success_rate = successful_evaluations / num_questions
        average_confidence = sum(
            r.get("confidence", 0) for r in successful_results
        ) / max(1, len(successful_results))
        average_path_length = sum(
            r.get("path_length", 0) for r in successful_results
        ) / max(1, len(successful_results))

        # Get LLM usage statistics
        llm_stats = (
            self.llm.get_usage_stats().to_dict()
            if hasattr(self.llm.get_usage_stats(), "to_dict")
            else vars(self.llm.get_usage_stats())
        )

        # Create benchmark result
        result = BenchmarkResult(
            experiment_name=self.config.experiment_name,
            config=self.config,
            dataset_name=dataset.name,
            num_questions=num_questions,
            start_time=start_time,
            end_time=end_time,
            total_runtime_seconds=total_runtime,
            question_results=question_results,
            metrics=metrics,
            success_rate=success_rate,
            average_confidence=average_confidence,
            average_path_length=average_path_length,
            llm_stats=llm_stats,
            metadata={
                "search_stats": (
                    self.search_algorithm.get_search_statistics()
                    if hasattr(self.search_algorithm, "get_search_statistics")
                    else {}
                ),
                "uncertainty_info": self.uncertainty_method.get_method_info(),
            },
        )

        print("-" * 60)
        print(f"Benchmark completed in {total_runtime:.2f} seconds")
        print(
            f"Success rate: {success_rate:.3f} ({successful_evaluations}/{num_questions})"
        )
        print(f"Average confidence: {average_confidence:.3f}")
        print(f"Average path length: {average_path_length:.1f} steps")

        # Save results if requested
        if save_results:
            output_dir = output_dir or self.config.benchmark.output_dir
            output_path = (
                Path(output_dir)
                / f"{self.config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            result.save(output_path)

        return result

    def run_comparison(
        self,
        uncertainty_methods: List[str],
        additional_configuration: Dict[str, List] = {},
        dataset: Optional[Dataset] = None,
        num_questions: Optional[int] = None,
        generated_outputs: Union[str, List[Dict[str, List[str]]]] = None,
        create_outputs: bool = False,
    ) -> Dict[str, BenchmarkResult]:
        """
        Run comparison across multiple uncertainty quantification methods.

        Args:
            uncertainty_methods: List of method names to compare
            additional_configuration: a dictionary having keys equal to one or more listed uncertainty methods '
                                      and values being List of different parameters to compare for that method
                                      (each parameter should be a dictionary in the form {(tuple of path to parameter):value_of_parameter})
            dataset: Dataset to evaluate on
            num_questions: Number of questions to evaluate
            generated_outputs: either the path to a json file or a list of dictionaries with the structure {id_of_dataset_instance:list_of_answers_to_the_dataset_instance}
            create_outputs: whether to create and save the options for best of n method and then pass them as precomputed options
        Returns:
            Dictionary mapping method names to benchmark results
        """
        results = {}
        original_method = self.config.uncertainty.method
        original_experiment_name = self.config.experiment_name
        first = True

        for method in uncertainty_methods:
            if not first:
                create_outputs = False  # only create outputs for the first method

            print(f"\n{'='*60}")
            print(f"Running comparison: {method}")
            print(f"{'='*60}")

            if method in additional_configuration:

                for parameter in additional_configuration[method]:
                    parms = []
                    for idx in range(len(parameter)):
                        path_to_attr = list(parameter.keys())[idx]
                        value = list(parameter.values())[idx]
                        parms.append("_".join(path_to_attr) + ":" + str(value))
                        set_nested_attribute(self.config, path_to_attr, value)

                    parms = "-".join(parms)

                    # Update config for this method
                    self.config.uncertainty.method = method
                    self.config.experiment_name = (
                        f"{self.config.experiment_name}_{method}_{parms}"
                    )

                    # Recreate uncertainty method
                    self.uncertainty_method = self._create_uncertainty_method()
                    self.search_algorithm = (
                        self._create_search_algorithm()
                    )  # Recreate with new uncertainty method

                    # Run benchmark
                    result = self.run(
                        dataset,
                        num_questions,
                        save_results=True,
                        generated_outputs=generated_outputs,
                        create_outputs=create_outputs,
                    )

                    results[method + "_" + parms] = result

                    if create_outputs:
                        # this will apply only for the first method in the list and only if create_outputs is set to True
                        beam_width = self.config.search.beam_width
                        with open(
                            f"{self.config.benchmark.dataset}-generated-outputs-{beam_width}.json",
                        ) as f:
                            generated_outputs = json.load(f)

                    create_outputs = False  # only create outputs for the first method

                    self.config.experiment_name = original_experiment_name

            else:
                # Update config for this method
                self.config.uncertainty.method = method
                self.config.experiment_name = f"{self.config.experiment_name}_{method}"

                # Recreate uncertainty method
                self.uncertainty_method = self._create_uncertainty_method()
                self.search_algorithm = (
                    self._create_search_algorithm()
                )  # Recreate with new uncertainty method

                # Run benchmark
                result = self.run(
                    dataset,
                    num_questions,
                    save_results=True,
                    generated_outputs=generated_outputs,
                    create_outputs=create_outputs,
                )
                results[method] = result

                if create_outputs:
                    # this will apply only for the first method in the list and only if create_outputs is set to True
                    beam_width = self.config.search.beam_width
                    with open(
                        f"{self.config.benchmark.dataset}-generated-outputs-{beam_width}.json",
                    ) as f:
                        generated_outputs = json.load(f)

            self.config.experiment_name = original_experiment_name

        # Restore original configuration
        self.config.uncertainty.method = original_method

        return results

    def _create_llm(self) -> BaseLLM:
        """Create LLM instance based on configuration."""
        llm_config = self.config.llm

        if llm_config.provider == "openai":
            from ..models.openai_llm import OpenAILLM

            return OpenAILLM(
                api_key=llm_config.api_key,
                model=llm_config.model,
                temperature=llm_config.temperature,
                max_retries=llm_config.max_retries,
            )
        elif llm_config.provider == "huggingface":
            try:
                from ..models.huggingface_llm import HuggingFaceLLM

                return HuggingFaceLLM(
                    model_name=llm_config.model,
                    device=llm_config.hf_device,
                    temperature=llm_config.temperature,
                    max_retries=llm_config.max_retries,
                    max_new_tokens=llm_config.max_tokens,
                    use_quantization=llm_config.hf_use_quantization,
                    load_in_8bit=llm_config.hf_load_in_8bit,
                    load_in_4bit=llm_config.hf_load_in_4bit,
                    **llm_config.extra_params,
                )
            except ImportError:
                raise ValueError(
                    "HuggingFace transformers not installed. Install with: pip install transformers torch"
                )
        elif llm_config.provider == "ollama":
            try:
                from ..models.ollama_llm import OllamaLLM

                return OllamaLLM(
                    model=llm_config.model,
                    base_url=llm_config.ollama_base_url,
                    temperature=llm_config.temperature,
                    max_retries=llm_config.max_retries,
                    timeout=int(llm_config.timeout),
                    system_prompt=llm_config.ollama_system_prompt,
                )
            except ImportError:
                raise ValueError(
                    "Requests library not installed. Install with: pip install requests"
                )
        elif llm_config.provider == "mock":
            from ..models.mock_llm import MockLLM

            return MockLLM(
                model=llm_config.model,
                temperature=llm_config.temperature,
                max_retries=llm_config.max_retries,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_config.provider}")

    def _create_uncertainty_method(self):
        """Create uncertainty quantification method based on configuration."""
        method_name = self.config.uncertainty.method

        if method_name == "selfcheck":
            uncertainty = SelfCheck(
                self.llm,
                lambda_neg1=self.config.uncertainty.lambda_neg1,
                lambda_0=self.config.uncertainty.lambda_0,
            )
        elif method_name == "entropy_based":
            uncertainty = EntropyBasedUQ(
                self.llm,
                num_samples=self.config.uncertainty.num_samples,
                temperature=self.config.uncertainty.sampling_temperature,
            )
        elif method_name == "consistency_based":
            uncertainty = ConsistencyBasedUQ(self.llm)
        elif method_name == "random_baseline":
            uncertainty = RandomBaselineUQ(seed=self.config.random_seed)
        elif method_name == "majority_vote":
            return MajorityVoteUQ()
        elif method_name == "semantic_entropy":
            uncertainty = SemanticEntropyBasedUQ(
                semantic_entropy=SemanticEntropy(
                    entailment_checker=EntailmentChecker.from_pretrained(
                        self.config.uncertainty.entailment_model
                    ),
                    strict_entailment=self.config.uncertainty.strict_entailment,
                    verbose=self.config.uncertainty.verbose,
                ),
                add_consistency=self.config.uncertainty.add_consistency,
            )
        elif method_name == "coherence_based":
            # Get coherence-specific parameters from extra_params
            extra_params = self.config.uncertainty.extra_params
            coherence_method = extra_params.get(
                "coherence_method", "mean_cosine_similarity"
            )
            model_name = extra_params.get("model_name", "all-MiniLM-L6-v2")
            decay = extra_params.get("decay", 0.9)
            add_topic_score = extra_params.get("add_topic_score", False)
            question_weight = extra_params.get("question_weight", 0.2)

            uncertainty = CoherenceBasedUQ(
                model_name=model_name,
                coherence_method=coherence_method,
                decay=decay,
                add_topic_score=add_topic_score,
                question_weight=question_weight,
            )
        elif method_name == "relative_coherence_based":
            extra_params = self.config.uncertainty.extra_params
            coherence_method = extra_params.get("coherence_method", "arp_pair")
            model_name = extra_params.get("model_name", "all-MiniLM-L6-v2")
            add_topic_score = extra_params.get("add_topic_score", False)
            question_weight = extra_params.get("question_weight", 0.2)

            uncertainty = RelativeCoherenceBasedUQ(
                model_name=model_name,
                coherence_method=coherence_method,
                add_topic_score=add_topic_score,
                question_weight=question_weight,
            )

        else:
            raise ValueError(f"Unsupported uncertainty method: {method_name}")

        if self.config.uncertainty.answer_weight:
            uncertainty = add_weighted_aggregation(uncertainty, problem_type="math")
        return uncertainty

    def _create_search_algorithm(self):
        """Create search algorithm based on configuration."""
        algorithm_name = self.config.search.algorithm

        if algorithm_name == "tree_search":
            return TreeSearchCoT(
                llm=self.llm,
                uncertainty_method=self.uncertainty_method,
                beam_width=self.config.search.beam_width,
                max_depth=self.config.search.max_depth,
                confidence_threshold=self.config.search.confidence_threshold,
                exploration_weight=self.config.search.exploration_weight,
                structured_format=self.config.search.structured_format,
            )
        elif algorithm_name == "beam_search":
            return BeamSearchCoT(
                llm=self.llm,
                uncertainty_method=self.uncertainty_method,
                beam_width=self.config.search.beam_width,
                max_depth=self.config.search.max_depth,
                diversity_penalty=self.config.search.diversity_penalty,
                structured_format=self.config.search.structured_format,
            )
        elif algorithm_name == "best_of_n":
            return BestNSearchCoT(
                llm=self.llm,
                uncertainty_method=self.uncertainty_method,
                beam_width=self.config.search.beam_width,
                structured_format=self.config.search.structured_format,
                structured_output=self.config.search.structured_output,
            )
        else:
            raise ValueError(f"Unsupported search algorithm: {algorithm_name}")
