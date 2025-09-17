"""
Examples of working with different datasets in BMUq.
"""

import json
from pathlib import Path
from bmuq.benchmarks import load_dataset, create_dataset_from_questions, CustomDataset
from bmuq.config import get_preset_config
from bmuq import BMUqBenchmark


def explore_builtin_datasets():
    """Explore the built-in datasets."""
    
    print("Built-in Dataset Examples")
    print("=" * 40)
    
    # Load GSM8K dataset
    print("\n1. GSM8K Dataset (Grade School Math)")
    gsm8k = load_dataset("gsm8k")
    print(f"   Dataset size: {len(gsm8k)} problems")
    
    # Show first problem
    first_problem = gsm8k[0]
    print(f"   Sample question: {first_problem['question'][:100]}...")
    print(f"   Answer: {first_problem['answer']}")
    
    # Load Math dataset
    print("\n2. Math Dataset")
    math_dataset = load_dataset("math")
    print(f"   Dataset size: {len(math_dataset)} problems")
    
    # Show first problem
    first_math = math_dataset[0]
    print(f"   Sample question: {first_math['question']}")
    print(f"   Answer: {first_math['answer']}")
    print(f"   Level: {first_math.get('level', 'N/A')}")


def create_custom_datasets():
    """Create custom datasets from different sources."""
    
    print("\n\nCustom Dataset Examples")
    print("=" * 40)
    
    # 1. Simple questions and answers
    print("\n1. From question-answer pairs")
    questions = [
        "What is the capital of France?",
        "How many sides does a triangle have?",
        "What is 7 × 8?",
        "Who wrote Romeo and Juliet?"
    ]
    answers = ["Paris", "3", "56", "William Shakespeare"]
    
    qa_dataset = create_dataset_from_questions(questions, answers, name="general_knowledge")
    print(f"   Created dataset with {len(qa_dataset)} questions")
    
    # 2. Math word problems
    print("\n2. Math word problems")
    math_questions = [
        "Sarah has 12 apples. She gives 3 to her friend and eats 2. How many apples does she have left?",
        "A car travels 60 miles per hour for 2.5 hours. How far does it travel?",
        "If a pizza is cut into 8 slices and you eat 3 slices, what fraction of the pizza is left?",
        "A rectangle has length 15 cm and width 8 cm. What is its perimeter?"
    ]
    math_answers = ["7", "150", "5/8", "46"]
    
    math_dataset = create_dataset_from_questions(math_questions, math_answers, name="math_word_problems")
    print(f"   Created math dataset with {len(math_dataset)} problems")
    
    # 3. From dictionary data
    print("\n3. From structured data")
    structured_data = [
        {
            "id": 0,
            "question": "What is the freezing point of water in Celsius?",
            "answer": "0",
            "category": "science",
            "difficulty": "easy"
        },
        {
            "id": 1,
            "question": "How many planets are in our solar system?",
            "answer": "8",
            "category": "science",
            "difficulty": "easy"
        }
    ]
    
    structured_dataset = CustomDataset(structured_data, name="science_facts")
    print(f"   Created structured dataset with {len(structured_dataset)} questions")
    
    return qa_dataset, math_dataset, structured_dataset


def save_and_load_datasets():
    """Demonstrate saving and loading custom datasets."""
    
    print("\n\nSaving and Loading Datasets")
    print("=" * 40)
    
    # Create a dataset
    questions = [
        "Convert 32°F to Celsius",
        "What is the square root of 144?",
        "Solve: 3x - 5 = 10"
    ]
    answers = ["0", "12", "5"]
    
    dataset = create_dataset_from_questions(questions, answers, name="conversion_problems")
    
    # Save to JSON
    data_to_save = dataset.to_list()
    output_file = "sample_dataset.json"
    
    with open(output_file, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    
    print(f"Dataset saved to {output_file}")
    
    # Load it back
    loaded_dataset = load_dataset("custom", data_path=output_file)
    print(f"Dataset loaded: {len(loaded_dataset)} questions")
    
    # Verify the data
    assert len(loaded_dataset) == len(dataset)
    print("Dataset integrity verified!")
    
    # Clean up
    Path(output_file).unlink()
    print("Temporary file cleaned up")


def sample_and_filter_datasets():
    """Demonstrate dataset sampling and filtering."""
    
    print("\n\nDataset Sampling and Filtering")
    print("=" * 40)
    
    # Create larger dataset
    questions = [f"What is {i} + {i+1}?" for i in range(1, 21)]
    answers = [str(i + (i+1)) for i in range(1, 21)]
    
    full_dataset = create_dataset_from_questions(questions, answers, name="addition_problems")
    print(f"Full dataset: {len(full_dataset)} questions")
    
    # Sample subset
    sample_dataset = full_dataset.sample(5, seed=42)
    print(f"Sampled dataset: {len(sample_dataset)} questions")
    
    # Show sample questions
    print("Sample questions:")
    for i, question in enumerate(sample_dataset.to_list()[:3]):
        print(f"  {i+1}. {question['question']} → {question['answer']}")


def run_benchmark_on_custom_dataset():
    """Run BMUq benchmark on custom dataset."""
    
    print("\n\nRunning Benchmark on Custom Dataset")
    print("=" * 40)
    
    # Create a math-focused dataset
    math_questions = [
        "Find the area of a square with side length 6.",
        "If 3x + 2 = 11, what is x?",
        "Calculate 15% of 200.",
        "What is the perimeter of a rectangle with length 10 and width 4?",
        "Solve: 2(x + 3) = 14"
    ]
    math_answers = ["36", "3", "30", "28", "4"]
    
    math_dataset = create_dataset_from_questions(
        math_questions, math_answers, name="custom_math"
    )
    
    # Use fast configuration for demo
    config = get_preset_config("fast_development")
    config.experiment_name = "custom_dataset_test"
    config.benchmark.num_questions = len(math_questions)
    
    # Run benchmark
    benchmark = BMUqBenchmark(config)
    results = benchmark.run(
        dataset=math_dataset,
        save_results=True,
        output_dir="custom_results"
    )
    
    print(f"Benchmark completed:")
    print(f"  Success rate: {results.success_rate:.3f}")
    print(f"  Average confidence: {results.average_confidence:.3f}")
    print(f"  Results saved to: custom_results/")
    
    # Show some individual results
    print(f"\nSample results:")
    for i, result in enumerate(results.question_results[:3]):
        if result.get("success", False):
            print(f"  Q{i+1}: {'✓' if result.get('correct') else '✗'} "
                  f"(confidence: {result.get('confidence', 0):.2f})")


def create_domain_specific_datasets():
    """Create datasets for specific domains."""
    
    print("\n\nDomain-Specific Dataset Examples")
    print("=" * 40)
    
    # 1. Algebra problems
    print("\n1. Algebra Dataset")
    algebra_data = [
        {"question": "Solve for x: 2x + 5 = 13", "answer": "4", "type": "linear_equation"},
        {"question": "Factor: x² - 5x + 6", "answer": "(x-2)(x-3)", "type": "factoring"},
        {"question": "Simplify: 3(2x + 1) - 4x", "answer": "2x + 3", "type": "simplification"},
    ]
    
    algebra_dataset = CustomDataset(algebra_data, name="algebra_problems")
    print(f"   Created algebra dataset: {len(algebra_dataset)} problems")
    
    # 2. Geometry problems  
    print("\n2. Geometry Dataset")
    geometry_questions = [
        "Find the area of a circle with radius 4.",
        "What is the volume of a cube with side length 3?",
        "Calculate the hypotenuse of a right triangle with legs 3 and 4."
    ]
    geometry_answers = ["16π", "27", "5"]
    
    geometry_dataset = create_dataset_from_questions(
        geometry_questions, geometry_answers, name="geometry_problems"
    )
    print(f"   Created geometry dataset: {len(geometry_dataset)} problems")
    
    # 3. Word problems
    print("\n3. Word Problems Dataset")
    word_problems = [
        {
            "question": "Tom has twice as many marbles as Jerry. If Jerry has 15 marbles, how many does Tom have?",
            "answer": "30",
            "category": "multiplication",
            "difficulty": "easy"
        },
        {
            "question": "A train travels 300 miles in 4 hours. What is its average speed?",
            "answer": "75",
            "category": "rate_problems", 
            "difficulty": "medium"
        }
    ]
    
    word_dataset = CustomDataset(word_problems, name="word_problems")
    print(f"   Created word problems dataset: {len(word_dataset)} problems")
    
    return algebra_dataset, geometry_dataset, word_dataset


if __name__ == "__main__":
    # Run all dataset examples
    explore_builtin_datasets()
    create_custom_datasets()
    save_and_load_datasets()
    sample_and_filter_datasets()
    create_domain_specific_datasets()
    run_benchmark_on_custom_dataset()
    
    print("\n" + "="*60)
    print("Dataset Examples Completed!")
    print("\nKey features demonstrated:")
    print("1. Built-in datasets (GSM8K, Math)")
    print("2. Custom datasets from questions/answers")
    print("3. Loading datasets from JSON/CSV files")
    print("4. Dataset sampling and filtering") 
    print("5. Domain-specific dataset creation")
    print("6. Running benchmarks on custom datasets")
    print("\nNext steps:")
    print("- Try loading your own dataset files")
    print("- Experiment with different question types")
    print("- Create domain-specific evaluation sets")