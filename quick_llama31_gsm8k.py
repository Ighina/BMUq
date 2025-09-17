#!/usr/bin/env python3
"""
Quick script to run llama3.1:8b on GSM8K with SelfCheck.

Usage:
    python quick_llama31_gsm8k.py

Prerequisites:
    ollama serve
    ollama pull llama3.1:8b
"""

from bmuq.config.settings import load_config
from bmuq.benchmarks.benchmark import BMUqBenchmark

def main():
    # Load configuration
    config = load_config('llama31_gsm8k_config.yaml')

    # Run benchmark
    benchmark = BMUqBenchmark(config)
    result = benchmark.run()

    print(f"✅ Completed! Success rate: {result.success_rate:.3f}")

if __name__ == "__main__":
    main()