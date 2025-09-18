from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bmuq",
    version="0.1.0",
    author="BMUq Contributors",
    description="Bayesian Methods for Uncertainty Quantification in Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "openai>=1.0.0",
        "tenacity>=8.0.0",
        "dataclasses-json>=0.5.0",
        "pydantic>=1.8.0",
        "tqdm>=4.60.0",
        "matplotlib>=3.5.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "sentence-transformers>=5.1.0",
        "datasets>=4.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "pyyaml>=5.4",
            "requests>=2.25.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "jupyter>=1.0.0",
        ],
        "anthropic": ["anthropic>=0.3.0"],
        "huggingface": [
            "transformers>=4.20.0",
            "torch>=1.12.0",
            "accelerate>=0.20.0",
            "bitsandbytes>=0.39.0",
        ],
        "huggingface-cpu": [
            "transformers>=4.20.0",
            "torch>=1.12.0",
            "accelerate>=0.20.0",
        ],
        "all": [
            "anthropic>=0.3.0",
            "transformers>=4.20.0",
            "torch>=1.12.0",
            "accelerate>=0.20.0",
            "bitsandbytes>=0.39.0",
        ],
    },
)
