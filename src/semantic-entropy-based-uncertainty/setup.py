from setuptools import setup, find_packages

setup(
    name="semantic-entropy",
    version="0.1.0",
    author="Donia Gasmi",
    description="Semantic entropy-based uncertainty quantification for LLMs",
    packages=find_packages(),
    install_requires=[
        "sentence-transformers>=2.2.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
    ],
    python_requires=">=3.8",
)