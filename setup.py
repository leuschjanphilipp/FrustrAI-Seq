"""Setup script for FrustrAISeq"""

from setuptools import setup, find_packages
import os

# Define minimal core requirements (without version pins for flexibility)
# When installing in conda environments, most dependencies are already satisfied
core_requirements = [
    "torch>=2.5.0",
    "lightning>=2.5.0",
    "transformers>=5.0.0",
    "peft>=0.17.0",
    "pandas>=1.5.0",
    "numpy>=1.21.0",
    "biopython>=1.79",
    "pyyaml>=5.4",
    "datasets>=2.0.0",
    "pyarrow>=10.0.0",
    "scipy>=1.7.0",
    "huggingface_hub>=0.16.0", 
]

# Optional requirements for HuggingFace integration
extras_require = {
    "dev": [
        "pytest>=7.0.0",
        "black>=22.0.0",
        "flake8>=4.0.0",
    ],
}

setup(
    name="frustraiseq",
    version="0.1.0",
    description="FrustrAISeq - Per-Residue Local Energetic Frustration Prediction using Sequence-only Deep Learning",
    author="Jan Leusch",
    author_email="jan.leusch@helmholtz-munich.de",
    url="https://github.com/leuschjanphilipp/FrustrAI-Seq",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=core_requirements,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "frustraiseq=frustraiseq.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="protein bioinformatics deep-learning local energetic frustration",
)
