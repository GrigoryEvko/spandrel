#!/usr/bin/env python3
"""Setup.py for git subdirectory installations."""
from setuptools import setup, find_packages

# Explicitly provide package info for git subdirectory installs
setup(
    name="spandrel",
    version="0.4.1",
    packages=find_packages(),
    package_data={
        "spandrel": [
            "**/*.py",
            "**/LICENSE",
        ]
    },
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "torchvision",
        "safetensors",
        "numpy",
        "einops",
        "typing_extensions",
    ],
)