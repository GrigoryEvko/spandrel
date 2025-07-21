#!/usr/bin/env python3
"""Setup.py for git subdirectory installations."""
from setuptools import setup, find_packages

# Explicitly provide package info for git subdirectory installs
setup(
    name="spandrel_extra_arches",
    version="0.2.0",
    packages=find_packages(),
    package_data={
        "spandrel_extra_arches": [
            "**/*.py",
            "**/LICENSE",
        ]
    },
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "spandrel>=0.4.0",
        "torch",
        "torchvision",
        "numpy",
        "einops",
        "typing_extensions",
    ],
)