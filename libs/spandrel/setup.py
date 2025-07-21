#!/usr/bin/env python3
"""Setup.py for git subdirectory installations."""
from setuptools import setup, find_packages

# Explicitly provide package info for git subdirectory installs
setup(
    name="spandrel",
    version="0.4.1",
    packages=find_packages(include=["spandrel*"]),
    python_requires=">=3.8",
)