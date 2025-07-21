#!/usr/bin/env python3
"""Setup.py for git subdirectory installations."""
from setuptools import setup, find_packages

# Explicitly provide package info for git subdirectory installs
setup(
    name="spandrel_extra_arches",
    version="0.2.0",
    packages=find_packages(include=["spandrel_extra_arches*"]),
    python_requires=">=3.8",
)