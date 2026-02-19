#!/usr/bin/env python3
"""
Setup script for PySim - Python Train Simulation Package
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Python Train Simulation Package"

# Read requirements from requirements.txt
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    requirements = []
    if os.path.exists(req_path):
        with open(req_path, "r", encoding="utf-8") as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return requirements

setup(
    name="pysim",
    version="1.0.0",
    author="Joseph Wakeman",
    author_email="joseph.wakeman@wabtec.com",
    description="Python Train Simulation Package for FastSim dynamics modeling",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://gitlab.corp.wabtec.com/Joseph.Wakeman/pysim",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "viz": [
            "matplotlib>=3.4.0",
            "plotly>=5.0.0",
            "seaborn>=0.11.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.0.0",
            "matplotlib>=3.4.0",
        ],
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
        "all": [
            "matplotlib>=3.4.0",
            "plotly>=5.0.0", 
            "seaborn>=0.11.0",
            "jupyter>=1.0.0",
            "ipywidgets>=7.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pysim-run=pysim.cli:main_run",
            "pysim-postrun=pysim.cli:main_postrun",
            "pysim-analysis=pysim.cli:main_analysis",
        ],
    },
    include_package_data=True,
    package_data={
        "pysim": [
            "data/*.json",
            "data/*.csv",
            "data/*.txt",
        ],
    },
    zip_safe=False,
)