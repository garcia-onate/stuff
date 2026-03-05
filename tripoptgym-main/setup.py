"""Setup file for TripOpt Gym package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tripoptgym",
    version="1.0.0",
    author="Joe Wakeman",
    author_email="",
    description="A reinforcement learning environment for freight train optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/joewak/tripoptgym",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "gymnasium>=0.28.0",
        "pygame>=2.0.0",
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "torch>=1.10.0",
        "imageio>=2.9.0",
        "imageio-ffmpeg>=0.4.0",
        "pyyaml>=5.4.0",
        "tensorboard>=2.10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "tripopt=tripoptgym.scripts.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "tripoptgym": ["../configs/*.yaml"],
    },
)
