"""Setup script for the quantum_retrieval package."""
from setuptools import setup, find_packages

setup(
    name="quantum_retrieval",
    version="0.1.0",
    packages=find_packages(),
    package_dir={"": "."},
    python_requires=">=3.10",
    install_requires=[
        "wikipedia-api",
        "httpx",
        "transformers",
        "torch",
        "numpy",
        "fastapi",
        "uvicorn",
        "h5py",
        "pytest",
    ],
)
