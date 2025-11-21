from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="numpy-dl",
    version="0.1.0",
    author="NumpyDeepLearning Contributors",
    description="A deep learning framework built from scratch using NumPy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vivekkr1809/NumpyDeepLearning",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pyyaml>=6.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "pandas>=1.3.0",
        "pillow>=8.3.0",
        "scikit-learn>=0.24.0",
    ],
    extras_require={
        "gpu": ["cupy-cuda11x>=11.0.0"],
        "data": ["kaggle>=1.5.0", "datasets>=2.0.0", "huggingface-hub>=0.10.0"],
        "docs": ["sphinx>=4.0.0", "sphinx-rtd-theme>=1.0.0", "sphinx-autodoc-typehints>=1.12.0"],
        "dev": ["pytest>=6.2.0", "pytest-cov>=2.12.0", "black>=21.0", "flake8>=3.9.0", "mypy>=0.910"],
    },
)
