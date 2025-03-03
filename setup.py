from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gitmetrics",
    version="0.1.0",
    author="GitMetrics Team",
    author_email="example@example.com",
    description="A Python package for analyzing Git repositories",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/gitmetrics",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "gitpython>=3.1.0",
        "pandas>=1.0.0",
        "numpy>=1.18.0",
        "matplotlib>=3.1.0",
        "plotly>=4.14.0",
        "networkx>=2.5",
        "dash>=2.0.0",
        "dash-bootstrap-components>=1.0.0",
        "scikit-learn>=0.24.0",
        "requests>=2.25.0",
        "tqdm>=4.50.0",
        "orjson>=3.6.0",  # Optional: for improved JSON serialization performance with Dash 2.0
    ],
    entry_points={
        "console_scripts": [
            "gitmetrics=gitmetrics.cli:main",
        ],
    },
)
