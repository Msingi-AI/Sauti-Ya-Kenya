from setuptools import setup, find_packages

setup(
    name="sauti-ya-kenya",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "pytest>=7.0.0",
        "pytest-asyncio>=0.20.0",
        "httpx>=0.24.0",
        "pesq>=0.0.3",
        "pystoi>=0.3.3",
        "librosa>=0.10.0"
    ]
)
