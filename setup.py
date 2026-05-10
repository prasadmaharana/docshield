from setuptools import setup, find_packages

setup(
    name="docshield",
    version="0.1.0",
    description="Computer vision pipeline for PII redaction in scanned financial documents",
    author="Prasad Maharana",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "ultralytics>=8.4.0",
        "pypdfium2>=4.0.0",
        "Pillow>=9.0.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "ocr": ["paddleocr>=3.0.0", "paddlepaddle>=3.0.0"],
        "app": ["streamlit>=1.20.0", "pandas>=1.5.0"],
        "train": ["datasets>=2.0.0", "faker>=15.0.0", "reportlab>=3.6.0", "pyyaml>=6.0"],
    },
)
