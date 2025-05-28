"""
ملف إعداد مشروع باصرة
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="baserah",
    version="0.1.0",
    author="فريق باصرة",
    author_email="info@baserah.ai",
    description="نظام ذكاء اصطناعي ومعرفي مبتكر",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/baserah",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Natural Language :: Arabic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "baserah=src.main:main",
        ],
    },
)
