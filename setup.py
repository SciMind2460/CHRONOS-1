from setuptools import setup, find_packages

setup(
    name="chronostextgen",
    version="1.0.0",
    description="An repo with a small-scale AI model, pre-trained with the WikiText Dataset",
    author="Kurt Heiritz",
    author_email="saarthkarkera@gmail.com",
    url="https://github.com/SciMind2460/chronos",
    packages=find_packages(),
    install_requires=[
        "datasets>=3.0.1",
        "transformers>=4.45.1",
        "torch>=2.4.1",
        "inflect>=7.4.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
