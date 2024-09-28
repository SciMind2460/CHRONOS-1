# setup.py

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README.md
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='chronos_textgen',  # Replace with your desired package name
    version='1.0.0',
    author='Saarth Karkera',
    author_email='saarthkarkera@gmail.com',
    description='A transformer-based text generation model using the WikiText dataset.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/SciMind2460/chronos',  # Replace with your project's URL
    packages=find_packages(),
    install_requires=[
        'torch>=7.4.0',
        'transformers>=4.45.1',
        'datasets>=3.0.1',
        'inflect>=7.4.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Replace with your license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
    include_package_data=True,  # Include files specified in MANIFEST.in
    entry_points={
        'console_scripts': [
            'train-textgen=chronos.scripts.main:main',  # Optional: Create a CLI entry point
        ],
    },
)
