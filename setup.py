#!/usr/bin/env python
from pkg_resources import get_distribution
from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

version = get_distribution("autolabel").version

setup(
    packages=find_packages(),
    install_requires=[
        'click',
        'more-itertools',
        'torchvision',
        'torch',
        'pillow',
        'numpy'
    ],
    entry_points='''
        [console_scripts]
        autolabel=autolabel.cli:main
    ''',
    url='https://github.com/walwe/autolabel',
    version=version,
    author='walwe',
    python_requires='>=3.6',
    description='Autolabel is an image labeling tool using Neural Network',
    long_description_content_type="text/markdown",
    long_description=long_description,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
