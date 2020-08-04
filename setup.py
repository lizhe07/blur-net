# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 00:03:26 2020

@author: Zhe
"""

from blurnet import __version__
from setuptools import setup, find_packages

setup(
    name="blurnet",
    version=__version__,
    author='Zhe Li',
    python_requires='>=3',
    packages=find_packages(),
    install_requires=['torch', 'jarvis'],
)
