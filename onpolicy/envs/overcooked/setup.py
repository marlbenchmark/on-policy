#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='overcooked_ai',
      version='0.0.1',
      description='Cooperative multi-agent environment based on Overcooked',
      author='Micah Carroll',
      author_email='micah.d.carroll@berkeley.edu',
      packages=find_packages(),
      install_requires=[
        'numpy',
        'tqdm',
        'gym'
      ]
    )