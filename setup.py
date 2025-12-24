#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script for ARSLM (Adaptive Reasoning Semantic Language Model).

This setup.py is maintained for backward compatibility.
For modern installations, pyproject.toml is the primary configuration.
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Ensure Python version
if sys.version_info < (3, 8):
    sys.exit('Python 3.8 or higher is required for ARSLM.')

# Get the absolute path of this file
HERE = Path(__file__).parent.resolve()

# Read long description from README
def read_file(filename: str) -> str:
    """Read content from a file."""
    filepath = HERE / filename
    if filepath.exists():
        with open(filepath, encoding='utf-8') as f:
            return f.read()
    return ''

# Get version from __init__.py
def get_version() -> str:
    """Extract version from __init__.py."""
    init_file = HERE / 'arslm' / '__init__.py'
    with open(init_file, encoding='utf-8') as f:
        for line in f:
            if line.startswith('__version__'):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
    raise RuntimeError('Unable to find version string.')

# Read requirements
def read_requirements(filename: str) -> list:
    """Read requirements from a file."""
    filepath = HERE / filename
    if filepath.exists():
        with open(filepath, encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

# Package metadata
NAME = 'arslm'
VERSION = get_version()
DESCRIPTION = 'Adaptive Reasoning Semantic Language Model - Lightweight AI engine'
LONG_DESCRIPTION = read_file('README.md')
AUTHOR = 'Benjamin Amaad Kama'
AUTHOR_EMAIL = 'benjokama@hotmail.fr'
URL = 'https://github.com/benjaminpolydeq/ARSLM'
LICENSE = 'MIT'
PYTHON_REQUIRES = '>=3.8'

# Core dependencies
INSTALL_REQUIRES = [
    'numpy>=1.21.0,<2.0.0',
    'torch>=2.0.0',
    'transformers>=4.30.0',
    'tokenizers>=0.13.0',
    'sentencepiece>=0.1.99',
    'fastapi>=0.100.0',
    'uvicorn[standard]>=0.23.0',
    'pydantic>=2.0.0',
    'pydantic-settings>=2.0.0',
    'python-multipart>=0.0.6',
    'aiofiles>=23.0.0',
    'httpx>=0.24.0',
    'click>=8.1.0',
    'tqdm>=4.65.0',
    'requests>=2.31.0',
    'pyyaml>=6.0',
    'python-dotenv>=1.0.0',
    'rich>=13.0.0',
    'loguru>=0.7.0',
]

# Optional dependencies
EXTRAS_REQUIRE = {
    'dev': [
        'pytest>=7.4.0',
        'pytest-cov>=4.1.0',
        'pytest-asyncio>=0.21.0',
        'pytest-mock>=3.11.0',
        'black>=23.7.0',
        'isort>=5.12.0',
        'flake8>=6.1.0',
        'mypy>=1.5.0',
        'pylint>=2.17.0',
        'pre-commit>=3.3.0',
    ],
    'docs': [
        'mkdocs>=1.5.0',
        'mkdocs-material>=9.2.0',
        'mkdocstrings[python]>=0.22.0',
    ],
    'streamlit': [
        'streamlit>=1.25.0',
        'plotly>=5.15.0',
        'pandas>=2.0.0',
    ],
    'jupyter': [
        'jupyter>=1.0.0',
        'notebook>=7.0.0',
        'ipywidgets>=8.0.0',
        'matplotlib>=3.7.0',
    ],
    'database': [
        'sqlalchemy>=2.0.0',
        'alembic>=1.11.0',
        'psycopg2-binary>=2.9.0',
        'redis>=4.6.0',
    ],
}

# Add 'all' extra that includes everything
EXTRAS_REQUIRE['all'] = list(set(sum(EXTRAS_REQUIRE.values(), [])))

# Classifiers
CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Education',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Topic :: Text Processing :: Linguistic',
]

# Keywords
KEYWORDS = [
    'ai', 'machine-learning', 'nlp', 'language-model', 'chatbot',
    'deep-learning', 'transformers', 'attention', 'conversational-ai'
]

# Entry points for CLI commands
ENTRY_POINTS = {
    'console_scripts': [
        'arslm=arslm.cli.commands:main',
        'arslm-server=arslm.api.server:run_server',
        'arslm-train=arslm.cli.train:train_cli',
        'arslm-eval=arslm.cli.evaluate:eval_cli',
    ],
}

# Project URLs
PROJECT_URLS = {
    'Bug Tracker': f'{URL}/issues',
    'Documentation': 'https://arslm.readthedocs.io',
    'Source Code': URL,
    'Changelog': f'{URL}/blob/main/CHANGELOG.md',
}

# Setup configuration
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    project_urls=PROJECT_URLS,
    license=LICENSE,
    packages=find_packages(exclude=['tests', 'tests.*', 'docs', 'examples']),
    include_package_data=True,
    package_data={
        'arslm': [
            'py.typed',
            '*.json',
            '*.yaml',
            '*.yml',
            'models/*.pt',
            'models/*.bin',
            'config/*.json',
        ],
    },
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points=ENTRY_POINTS,
    classifiers=CLASSIFIERS,
    keywords=KEYWORDS,
    zip_safe=False,
)

# Print post-installation message
if __name__ == '__main__':
    print('\n' + '='*70)
    print('✅ ARSLM installation complete!')
    print('='*70)
    print('\n📖 Quick Start:')
    print('   from arslm import ARSLM')
    print('   model = ARSLM()')
    print('   response = model.generate("Hello, world!")')
    print('\n📚 Documentation: https://arslm.readthedocs.io')
    print('🐛 Issues: https://github.com/benjaminpolydeq/ARSLM/issues')
    print('='*70 + '\n')