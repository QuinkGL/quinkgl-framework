# Copyright 2026 Ali Seyhan, Baki Turhan
"""Sphinx configuration for QuinkGL documentation."""

from __future__ import annotations

import os
import sys

# Allow autodoc to import quinkgl
sys.path.insert(0, os.path.abspath("../src"))

import quinkgl

project = "QuinkGL"
copyright = "2026, Ali Seyhan, Baki Turhan"
author = "Ali Seyhan, Baki Turhan"
version = quinkgl.__version__
release = version

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# Warnings as errors in CI
nitpicky = True
