# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ALF'
copyright = '2024, B. Nebgen, V. Grizzi, N. Fedik, N. Lubbers, Y.-W. Li, S. Tretiak'
author = 'B. Nebgen, V. Grizzi, N. Fedik, N. Lubbers, Y.-W. Li, S. Tretiak'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys
sys.path.insert(0, os.path.abspath('../../alframework'))

extensions = [
    'sphinx.ext.autodoc',  # Automatically document from docstrings
    'sphinx.ext.napoleon', # Support for Google and NumPy docstrings
    'sphinx.ext.viewcode', # Add links to highlighted source code
    'sphinx.ext.autosummary', # Generate summary tables for modules/classes/functions
    'sphinx_autodoc_typehints', # Better type hint formatting
    'myst_parser',  # Support for .md files
            ]           

templates_path = ['_templates']
exclude_patterns = []
autodoc_mock_imports  = ['neurochem_interface', 'ase_interface', 'anitraintools']

autodoc_default_options = {
    'members': True,  # Document class members
    'undoc-members': True,  # Include members without docstrings
    'show-inheritance': True,  # Show class inheritance
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
