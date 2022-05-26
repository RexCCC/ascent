# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../config/system/run_utils'))

# -- Project information -----------------------------------------------------

project = 'ASCENT'
copyright = '2021, Duke University'
author = 'Musselman ED, Cariello JE, Grill WM, Pelot NA.'

# The full version, including alpha/beta/rc tags
release = 'v1.1.2'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = ['myst_parser','sphinxarg.ext','sphinx.ext.intersphinx','sphinx.ext.autodoc']

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'restructuredtext',
    '.md': 'markdown',
}

import mock

MOCK_MODULES = ['numpy','pandas']

for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_show_copyright = True

html_show_sphinx = False
