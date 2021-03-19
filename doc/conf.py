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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
from datetime import date

# -- Project information -----------------------------------------------------

project = 'CADET-Match'
copyright = f'2017-{date.today().year}'
author = 'CADET-Match Authors'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.todo',
    'sphinx.ext.githubpages',
    'sphinx_sitemap',
    'sphinxcontrib.bibtex',
    #'sphinxcontrib.tikz',
]

bibtex_bibfiles = ['literature.bib']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Support for labelling and referencing equations
numfig = True
math_numfig = True
numfig_secnum_depth = 2
math_eqref_format = '{number}'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'
html_theme_options = {
    # 'logo': 'cadet_logo.png',
    # 'description': 'Chromatography Analysis and Design Toolkit',
    'sidebar_collapse': True,
    'fixed_sidebar': True,
    'show_powered_by': False,
}

html_favicon = '_static/cadet_icon.png'
html_title = 'CADET-Match'
html_baseurl = 'https://cadet.github.io/'
html_static_path = ['_static']
html_extra_path = ['robots.txt', 'google7a5fbf15028eb634.html']
html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'searchbox.html',
    ]
}

html_style = 'css/custom.css'

# Options for LaTeX output
# ------------------------

# The paper size ('letter' or 'a4').
latex_paper_size = "a4"

# The font size ('10pt', '11pt' or '12pt').
latex_font_size = '10pt'

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, document class [howto/manual]).
latex_documents = [
        (
            "modelling/index",
            "manual.pdf",
            "CADET Manual",
            "CADET Authors",
            "manual",
        )
    ]

