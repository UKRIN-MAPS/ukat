# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath('..'))

# Some notebooks have stale 'ipython2' metadata
from IPython.lib.lexers import IPython3Lexer
from sphinx.highlighting import lexers
lexers['ipython2'] = IPython3Lexer()

# -- Project information -----------------------------------------------------
project = 'UKAT'
copyright = f'{datetime.now().year}, UKRIN-MAPS'
author = 'UKRIN-MAPS'
release = '0.7.3'

# -- General configuration ---------------------------------------------------
extensions = ['sphinx.ext.napoleon',
              'sphinx.ext.viewcode',
              'sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.todo',
              'myst_parser',
              'nbsphinx',
              'nbsphinx_link']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

suppress_warnings = ['myst.header']

napoleon_include_init_with_doc = True
autodoc_member_order = 'bysource'

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    'navigation_depth': 2,
    'collapse_navigation': True,
}
html_context = {
    "sidebar_external_links_caption": "Links",
    "sidebar_external_links": [
        (
            '<i class="fa fa-github fa-fw"></i> Source Code',
            "https://github.com/UKRIN-MAPS/ukat",
        ),
        (
            '<i class="fa fa-bug fa-fw"></i> Issue Tracker',
            "https://github.com/UKRIN-MAPS/ukat/issues",
        ),
        (
            '<i class="fa fa-file-text fa-fw"></i> Citation',
            "https://zenodo.org/badge/latestdoi/256993671",
        ),
    ],
}
html_style = os.path.join("css", "custom.css")
html_favicon = os.path.join("_static", "favicon.ico")
html_logo = os.path.join("_static", "logo.png")
