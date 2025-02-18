# Configuration file for the Sphinx documentation builder.

import os
import sys

# -- Path setup --------------------------------------------------------------
# If your package is one level above 'doc/', add that directory to sys.path.
# This allows Sphinx to find your 'portfolio_management' package automatically.
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------
project = 'Portfolio Management'
copyright = '2024'
author = 'University of Chicago Financial Mathematics'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',  # For NumPy/Google style docstrings
    'sphinx.ext.viewcode',  # Add "View Source" links on the docs
    # 'myst_parser',         # Uncomment if you want to parse .md files
]

# Generate autosummary even if no references
autosummary_generate = True

add_module_names = False

# Napoleon settings (for NumPy style)
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# -- Options for HTML output -------------------------------------------------
html_theme = 'pydata_sphinx_theme'

# Optional theme options
html_theme_options = {
    'github_url': 'https://github.com/fernando-urbano/portfolio-management',
    # You can add more links or theme customizations here
}

# If you have a custom logo:
# html_logo = '_static/logo.png'



# -- Options for EPUB output
epub_show_urls = 'footnote'

# -- GitHub Pages ------------------------------------------------------------
# This ensures GitHub Pages doesn’t ignore underscores (_build, _static, etc.)
html_extra_path = ['.nojekyll']  # Will copy an empty .nojekyll file to your build

# -- In case you have your .rst files or static assets in other subfolders:
# templates_path = ['_templates']
html_static_path = ['_static']
html_css_files = [
    'custom.css',  # or whatever you named it
]
