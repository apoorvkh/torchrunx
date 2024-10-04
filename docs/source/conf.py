import os
import sys

sys.path.insert(0, os.path.abspath('../../src'))

# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'torchrunx'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'myst_parser',
    'sphinx_toolbox.sidebar_links',
    'sphinx_toolbox.github',
    "sphinx_autodoc_typehints",
]

typehints_defaults = 'comma'

github_username = 'apoorvkh'
github_repository = 'torchrunx'

autodoc_mock_imports = ['torch', 'fabric', 'cloudpickle', 'typing_extensions']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'furo'

# -- Options for EPUB output
epub_show_urls = 'footnote'

# code block syntax highlighting
#pygments_style = 'sphinx'
