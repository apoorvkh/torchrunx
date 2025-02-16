"""Configuration file for the Sphinx documentation builder."""
from glob import glob
import os
import shutil

shutil.copyfile("../README.md", "source/README.md")
shutil.copyfile("../CONTRIBUTING.md", "source/contributing.md")

os.makedirs("source/examples/scripts", exist_ok=True)
[shutil.copy(f, "source/examples/scripts/") for f in glob("../scripts/examples/*.py")]
html_extra_path = list(glob("source/examples/scripts/*.py"))

project = "torchrunx"
copyright = 'Apoorv Khandelwal & Peter Curtin'
github_username = "apoorvkh"
github_repository = "torchrunx"
html_theme = "furo"
language = "en"

extensions = [
    "sphinx.ext.autodoc",
    "myst_parser",  # support markdown
    "sphinx.ext.intersphinx",  # link to external docs
    "sphinx.ext.napoleon",  # for google style docstrings
    "sphinx.ext.linkcode",  # link to github source
    # sidebar
    "sphinx_toolbox.sidebar_links",
    "sphinx_toolbox.github",
]

maximum_signature_line_length = 90
autodoc_member_order = "bysource"

intersphinx_mapping = {
    'python': ('https://docs.python.org/3.9', None),
}

from docs.linkcode_github import generate_linkcode_resolve_fn
linkcode_resolve = generate_linkcode_resolve_fn(project, github_username, github_repository)
