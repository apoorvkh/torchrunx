import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

# Configuration file for the Sphinx documentation builder.

project = "torchrunx"
github_username = "apoorvkh"
github_repository = "torchrunx"
html_theme = "furo"

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_toolbox.sidebar_links",
    "sphinx_toolbox.github",
    "sphinx.ext.autodoc.typehints",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
]

autodoc_typehints = "description"

autodoc_mock_imports = ["torch", "fabric", "cloudpickle", "typing_extensions"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]


## Link code to Github source
# From: https://github.com/scikit-learn/scikit-learn/blob/main/doc/sphinxext/github_link.py

import inspect
import os
import subprocess
import sys
from operator import attrgetter


try:
    revision = (
        subprocess.check_output("git rev-parse --short HEAD".split()).strip().decode("utf-8")
    )
except (subprocess.CalledProcessError, OSError):
    print("Failed to execute git to get revision")
    revision = None

url_fmt = (
    f"https://github.com/{github_username}/{github_repository}/"
    "blob/{revision}/{package}/{path}#L{lineno}"
)

def linkcode_resolve(domain, info):
    if revision is None:
        return
    if domain not in ("py", "pyx"):
        return
    if not info.get("module") or not info.get("fullname"):
        return

    class_name = info["fullname"].split(".")[0]
    module = __import__(info["module"], fromlist=[class_name])
    obj = attrgetter(info["fullname"])(module)

    # Unwrap the object to get the correct source
    # file in case that is wrapped by a decorator
    obj = inspect.unwrap(obj)

    try:
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None
    if not fn:
        try:
            fn = inspect.getsourcefile(sys.modules[obj.__module__])
        except Exception:
            fn = None
    if not fn:
        return

    fn = os.path.relpath(fn, start=os.path.dirname(__import__(package).__file__))
    try:
        lineno = inspect.getsourcelines(obj)[1]
    except Exception:
        lineno = ""
    return url_fmt.format(revision=revision, package=package, path=fn, lineno=lineno)
