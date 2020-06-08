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
# sys.path.insert(0, os.path.abspath("."))

# -- Project information -----------------------------------------------------

project = "Awkward Array"
copyright = "2020, Jim Pivarski"
author = "Jim Pivarski"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = []

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_show_sourcelink = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Additional stuff
master_doc = "index"

import os
import subprocess

subprocess.check_call(["doxygen", os.path.join("docs-doxygen", "Doxyfile")], cwd="..")

exec(open("prepare_docstrings.py").read(), dict(globals()))

pythongen = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "dev", "genpython.py"
)
identities = os.path.join(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "src", "cpu-kernels"
    ),
    "identities.cpp",
)
operations = os.path.join(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "src", "cpu-kernels"
    ),
    "operations.cpp",
)
reducers = os.path.join(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "src", "cpu-kernels"
    ),
    "reducers.cpp",
)
getitem = os.path.join(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "src", "cpu-kernels"
    ),
    "getitem.cpp",
)
subprocess.check_call(["python", pythongen, identities, operations, reducers, getitem])

exec(open("make_changelog.py").read(), dict(globals()))
