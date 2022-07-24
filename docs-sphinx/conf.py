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
extensions = [
    'sphinx_external_toc',
    'sphinx_copybutton',
    'sphinx_design',
    'myst_nb',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "_templates", "jupyter_execute", ".jupyter_cache"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
html_show_sourcelink = True
html_theme_options = {
  "logo": {
      "image_light": "logo-300px.png",
      "image_dark": "logo-300px-white.png",
  },
  "github_url": "https://github.com/scikit-hep/awkward",
  "collapse_navigation": True,
  # Add light/dark mode and documentation version switcher:
  "navbar_end": ["theme-switcher", "navbar-icon-links"],
  "footer_items": ["copyright", "sphinx-version", "funding"],
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static", "_assets"]
html_css_files = ['css/awkward.css']

# MyST settings
myst_enable_extensions = [
  "colon_fence",
]

nb_execution_mode = "cache"
nb_execution_raise_on_error = True

# Additional stuff
master_doc = "index"

import os
import sys
import subprocess

subprocess.check_call(["doxygen", os.path.join("docs-doxygen", "Doxyfile")], cwd="..")

exec(open("prepare_docstrings.py").read(), dict(globals()))

current_dir = os.path.dirname(os.path.realpath(__file__))
docgen = os.path.join(current_dir, "..", "dev", "generate-kerneldocs.py")
subprocess.check_call([sys.executable, docgen])

#exec(open("make_changelog.py").read(), dict(globals()))
