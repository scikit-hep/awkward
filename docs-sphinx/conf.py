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
import datetime
import re
import sys
import subprocess

# sys.path.insert(0, os.path.abspath("."))

# -- Project information -----------------------------------------------------

project = "Awkward Array"
copyright = f"{datetime.datetime.now().year}, Awkward Array development team"
author = "Jim Pivarski"

release = os.environ["DOCS_VERSION"]
version_match = re.match(r"(\d+)\.(\d+)\.(\d+)", release)
if not version_match:
    raise RuntimeError("Invalid version given", release)
version = ".".join(version_match.groups()[:2])

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    "sphinxext.opengraph",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# Specify a canonical version
html_baseurl = "https://awkward-array.org/doc/main/"

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_logo = "../docs-img/logo/logo-300px-white.png"

html_context = {
    "github_user": "scikit-hep",
    "github_repo": "awkward",
    "github_version": "main",
    "doc_path": "docs-sphinx",
}
html_theme = "pydata_sphinx_theme"
html_show_sourcelink = True
html_theme_options = {
    "logo": {
        "image_light": "logo-300px.png",
        "image_dark": "logo-300px-white.png",
    },
    "github_url": "https://github.com/scikit-hep/awkward",
    # Add light/dark mode and documentation version switcher:
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "footer_items": ["copyright", "sphinx-version"],
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/awkward",
            "icon": "fab fa-python",
        }
    ],
    "use_edit_page_button": True,
    "external_links": [
        {
            "name": "Contributor guide",
            "url": "https://github.com/scikit-hep/awkward/blob/main/CONTRIBUTING.md",
        },
        {
            "name": "Release history",
            "url": "https://github.com/scikit-hep/awkward/releases",
        },
    ],
}

ogp_custom_meta_tags = [
    '<meta name="robots" content="noindex,nofollow">',
]

# Don't show version for offline builds by default
if "DOCS_SHOW_VERSION" in os.environ:
    html_theme_options["switcher"] = {
        "json_url": "https://awkward-array.org/doc/switcher.json",
        "version_match": version,
    }
    html_theme_options["navbar_start"] = ["navbar-logo", "version-switcher"]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static", "_image"]

# Additional stuff
master_doc = "index"

subprocess.check_call(["doxygen", os.path.join("docs-doxygen", "Doxyfile")], cwd="..")

exec(open("prepare_docstrings.py").read(), dict(globals()))

current_dir = os.path.dirname(os.path.realpath(__file__))
docgen = os.path.join(current_dir, "..", "dev", "generate-kerneldocs.py")
subprocess.check_call([sys.executable, docgen])
