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
import json
import datetime
import runpy
import sys
import subprocess
import pathlib

# -- Project information -----------------------------------------------------

project = "Awkward Array"
copyright = f"{datetime.datetime.now().year}, Awkward Array development team"
author = "Jim Pivarski"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_external_toc",
    "sphinx.ext.intersphinx",
    "myst_nb",
    # Preserve old links
    "sphinx_reredirects",
    "jupyterlite_sphinx",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "_templates", "Thumbs.db", "jupyter_execute", ".*"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_context = {
    "github_user": "scikit-hep",
    "github_repo": "awkward",
    # TODO: set this
    "github_version": os.environ.get("READTHEDOCS_VERSION", "main"),
    "doc_path": "docs",
}
html_theme = "pydata_sphinx_theme"
html_show_sourcelink = True
html_theme_options = {
    "logo": {
        "image_light": "image/logo-300px.png",
        "image_dark": "image/logo-300px-white.png",
    },
    "github_url": "https://github.com/scikit-hep/awkward",
    # Add light/dark mode and documentation version switcher:
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "footer_items": ["copyright", "sphinx-version", "funding"],
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
    "analytics": {
        "plausible_analytics_domain": "awkward-array.org",
        "plausible_analytics_url": "https://views.scientific-python.org/js/plausible.js"
    }
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/awkward.css"]

# MyST settings
myst_enable_extensions = [
    "colon_fence",
]

nb_execution_mode = "cache"
nb_execution_raise_on_error = True
# unpkg is currently _very_ slow
nb_ipywidgets_js = {
    # Load RequireJS, used by the IPywidgets for dependency management
    "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js": {
        "integrity": "sha256-Ae2Vz/4ePdIu6ZyI/5ZGsYnb+m0JlOmKPjt6XZ9JJkA=",
        "crossorigin": "anonymous",
    },
    # Load IPywidgets bundle for embedding.
    "https://cdn.jsdelivr.net/npm/@jupyter-widgets/html-manager@0.20.0/dist/embed-amd.js": {
        "data-jupyter-widgets-cdn": "https://cdn.jsdelivr.net/npm/",
        "crossorigin": "anonymous",
    },
}
# Additional stuff
master_doc = "index"

# Cross-reference existing Python objects
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "numba": ("https://numba.pydata.org/numba-doc/latest", None),
    "arrow": ("https://arrow.apache.org/docs/", None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
}

# Preserve legacy routes
with open("redirects.json") as f:
    redirects = json.load(f)

redirect_html_template_file = "_templates/redirect.html"

# JupyterLite configuration
jupyterlite_dir = "./lite"
# Don't override ipynb format
jupyterlite_bind_ipynb_suffix = False
# We've disabled localstorage, so we must provide the contents explicitly
jupyterlite_contents = ["getting-started/demo/*"]

HERE = pathlib.Path(__file__).parent

# Generate Python docstrings
runpy.run_path(HERE / "prepare_docstrings.py", run_name="__main__")


# Sphinx doesn't usually want content to fit the screen, so we hack the styles for this page
def install_jupyterlite_styles(app, pagename, templatename, context, event_arg) -> None:
    if pagename != "getting-started/try-awkward-array":
        return

    app.add_css_file("css/try-awkward-array.css")


def setup(app):
    app.connect('html-page-context', install_jupyterlite_styles)
