#
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import inspect
import subprocess
import sys
import operator
import os
import os.path

try:
    import awkward
except ImportError:
    raise ImportError("Could not find Awkward on sys.path. Please ensure that it is installed.")

src_path = os.path.join(awkward.__path__[0], "..")

# -- Project information -----------------------------------------------------

project = "Awkward Array"
copyright = "2020, Jim Pivarski"
author = "Jim Pivarski"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
    "sphinx_reredirects",
    "sphinx_external_toc",
    'sphinx_copybutton',
    'sphinx_design',
    'myst_nb',
]

myst_enable_extensions = [
  "colon_fence",
]

nb_execution_mode = "cache"

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

autosummary_generate = True
autosummary_ignore_module_all = False
autosummary_imported_members = True

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "_templates", "jupyter_execute"]
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
html_show_sourcelink = False
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
html_static_path = ["_static"]
html_css_files = [
    'awkward.css',
]

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "numba": ("https://numba.pydata.org/numba-doc/latest", None),
}


# Additional stuff
master_doc = "index"

revision = (
    subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE)
    .stdout.decode("utf-8")
    .strip()
)


def linkcode_resolve(domain, info):
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    # Load module
    mod = sys.modules.get(modname)

    if mod is None:
        return None

    # Lookup named object
    getter = operator.attrgetter(fullname)
    try:
        obj = getter(mod)
    except AttributeError:
        return None

    # Try to get the source file
    try:
        path = inspect.getsourcefile(inspect.unwrap(obj))
    except TypeError:
        path = None
    if path is None:
        return None

    # Try to get the line number
    try:
        source, line_num = inspect.getsourcelines(obj)
    except OSError:
        line_spec = ""
    else:
        line_spec = f"#L{line_num}-L{line_num + len(source) - 1}"

    src_blob_path = os.path.relpath(path, start=src_path)

    assert revision
    return f"https://github.com/scikit-hep/awkward-1.0/blob/{revision}/src/{src_blob_path}{line_spec}"


# https://stackoverflow.com/questions/38765577/overriding-sphinx-autodoc-alias-of-for-import-of-private-class
def skip(*args):
    ...


# https://stackoverflow.com/questions/14141170/how-can-i-just-list-undocumented-members-with-sphinx-autodoc
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
def modify_signatures(app, what, name, obj, options, signature, return_annotation):
    ...


def setup(app):
    app.connect("autodoc-process-signature", modify_signatures)
