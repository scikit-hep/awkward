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

sys.path.insert(0, "/home/angus/Git/awkward")#os.path.dirname(os.getcwd()))


src_path = os.path.abspath(os.path.join(os.getcwd(), "..", "src"))

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
    'sphinx_copybutton',
]

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

autosummary_generate = True
autosummary_ignore_module_all = True
autosummary_imported_members = True

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "_templates"]
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_show_sourcelink = False
html_logo = "../docs-img/logo/logo-300px-white.png"
html_theme_options = {"logo_only": True, "sticky_navigation": False}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

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

    blob_path = os.path.relpath(path, start=src_path)
    return f"https://github.com/scikit-hep/awkward-1.0/blob/{revision}/{blob_path}{line_spec}"


# https://stackoverflow.com/questions/38765577/overriding-sphinx-autodoc-alias-of-for-import-of-private-class
def skip(*args):
    ...


# https://stackoverflow.com/questions/14141170/how-can-i-just-list-undocumented-members-with-sphinx-autodoc
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
def modify_signatures(app, what, name, obj, options, signature, return_annotation):
    ...


def setup(app):
    app.connect("autodoc-process-signature", modify_signatures)
