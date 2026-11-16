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
import awkward
import datetime
import os
import pathlib
import sys

# The awkward_docs extension
sys.path.insert(0, str(pathlib.Path(__file__).parent / "_ext"))

# -- Project information -----------------------------------------------------

project = "Awkward Array"
copyright = f"2019-{datetime.datetime.now().year}, the Awkward Array developers"
author = "Jim Pivarski"

parts = awkward.__version__.split(".")
version = ".".join(parts[:2])
release = ".".join(parts)

# -- Environment variables ---------------------------------------------------
# Allow the CI to set version_match="main"
version_match = os.environ.get("DOCS_VERSION", version)
canonical_version = os.environ.get("DOCS_CANONICAL_VERSION")
report_analytics = os.environ.get("DOCS_REPORT_ANALYTICS", False)
show_version_switcher = os.environ.get("DOCS_SHOW_VERSION", False)
run_cuda_notebooks = os.environ.get("DOCS_RUN_CUDA", False)

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_external_toc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "autoapi.extension",
    "myst_nb",
    "sphinx_llm.txt",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "awkward_docs",
]

# -- sphinx-autoapi configuration --------------------------------------------
autoapi_dirs = ["../src/awkward"]
autoapi_type = "python"
autoapi_ignore = []
autoapi_options = [
    "members",
    "undoc-members",
    "private-members",
    "special-members",
    "show-module-summary",
    "imported-members",
]
autoapi_root = "reference/generated"
autoapi_keep_files = True  # keep generated RST for inspection
autoapi_add_toctree_entry = False  # manual toctree control
autoapi_own_page_level = "function"  # each function/class gets own page

# Napoleon settings (Google-style docstrings)
napoleon_google_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True


autoapi_template_dir = "_autoapi_templates"

from awkward_docs import autoapi_prepare_jinja_env  # noqa: E402


# Specify a canonical version
if canonical_version is not None:
    html_baseurl = f"https://awkward-array.org/doc/{canonical_version}/"

    # Build sitemap on main
    if version_match == canonical_version:
        extensions.append("sphinx_sitemap")
        # Sitemap URLs are relative to `html_baseurl`
        sitemap_url_scheme = "{link}"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "_templates", "_autoapi_templates", "Thumbs.db", "jupyter_execute", ".*"]

# -- Options for LLM-friendly output -----------------------------------------

# The default would be the awkward package metadata summary
llms_txt_description = (
    "Documentation of Awkward Array, a library for nested, variable-sized data"
    " (arbitrary-length lists, records, mixed types, and missing data) using"
    " NumPy-like idioms."
)

# Run the Markdown sub-build after the HTML build so that it can reuse the
# doctrees; autoapi writes generated files into the source tree, so the two
# builds must not run at the same time.
llms_txt_build_parallel = False

# Write only page.html.md, not also page.md
llms_txt_suffix_mode = "append"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_context = {
    "github_user": "scikit-hep",
    "github_repo": "awkward",
    "github_version": "main",
    "doc_path": "docs",
}
html_theme = "pydata_sphinx_theme"
html_show_sourcelink = True
html_theme_options = {
    "logo": {
        "image_light": "_static/image/logo-300px.png",
        "image_dark": "_static/image/logo-300px-white.png",
    },
    "github_url": "https://github.com/scikit-hep/awkward",
    # Add light/dark mode and documentation version switcher:
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "footer_start": ["copyright", "sphinx-version", "funding"],
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/awkward",
            "icon": "fa-brands fa-python",
        },
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

# Disable analytics for previews
if report_analytics:
    html_theme_options["analytics"] = {
        "plausible_analytics_domain": "awkward-array.org",
        "plausible_analytics_url": "https://views.scientific-python.org/js/plausible.js",
    }

# Don't show version for offline builds by default
if show_version_switcher:
    html_theme_options["switcher"] = {
        "json_url": "https://awkward-array.org/doc/switcher.json",
        "version_match": version_match,
    }
    html_theme_options["navbar_start"] = ["navbar-logo", "version-switcher"]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/awkward.css"]
html_js_files = ["js/awkward.js"]

# MyST settings
myst_enable_extensions = ["colon_fence", "deflist"]

nb_execution_mode = "cache"
# Share the cache with the Markdown sub-build, which has its own output directory
nb_execution_cache_path = str(pathlib.Path(__file__).parent / "_build" / ".jupyter_cache")
nb_execution_raise_on_error = True
# Text outputs for the Markdown sub-build (lower is preferred)
nb_mime_priority_overrides = [
    ("llms-markdown", "text/markdown", 10),
    ("llms-markdown", "text/plain", 20),
]
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
nb_execution_show_tb = True
# Increase cell execution timeout (seconds)
nb_execution_timeout = 120  # two minutes per cell

if not run_cuda_notebooks:
    nb_execution_excludepatterns = [
        # We have no CUDA executors, so disable this
        "user-guide/how-to-use-in-numba-cuda.ipynb",
        # We have no cppyy 3.0.1 yet, so disable this
        "user-guide/how-to-use-in-cpp-cppyy.ipynb",
    ]

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


linkcheck_ignore = [
    r"^https?:\/\/github\.com\/.*$",
    r"^https?:\/\/$",  # Bare https:// allowed
]
# Eventually we need to revisit these
if (datetime.date.today() - datetime.date(2022, 12, 13)) < datetime.timedelta(days=30):
    linkcheck_ignore.extend(
        [
            r"^https:\/\/doi.org\/10\.1051\/epjconf\/202024505023$",
            r"^https:\/\/doi.org\/10\.1051\/epjconf\/202125103002$",
        ]
    )
