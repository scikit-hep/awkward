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
import re
import subprocess

import awkward
import datetime
import os
import pathlib

# -- Project information -----------------------------------------------------

project = "Awkward Array"
copyright = f"{datetime.datetime.now().year}, Awkward Array development team"
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
    # Preserve old links
    # "jupyterlite_sphinx",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
]

# -- sphinx-autoapi configuration --------------------------------------------
autoapi_dirs = ["../src/awkward"]
autoapi_type = "python"
autoapi_ignore = []
autoapi_options = [
    "members",
    "undoc-members",
    "private-members",
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


# -- Monkey-patch autoapi to produce ak.* file names -------------------------
# autoapi generates paths from Python module structure (awkward.operations.ak_flatten.flatten).
# We remap to public API names (ak.flatten) to match existing URLs.

def _awkward_to_ak(name):
    """Map internal module paths to public API names.

    Mirrors the mapping in the old prepare_docstrings.py:
    - awkward.operations.ak_flatten.flatten -> ak.flatten
    - awkward.operations.str.akstr_is_alnum.is_alnum -> ak.str.is_alnum
    - awkward.contents.recordarray.RecordArray -> ak.contents.RecordArray
    - awkward.highlevel.Array -> ak.Array
    """
    if not name.startswith("awkward"):
        return name
    name = re.sub(r"^awkward\.operations\.str\.akstr_\w+\.", "ak.str.", name)
    name = re.sub(r"^awkward\.operations\.str\.", "ak.str.", name)
    name = re.sub(r"^awkward\.operations\.ak_\w+\.", "ak.", name)
    name = re.sub(r"^awkward\.operations\.", "ak.", name)
    name = re.sub(r"^awkward\.highlevel\.", "ak.", name)
    name = re.sub(r"^awkward\.(contents|types|forms|index|record)\.\w+\.", r"ak.\1.", name)
    name = re.sub(r"^awkward\.", "ak.", name)
    return name


def _flat_output_dir(self, root):
    """Put all objects in the root directory (flat structure)."""
    return pathlib.PurePosixPath(root)


def _ak_output_filename(self):
    """Use public API names: awkward.flatten -> ak.flatten."""
    name = _awkward_to_ak(self.id)
    if name == "index":
        name = ".index"
    # Avoid case collisions on case-insensitive filesystems:
    # module "ak.record" would collide with class "ak.Record".
    # Prefix module-only pages with "_module." when they'd collide.
    if hasattr(self, "submodules"):  # TopLevelPythonObject (module/package)
        parts = name.rsplit(".", 1)
        if len(parts) == 2 and parts[-1].islower():
            # Check if a class with the same name (different case) might exist
            # This is conservative: prefix all lowercase leaf modules under
            # contents, types, forms, record, highlevel
            parent = parts[0]
            if parent in ("ak", "ak.contents", "ak.types", "ak.forms"):
                name = parts[0] + "._module." + parts[1]
    return name


from autoapi._objects import PythonObject, TopLevelPythonObject  # noqa: E402

PythonObject.output_dir = _flat_output_dir
PythonObject.output_filename = _ak_output_filename
TopLevelPythonObject.output_dir = _flat_output_dir
TopLevelPythonObject.output_filename = _ak_output_filename

autoapi_template_dir = "_autoapi_templates"

_latest_commit = (
    subprocess.run(
        ["git", "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        cwd=os.path.dirname(__file__) or ".",
    )
    .stdout.decode("utf-8")
    .strip()
)


def _github_source_link(obj):
    """Generate a 'Defined in module on line N' RST string with GitHub links."""
    all_objects = getattr(obj.app.env, "autoapi_all_objects", {})
    # For imported objects, resolve to the original definition
    source_obj = obj
    original_path = obj.obj.get("original_path")
    if original_path and original_path in all_objects:
        source_obj = all_objects[original_path]
    # Get file_path from the source object or its parent module
    file_path = source_obj.obj.get("file_path")
    if not file_path:
        module_name = source_obj.id[: -(len("." + source_obj.qual_name))] if source_obj.qual_name else source_obj.id
        module_obj = all_objects.get(module_name)
        if module_obj:
            file_path = module_obj.obj.get("file_path")
    if not file_path:
        return ""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rel_path = os.path.relpath(file_path, repo_root)
    module_name = rel_path.replace("/", ".").replace(".py", "").replace("src.", "")
    line_no = source_obj.obj.get("from_line_no")
    base_url = f"https://github.com/scikit-hep/awkward/blob/{_latest_commit}/{rel_path}"
    if line_no:
        return f"Defined in `{module_name} <{base_url}>`__ on `line {line_no} <{base_url}#L{line_no}>`__."
    return f"Defined in `{module_name} <{base_url}>`__."


def _process_docstring_filter(docstring, obj):
    """Process docstring markup: backticks, cross-refs, member links."""
    member_names = set()
    if hasattr(obj, "children"):
        for child in obj.children:
            member_names.add(child.name)
    qualname = _awkward_to_ak(obj.id)

    lines = docstring.split("\n")
    for i, line in enumerate(lines):
        line = line.replace("`", "``")
        line = re.sub(
            r"#(ak\.[A-Za-z0-9_\.]*[A-Za-z0-9_])",
            r":py:obj:`\1`",
            line,
        )
        if member_names:
            def _replace_member(m, _members=member_names, _qn=qualname):
                member = m.group(1)
                if member in _members:
                    return f":py:meth:`{member} <{_qn}.{member}>`"
                return m.group(0)
            line = re.sub(r"#([A-Za-z_][A-Za-z0-9_]*)", _replace_member, line)
        line = re.sub(
            r"\[([^\]]*)\]\(([^\)]*)\)",
            r"`\1 <\2>`__",
            line,
        )
        lines[i] = line
    return "\n".join(lines)


def _is_internal_module(obj):
    """Check if a module is an internal submodule that should be suppressed."""
    # Modules renamed with _module. prefix are internal submodules kept only
    # to avoid case collisions; they should not generate visible pages.
    return "._module." in obj.output_filename()


def autoapi_prepare_jinja_env(jinja_env):
    """Register custom filters and globals for autoapi templates."""
    jinja_env.filters["ak_name"] = _awkward_to_ak
    jinja_env.filters["process_docstring"] = _process_docstring_filter
    jinja_env.globals["github_source_link"] = _github_source_link
    jinja_env.globals["is_internal_module"] = _is_internal_module


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
exclude_patterns = ["_build", "_templates", "Thumbs.db", "jupyter_execute", ".*"]

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
        {
            "name": "Gitter",
            "url": "https://gitter.im/Scikit-HEP/awkward-array",
            "icon": "fa-brands fa-gitter",
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


# JupyterLite configuration
jupyterlite_dir = "./lite"
# Don't override ipynb format
jupyterlite_bind_ipynb_suffix = False
# We've disabled localstorage, so we must provide the contents explicitly
jupyterlite_contents = ["getting-started/demo/*"]

linkcheck_ignore = [
    r"^https?:\/\/github\.com\/.*$",
    r"^getting-started\/try-awkward-array\.html$",  # Relative link won't resolve
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


# Sphinx doesn't usually want content to fit the screen, so we hack the styles for this page
def install_jupyterlite_styles(app, pagename, templatename, context, event_arg) -> None:
    if pagename != "getting-started/try-awkward-array":
        return

    app.add_css_file("css/try-awkward-array.css")


def _skip_member(app, what, name, obj, skip, options):
    """Skip private modules/members but keep the top-level awkward package."""
    # Keep the top-level awkward package
    if name == "awkward":
        return False
    # Skip private modules (awkward._*) but not private methods/attributes
    # within public classes (those are controlled by the "private-members" option)
    if what in ("module", "package"):
        parts = name.split(".")
        if any(part.startswith("_") for part in parts):
            return True
    # Skip internal submodules that would collide with class pages on
    # case-insensitive filesystems (e.g., ak.contents.recordarray vs
    # ak.contents.RecordArray). The class is re-exported at the package
    # level, so the submodule page is redundant.
    if what == "module":
        parent_child = name.rsplit(".", 1)
        if len(parent_child) == 2:
            _parent, child = parent_child
            if re.match(r"awkward\.(contents|types|forms)\.\w+$", name) and child.islower():
                return True
    return skip


def _add_awkward_inventory_aliases(app, exception):
    """Duplicate ak.* inventory entries as awkward.* so intersphinx works.

    Downstream projects that annotate ``awkward.Array`` (resolved from
    ``import awkward as ak``) need ``awkward.*`` entries in objects.inv.
    See https://github.com/scikit-hep/awkward/issues/3950.
    """
    if exception:
        return
    import zlib

    inv_path = os.path.join(app.outdir, "objects.inv")
    with open(inv_path, "rb") as f:
        # Read the 4-line header verbatim
        header = b"".join(f.readline() for _ in range(4))
        # Decompress the body
        body = zlib.decompress(f.read()).decode("utf-8")

    extra_lines = []
    for line in body.splitlines():
        name = line.split(" ", 1)[0]
        if name.startswith("ak."):
            extra_lines.append("awkward." + line[3:])

    if extra_lines:
        new_body = body.rstrip("\n") + "\n" + "\n".join(extra_lines) + "\n"
        with open(inv_path, "wb") as f:
            f.write(header)
            f.write(zlib.compress(new_body.encode("utf-8")))


def setup(app):
    app.connect("html-page-context", install_jupyterlite_styles)
    app.connect("autoapi-skip-member", _skip_member)
    app.connect("build-finished", _add_awkward_inventory_aliases)
