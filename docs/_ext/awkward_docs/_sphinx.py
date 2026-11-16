# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
"""Connections to Sphinx and sphinx-autoapi."""

import os
import pathlib
import subprocess

from autoapi._objects import PythonObject, TopLevelPythonObject
from autoapi._parser import Parser
from docutils import nodes
from sphinx.domains.python import PythonDomain

from awkward_docs import (
    docstrings,
    experimental,
    inventory,
    members,
    naming,
    source_links,
)

# -- Monkey-patch autoapi to produce ak.* file names -------------------------
# autoapi generates paths from Python module structure (awkward.operations.ak_flatten.flatten).
# We remap to public API names (ak.flatten) to match existing URLs.


def _flat_output_dir(self, root):
    """Put all objects in the root directory (flat structure)."""
    return pathlib.PurePosixPath(root)


def _ak_output_filename(self):
    """Use public API names: awkward.flatten -> ak.flatten."""
    # TopLevelPythonObject (module/package) has submodules
    return naming.output_filename(self.id, is_module=hasattr(self, "submodules"))


_parse_functiondef = Parser.parse_functiondef


def _parse_functiondef_with_experimental(self, node):
    """Preserve the experimental marker for the AutoAPI templates."""
    parsed = _parse_functiondef(self, node)
    if node.decorators and any(
        experimental.is_experimental(decorator.as_string())
        for decorator in node.decorators.nodes
    ):
        for item in parsed:
            item["is_experimental"] = True
    return parsed


# -- Keep class references from resolving to attributes ----------------------
# Since Sphinx 9, a class reference that matches no class is retried as data or
# an attribute, with suffix matching for type fields. ``bool`` in a parameter
# type would then link to ``ak.prettyprint.FormatterOptions.bool`` instead of
# the Python documentation through intersphinx. Skip that retry.

_resolve_xref = PythonDomain.resolve_xref
_find_obj = PythonDomain.find_obj


def _resolve_xref_without_data_fallback(
    self, env, fromdocname, builder, type, target, node, contnode
):
    self._awkward_resolving_class = type == "class"
    try:
        return _resolve_xref(
            self, env, fromdocname, builder, type, target, node, contnode
        )
    finally:
        self._awkward_resolving_class = False


def _find_obj_without_data_fallback(
    self, env, modname, classname, name, type, searchmode=0
):
    if getattr(self, "_awkward_resolving_class", False) and type in ("data", "attr"):
        return []
    return _find_obj(self, env, modname, classname, name, type, searchmode)


# -- Filters and globals for the autoapi templates ---------------------------

# Set in setup()
_latest_commit = None


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
        module_name = (
            source_obj.id[: -(len("." + source_obj.qual_name))]
            if source_obj.qual_name
            else source_obj.id
        )
        module_obj = all_objects.get(module_name)
        if module_obj:
            file_path = module_obj.obj.get("file_path")
    if not file_path:
        return ""
    repo_root = os.path.dirname(os.path.abspath(obj.app.confdir))
    rel_path = os.path.relpath(file_path, repo_root)
    line_no = source_obj.obj.get("from_line_no")
    return source_links.source_link(rel_path, line_no, _latest_commit)


def _process_docstring_filter(docstring, obj):
    """Process docstring markup: backticks, cross-refs, member links."""
    member_names = {child.name for child in getattr(obj, "children", ())}
    return docstrings.process_docstring(
        docstring, member_names, naming.public_name(obj.id)
    )


def _is_internal_module(obj):
    """Check if a module is an internal submodule that should be suppressed."""
    # Modules renamed with _module. prefix are internal submodules kept only
    # to avoid case collisions; they should not generate visible pages.
    return naming.is_internal_module(obj.output_filename())


def prepare_jinja_env(jinja_env):
    """Register custom filters and globals for autoapi templates."""
    jinja_env.filters["ak_name"] = naming.public_name
    jinja_env.filters["process_docstring"] = _process_docstring_filter
    jinja_env.globals["github_source_link"] = _github_source_link
    jinja_env.globals["is_internal_module"] = _is_internal_module


# -- Event handlers ----------------------------------------------------------


def _skip_member(app, what, name, obj, skip, options):
    """Skip private modules/members but keep the top-level awkward package."""
    return members.skip_member(what, name, skip)


def _add_awkward_inventory_aliases(app, exception):
    """Duplicate ak.* inventory entries as awkward.* so intersphinx works.

    Downstream projects that annotate ``awkward.Array`` (resolved from
    ``import awkward as ak``) need ``awkward.*`` entries in objects.inv.
    See https://github.com/scikit-hep/awkward/issues/3950.
    """
    # The sphinx-llm Markdown sub-build has no objects.inv
    if exception or app.builder.format != "html":
        return

    inv_path = os.path.join(app.outdir, "objects.inv")
    with open(inv_path, "rb") as f:
        data = f.read()
    new_data = inventory.add_awkward_aliases(data)
    if new_data != data:
        with open(inv_path, "wb") as f:
            f.write(new_data)


def _visit_abbreviation_markdown(self, node):
    """Write the text of an abbreviation; the title has no Markdown form."""


def _depart_abbreviation_markdown(self, node):
    pass


def setup(app):
    global _latest_commit
    _latest_commit = (
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            stdout=subprocess.PIPE,
            cwd=app.confdir,
        )
        .stdout.decode("utf-8")
        .strip()
    )

    PythonObject.output_dir = _flat_output_dir
    PythonObject.output_filename = _ak_output_filename
    TopLevelPythonObject.output_dir = _flat_output_dir
    TopLevelPythonObject.output_filename = _ak_output_filename
    Parser.parse_functiondef = _parse_functiondef_with_experimental
    PythonDomain.resolve_xref = _resolve_xref_without_data_fallback
    PythonDomain.find_obj = _find_obj_without_data_fallback

    app.connect("autoapi-skip-member", _skip_member)
    app.connect("build-finished", _add_awkward_inventory_aliases)
    # Sphinx 9 wraps the ``*`` and ``/`` separators in signatures in
    # abbreviation nodes, which sphinx-markdown-builder would otherwise drop
    app.add_node(
        nodes.abbreviation,
        override=True,
        markdown=(_visit_abbreviation_markdown, _depart_abbreviation_markdown),
    )
    return {"parallel_read_safe": True, "parallel_write_safe": True}
