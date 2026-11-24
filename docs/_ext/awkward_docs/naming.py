# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
"""Public ``ak.*`` names and the file names of the API reference pages."""

import re

# Parents whose lowercase submodules would have the same file name as a class
# on a case-insensitive file system, e.g. module ak.record and class ak.Record
_CASE_COLLISION_PARENTS = ("ak", "ak.contents", "ak.types", "ak.forms")


def public_name(name: str) -> str:
    """Return the public ``ak.*`` name of an object in the ``awkward`` package.

    For example, ``awkward.operations.ak_flatten.flatten`` is ``ak.flatten``,
    and ``awkward.contents.recordarray.RecordArray`` is
    ``ak.contents.RecordArray``. Names outside the package are returned as
    they are.
    """
    if not name.startswith("awkward"):
        return name
    name = re.sub(r"^awkward\.operations\.str\.akstr_\w+\.", "ak.str.", name)
    name = re.sub(r"^awkward\.operations\.str\.", "ak.str.", name)
    name = re.sub(r"^awkward\.operations\.ak_\w+\.", "ak.", name)
    name = re.sub(r"^awkward\.operations\.", "ak.", name)
    name = re.sub(r"^awkward\.highlevel\.", "ak.", name)
    name = re.sub(
        r"^awkward\.(contents|types|forms|index|record)\.\w+\.", r"ak.\1.", name
    )
    return re.sub(r"^awkward\.", "ak.", name)


def output_filename(object_id: str, *, is_module: bool) -> str:
    """Return the file name, without suffix, of the page for an object.

    A lowercase module directly under ``ak``, ``ak.contents``, ``ak.types``,
    or ``ak.forms`` gets ``_module.`` before its last component, so that its
    page does not collide with a class page on a case-insensitive file system.
    """
    name = public_name(object_id)
    if name == "index":
        name = ".index"
    if is_module:
        parent, _, child = name.rpartition(".")
        if parent in _CASE_COLLISION_PARENTS and child.islower():
            name = f"{parent}._module.{child}"
    return name


def is_internal_module(filename: str) -> bool:
    """Return whether a page file name is one renamed by ``output_filename``."""
    return "._module." in filename
