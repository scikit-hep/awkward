# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward._requirements import (
    build_requirement_context_factory,
    import_required_module,
)

_requirement_context_factory = build_requirement_context_factory("cppyy>=3.1.0")


def register_and_check():
    with _requirement_context_factory():
        return import_required_module("cppyy")
