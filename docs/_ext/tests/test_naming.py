# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import pytest
from awkward_docs.naming import is_internal_module, output_filename, public_name
from hypothesis import given
from hypothesis import strategies as st
from strategies import awkward_names, dotted_names


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("awkward.operations.ak_flatten.flatten", "ak.flatten"),
        ("awkward.operations.ak_flatten", "ak.ak_flatten"),
        ("awkward.operations.str.akstr_is_alnum.is_alnum", "ak.str.is_alnum"),
        ("awkward.operations.str.split_pattern", "ak.str.split_pattern"),
        ("awkward.operations.str", "ak.str"),
        ("awkward.highlevel.Array", "ak.Array"),
        ("awkward.highlevel.Array.to_list", "ak.Array.to_list"),
        ("awkward.contents.recordarray.RecordArray", "ak.contents.RecordArray"),
        ("awkward.types.listtype.ListType", "ak.types.ListType"),
        ("awkward.forms.numpyform.NumpyForm", "ak.forms.NumpyForm"),
        ("awkward.index.Index64", "ak.index.Index64"),
        ("awkward.record.Record", "ak.record.Record"),
        # Members of classes in these modules lose the class name; no page or
        # link uses these names, because members do not get their own pages
        ("awkward.record.Record.to_list", "ak.record.to_list"),
        ("awkward.index.Index.zeros", "ak.index.zeros"),
        ("awkward.prettyprint.FormatterOptions", "ak.prettyprint.FormatterOptions"),
        ("awkward.contents", "ak.contents"),
        ("awkward", "awkward"),
        ("awkwardly.thing", "awkwardly.thing"),
        ("numpy.ndarray", "numpy.ndarray"),
        ("", ""),
    ],
)
def test_public_name(name, expected):
    assert public_name(name) == expected


@given(dotted_names.filter(lambda name: not name.startswith("awkward")))
def test_public_name_keeps_other_names(name):
    assert public_name(name) == name


@given(awkward_names)
def test_public_name_of_awkward_names_starts_with_ak(name):
    assert public_name(name).startswith("ak.")


@given(st.one_of(dotted_names, awkward_names))
def test_public_name_is_idempotent(name):
    assert public_name(public_name(name)) == public_name(name)


@pytest.mark.parametrize(
    ("object_id", "is_module", "expected"),
    [
        ("awkward.operations.ak_flatten.flatten", False, "ak.flatten"),
        ("index", False, ".index"),
        ("index", True, ".index"),
        ("awkward", True, "awkward"),
        ("awkward.record", True, "ak._module.record"),
        ("awkward.record.Record", False, "ak.record.Record"),
        ("awkward.highlevel", True, "ak._module.highlevel"),
        ("awkward.operations.str", True, "ak._module.str"),
        ("awkward.contents.recordarray", True, "ak.contents._module.recordarray"),
        ("awkward.types.listtype", True, "ak.types._module.listtype"),
        ("awkward.forms.form", True, "ak.forms._module.form"),
        ("awkward.contents.recordarray", False, "ak.contents.recordarray"),
        ("awkward.contents.Content", True, "ak.contents.Content"),
        ("awkward.index.foo", True, "ak.index.foo"),
        ("awkward.prettyprint", True, "ak._module.prettyprint"),
        ("awkward._connect.numba", True, "ak._connect.numba"),
    ],
)
def test_output_filename(object_id, is_module, expected):
    assert output_filename(object_id, is_module=is_module) == expected


@given(st.one_of(dotted_names, awkward_names))
def test_output_filename_of_non_modules_is_the_public_name(object_id):
    expected = ".index" if object_id == "index" else public_name(object_id)
    assert output_filename(object_id, is_module=False) == expected


@given(st.one_of(dotted_names, awkward_names), st.booleans())
def test_output_filename_is_internal_only_for_colliding_modules(object_id, is_module):
    parent, _, child = public_name(object_id).rpartition(".")
    collides = (
        is_module
        and child.islower()
        and parent in ("ak", "ak.contents", "ak.types", "ak.forms")
    )
    assert (
        is_internal_module(output_filename(object_id, is_module=is_module)) == collides
    )


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("ak._module.record", True),
        ("ak.contents._module.recordarray", True),
        ("ak.record", False),
        ("ak.module.record", False),
        ("_module.record", False),
    ],
)
def test_is_internal_module(filename, expected):
    assert is_internal_module(filename) == expected
