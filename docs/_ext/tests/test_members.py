# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import pytest
from awkward_docs.members import skip_member
from hypothesis import given
from hypothesis import strategies as st
from strategies import awkward_names, dotted_names

OBJECT_TYPES = [
    "module",
    "package",
    "function",
    "class",
    "method",
    "attribute",
    "property",
    "data",
    "exception",
]


@pytest.mark.parametrize("what", OBJECT_TYPES)
@pytest.mark.parametrize("skip", [False, True])
def test_skip_member_documents_the_awkward_package(what, skip):
    assert skip_member(what, "awkward", skip) is False


@pytest.mark.parametrize(
    ("what", "name", "expected"),
    [
        # Module-level dunder functions
        ("function", "awkward.__dir__", True),
        ("function", "awkward.__getattr__", True),
        ("function", "awkward.operations.ak_flatten.__all__", True),
        ("function", "awkward.flatten", False),
        ("function", "awkward._private", False),
        ("function", "awkward.__private", False),
        ("function", "awkward.private__", False),
        ("method", "awkward.highlevel.Array.__getitem__", False),
        # Private modules and packages
        ("module", "awkward._do", True),
        ("package", "awkward._connect", True),
        ("module", "awkward._connect.numba.arrayview", True),
        ("module", "awkward.contents._private", True),
        ("module", "awkward.operations", False),
        ("class", "awkward._do.Something", False),
        ("method", "awkward.highlevel.Array._layout", False),
        # Lowercase submodules of contents, types, and forms
        ("module", "awkward.contents.recordarray", True),
        ("module", "awkward.types.listtype", True),
        ("module", "awkward.forms.numpyform", True),
        ("module", "awkward.contents.RecordArray", False),
        ("module", "awkward.contents.record_array2", True),
        ("module", "awkward.index.index", False),
        ("module", "awkward.contents", False),
        ("module", "awkward.contents.content.sub", False),
        ("package", "awkward.contents.recordarray", False),
        ("class", "awkward.contents.recordarray", False),
    ],
)
def test_skip_member(what, name, expected):
    assert skip_member(what, name, False) == expected


@given(
    st.sampled_from(["class", "method", "attribute", "property", "data", "exception"]),
    st.one_of(dotted_names, awkward_names),
    st.booleans(),
)
def test_skip_member_keeps_autoapis_decision_for_other_types(what, name, skip):
    if name != "awkward":
        assert skip_member(what, name, skip) == skip


@given(st.sampled_from(["module", "package"]), dotted_names, st.booleans())
def test_skip_member_skips_names_with_a_private_part(what, name, skip):
    private = f"{name}._x"
    assert skip_member(what, private, skip) is True


@given(st.sampled_from(OBJECT_TYPES), st.one_of(dotted_names, awkward_names))
def test_skip_member_skips_what_autoapi_skips(what, name):
    if name != "awkward":
        assert skip_member(what, name, True) is True
