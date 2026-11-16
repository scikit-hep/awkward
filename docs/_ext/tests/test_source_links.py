# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import pytest
from awkward_docs.source_links import source_link
from hypothesis import given
from hypothesis import strategies as st
from strategies import identifiers

COMMIT = "0123456789abcdef0123456789abcdef01234567"
BLOB = f"https://github.com/scikit-hep/awkward/blob/{COMMIT}"


@pytest.mark.parametrize(
    ("rel_path", "line", "expected"),
    [
        (
            "src/awkward/operations/ak_flatten.py",
            23,
            (
                f"Defined in `awkward.operations.ak_flatten <{BLOB}/src/awkward/operations/ak_flatten.py>`__"
                f" on `line 23 <{BLOB}/src/awkward/operations/ak_flatten.py#L23>`__."
            ),
        ),
        (
            "src/awkward/highlevel.py",
            None,
            f"Defined in `awkward.highlevel <{BLOB}/src/awkward/highlevel.py>`__.",
        ),
        (
            "src/awkward/highlevel.py",
            0,
            f"Defined in `awkward.highlevel <{BLOB}/src/awkward/highlevel.py>`__.",
        ),
        (
            "src/awkward/__init__.py",
            1,
            (
                f"Defined in `awkward.__init__ <{BLOB}/src/awkward/__init__.py>`__"
                f" on `line 1 <{BLOB}/src/awkward/__init__.py#L1>`__."
            ),
        ),
    ],
)
def test_source_link(rel_path, line, expected):
    assert source_link(rel_path, line, COMMIT) == expected


module_paths = st.lists(identifiers, min_size=1, max_size=4).map(
    lambda parts: "src/awkward/" + "/".join(parts) + ".py"
)


@given(module_paths, st.integers(min_value=1, max_value=100_000))
def test_source_link_links_the_file_and_line(rel_path, line):
    link = source_link(rel_path, line, COMMIT)
    assert f"<{BLOB}/{rel_path}>`__" in link
    assert f"`line {line} <{BLOB}/{rel_path}#L{line}>`__." in link


@given(module_paths)
def test_source_link_without_line_links_the_file(rel_path):
    assert source_link(rel_path, None, COMMIT).endswith(f"<{BLOB}/{rel_path}>`__.")
