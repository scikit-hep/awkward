# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import pytest
from awkward_docs.experimental import is_experimental
from hypothesis import given
from hypothesis import strategies as st


@pytest.mark.parametrize(
    ("decorator", "expected"),
    [
        ("experimental", True),
        ("experimental()", True),
        ("experimental(reason='x')", False),
        ("awkward._experimental.experimental", False),
        ("experimental_feature", False),
        ("not_experimental", False),
        ("functools.cache", False),
        ("property", False),
        ("", False),
    ],
)
def test_is_experimental(decorator, expected):
    assert is_experimental(decorator) == expected


@given(st.text().filter(lambda text: text not in ("experimental", "experimental()")))
def test_is_experimental_only_for_the_two_forms(decorator):
    assert not is_experimental(decorator)
