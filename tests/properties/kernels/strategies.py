# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Hypothesis strategies shared by two or more kernel property tests."""

from hypothesis import strategies as st


def st_bitmask() -> st.SearchStrategy[list[int]]:
    """Bytes of a BitMaskedArray mask, eight mask bits each."""
    return st.lists(st.integers(min_value=0, max_value=255), max_size=64)
