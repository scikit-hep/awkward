# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Hypothesis strategies shared by two or more kernel property tests."""

from hypothesis import strategies as st


def st_bitmask() -> st.SearchStrategy[list[int]]:
    """Bytes of a BitMaskedArray mask, eight mask bits each."""
    return st.lists(st.integers(min_value=0, max_value=255), max_size=64)


def st_bytemask() -> st.SearchStrategy[list[int]]:
    """Bytes of a ByteMaskedArray mask, any nonzero value set.

    Half the bytes are 0 or 1, the values awkward writes; uniform int8 values
    alone would almost never be 0.
    """
    byte = st.one_of(
        st.integers(min_value=0, max_value=1),
        st.integers(min_value=-128, max_value=127),
    )
    return st.lists(byte, max_size=64)
