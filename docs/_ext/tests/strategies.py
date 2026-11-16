# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
"""Hypothesis strategies shared by the tests."""

from hypothesis import strategies as st

identifiers = st.from_regex(r"[A-Za-z_][A-Za-z0-9_]{0,8}", fullmatch=True)

# Dotted names such as awkward.operations.ak_flatten.flatten
dotted_names = st.lists(identifiers, min_size=1, max_size=5).map(".".join)

awkward_names = dotted_names.map(lambda name: f"awkward.{name}")
