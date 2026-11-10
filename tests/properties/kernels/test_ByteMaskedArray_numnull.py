# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Property-based test for the ``awkward_ByteMaskedArray_numnull`` kernel."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests.properties.kernels import harness, strategies

KERNEL = "awkward_ByteMaskedArray_numnull"

# The kernel writes a count >= 0, never this value: a counter left unwritten
# by a backend keeps it and fails the comparison.
SENTINEL = int(np.iinfo(np.int64).min)

reference = harness.load_reference(KERNEL)


@pytest.mark.parametrize("run", harness.backends())
@given(mask=strategies.st_bytemask(), validwhen=st.booleans())
def test_matches_reference(run, mask: list[int], validwhen: bool) -> None:
    """Each backend kernel must agree with the spec's reference implementation."""
    length = len(mask)

    expected = [SENTINEL]
    reference(expected, mask, length, validwhen)

    numnull, *_ = run(
        KERNEL,
        np.full(1, SENTINEL, dtype=np.int64),
        np.array(mask, dtype=np.int8),
        length,
        validwhen,
    )

    assert numnull.tolist() == expected
