# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Property-based test for the ``awkward_BitMaskedArray_to_IndexedOptionArray`` kernel."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests.properties.kernels import harness, strategies

KERNEL = "awkward_BitMaskedArray_to_IndexedOptionArray"

# The kernel writes -1 or an index >= 0, never this value: an element left
# unwritten by a backend keeps it and fails the comparison.
SENTINEL = int(np.iinfo(np.int64).min)

reference = harness.load_reference(KERNEL)


@pytest.mark.parametrize("run", harness.backends())
@given(
    frombitmask=strategies.st_bitmask(),
    validwhen=st.booleans(),
    lsb_order=st.booleans(),
)
def test_matches_reference(
    run, frombitmask: list[int], validwhen: bool, lsb_order: bool
) -> None:
    """Each backend kernel must agree with the spec's reference implementation."""
    bitmasklength = len(frombitmask)

    expected = [SENTINEL] * (bitmasklength * 8)
    reference(expected, frombitmask, bitmasklength, validwhen, lsb_order)

    toindex, *_ = run(
        KERNEL,
        np.full(bitmasklength * 8, SENTINEL, dtype=np.int64),
        np.array(frombitmask, dtype=np.uint8),
        bitmasklength,
        validwhen,
        lsb_order,
    )

    assert toindex.tolist() == expected
