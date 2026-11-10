# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Property-based test for the ``awkward_IndexedOptionArray_rpad_and_clip_mask_axis1`` kernel."""

import numpy as np
import pytest
from hypothesis import given

from tests.properties.kernels import harness, strategies

KERNEL = "awkward_IndexedOptionArray_rpad_and_clip_mask_axis1"

# The kernel writes -1 or an index >= 0, never this value: an element left
# unwritten by a backend keeps it and fails the comparison.
SENTINEL = int(np.iinfo(np.int64).min)

reference = harness.load_reference(KERNEL)


@pytest.mark.parametrize("run", harness.backends())
@given(frommask=strategies.st_bytemask())
def test_matches_reference(run, frommask: list[int]) -> None:
    """Each backend kernel must agree with the spec's reference implementation."""
    length = len(frommask)

    expected = [SENTINEL] * length
    reference(expected, frommask, length)

    toindex, *_ = run(
        KERNEL,
        np.full(length, SENTINEL, dtype=np.int64),
        np.array(frommask, dtype=np.int8),
        length,
    )

    assert toindex.tolist() == expected
