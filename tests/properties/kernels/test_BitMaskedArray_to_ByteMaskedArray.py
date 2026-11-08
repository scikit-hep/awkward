# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Property-based test for the ``awkward_BitMaskedArray_to_ByteMaskedArray`` kernel.

Prototype for replacing the frozen samples in ``kernel-test-data.json`` with
Hypothesis-generated inputs. The property is::

    backend_kernel(inputs) == reference_definition(inputs)   for all valid inputs
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests.properties.kernels import harness, strategies

KERNEL = "awkward_BitMaskedArray_to_ByteMaskedArray"

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

    # Reference implementation: pure-Python, mutating a plain list in place.
    expected: list = [0] * (bitmasklength * 8)
    reference(expected, frombitmask, bitmasklength, validwhen, lsb_order)

    tobytemask, *_ = run(
        KERNEL,
        np.zeros(bitmasklength * 8, dtype=np.int8),
        np.array(frombitmask, dtype=np.uint8),
        bitmasklength,
        validwhen,
        lsb_order,
    )

    # Reference yields numpy.bool_, the kernels write int8 0/1 — normalise both.
    got = [int(bool(x)) for x in tobytemask]
    assert got == [int(bool(x)) for x in expected]
