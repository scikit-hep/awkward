# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

def test_typed_builder_form():
    builder = ak.layout.TypedArrayBuilder()
    array = ak.layout.NumpyArray(np.empty(122))
    builder.apply(array.form)

    assert builder.form() == array.form

    # builder.beginrecord()
    # builder.field("one")
    # builder.integer(1)
    # builder.field("two")
    # builder.real(1.1)
    # builder.endrecord()
    #
    # builder.beginrecord()
    # builder.field("two")
    # builder.real(2.2)
    # builder.field("one")
    # builder.integer(2)
    # builder.endrecord()
    #
    # builder.beginrecord()
    # builder.field("one")
    # builder.integer(3)
    # builder.field("two")
    # builder.real(3.3)
    # builder.endrecord()
    #
    # assert str(builder.type(typestrs)) == '{"one": int64, "two": float64}'
    # assert ak.to_list(builder.snapshot()) == [
    #     {"one": 1, "two": 1.1},
    #     {"one": 2, "two": 2.2},
    #     {"one": 3, "two": 3.3},
    # ]
