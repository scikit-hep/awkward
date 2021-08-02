# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_array_builder():
    builder = ak.ArrayBuilder()

    builder.real(1.1)
    builder.real(2.2)
    builder.real(3.3)

    assert ak.to_list(builder.snapshot()) == [1.1, 2.2, 3.3]


def test_array_bool_builder():
    builder = ak.ArrayBuilder()

    builder.boolean(True)
    builder.boolean(False)
    builder.boolean(True)

    assert ak.to_list(builder.snapshot()) == [True, False, True]


def test_array_builder_append():
    array = ak.Array(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], check_valid=True
    )

    builder = ak.ArrayBuilder()
    builder.append(array, 3)
    builder.append(array, 2)
    builder.append(array, 2)
    builder.append(array, 0)
    builder.append(array, 1)
    builder.append(array, -1)
    assert ak.to_list(builder.snapshot()) == [
        [5.5],
        [3.3, 4.4],
        [3.3, 4.4],
        [0.0, 1.1, 2.2],
        [],
        [6.6, 7.7, 8.8, 9.9],
    ]

    builder.extend(array)
    assert ak.to_list(builder.snapshot()) == [
        [5.5],
        [3.3, 4.4],
        [3.3, 4.4],
        [0.0, 1.1, 2.2],
        [],
        [6.6, 7.7, 8.8, 9.9],
        [0.0, 1.1, 2.2],
        [],
        [3.3, 4.4],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]
