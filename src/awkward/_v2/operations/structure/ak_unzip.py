# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def unzip(array):
    raise NotImplementedError


#     """
#     If the `array` contains tuples or records, this operation splits them
#     into a Python tuple of arrays, one for each field.

#     If the `array` does not contain tuples or records, the single `array`
#     is placed in a length 1 Python tuple.

#     For example,

#         >>> array = ak.Array([{"x": 1.1, "y": [1]},
#         ...                   {"x": 2.2, "y": [2, 2]},
#         ...                   {"x": 3.3, "y": [3, 3, 3]}])
#         >>> x, y = ak.unzip(array)
#         >>> x
#         <Array [1.1, 2.2, 3.3] type='3 * float64'>
#         >>> y
#         <Array [[1], [2, 2], [3, 3, 3]] type='3 * var * int64'>
#     """
#     fields = ak._v2.operations.describe.fields(array)
#     if len(fields) == 0:
#         return (array,)
#     else:
#         return tuple(array[n] for n in fields)
