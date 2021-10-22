# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def to_layout(
    array,
    allow_record=True,
    allow_other=False,
    numpytype=(np.number, np.bool_, np.str_, np.bytes_, np.datetime64, np.timedelta64),
):
    pass


#     """
#     Args:
#         array: Data to convert into an #ak.layout.Content and maybe
#             #ak.layout.Record and other types.
#         allow_record (bool): If True, allow #ak.layout.Record as an output;
#             otherwise, if the output would be a scalar record, raise an error.
#         allow_other (bool): If True, allow non-Awkward outputs; otherwise,
#             if the output would be another type, raise an error.
#         numpytype (tuple of NumPy types): Allowed NumPy types in
#             #ak.layout.NumpyArray outputs.

#     Converts `array` (many types supported, including all Awkward Arrays and
#     Records) into a #ak.layout.Content and maybe #ak.layout.Record and other
#     types.

#     This function is usually used to sanitize inputs for other functions; it
#     would rarely be used in a data analysis.
#     """
#     if isinstance(array, ak.highlevel.Array):
#         return array.layout

#     elif allow_record and isinstance(array, ak.highlevel.Record):
#         return array.layout

#     elif isinstance(array, ak.highlevel.ArrayBuilder):
#         return array.snapshot().layout

#     elif isinstance(array, ak.layout.ArrayBuilder):
#         return array.snapshot()

#     elif isinstance(array, (ak._v2.contents.Content, ak.partition.PartitionedArray)):
#         return array

#     elif allow_record and isinstance(array, ak._v2.record.Record):
#         return array

#     elif isinstance(array, (np.ndarray, numpy.ma.MaskedArray)):
#         if not issubclass(array.dtype.type, numpytype):
#             raise ValueError(
#                 "NumPy {0} not allowed".format(repr(array.dtype))
#                 + ak._v2._util.exception_suffix(__file__)
#             )
#         return from_numpy(array, regulararray=True, recordarray=True, highlevel=False)

#     elif (
#         type(array).__module__.startswith("cupy.") and type(array).__name__ == "ndarray"
#     ):
#         return from_cupy(array, regulararray=True, highlevel=False)

#     elif isinstance(array, (str, bytes)) or (
#         ak._v2._util.py27 and isinstance(array, ak._v2._util.unicode)
#     ):
#         return from_iter([array], highlevel=False)

#     elif isinstance(array, Iterable):
#         return from_iter(array, highlevel=False)

#     elif not allow_other:
#         raise TypeError(
#             "{0} cannot be converted into an Awkward Array".format(array)
#             + ak._v2._util.exception_suffix(__file__)
#         )

#     else:
#         return array
