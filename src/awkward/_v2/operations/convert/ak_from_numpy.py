# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()
numpy = ak.nplike.Numpy.instance()


def from_numpy(
    array, regulararray=False, recordarray=True, highlevel=True, behavior=None
):
    """
    Args:
        array (np.ndarray): The NumPy array to convert into an Awkward Array.
            This array can be a np.ma.MaskedArray.
        regulararray (bool): If True and the array is multidimensional,
            the dimensions are represented by nested #ak.layout.RegularArray
            nodes; if False and the array is multidimensional, the dimensions
            are represented by a multivalued #ak.layout.NumpyArray.shape.
            If the array is one-dimensional, this has no effect.
        recordarray (bool): If True and the array is a NumPy structured array
            (dtype.names is not None), the fields are represented by an
            #ak.layout.RecordArray; if False and the array is a structured
            array, the structure is left in the #ak.layout.NumpyArray `format`,
            which some functions do not recognize.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Converts a NumPy array into an Awkward Array.

    The resulting layout can only involve the following #ak.layout.Content types:

       * #ak.layout.NumpyArray
       * #ak.layout.ByteMaskedArray or #ak.layout.UnmaskedArray if the
         `array` is an np.ma.MaskedArray.
       * #ak.layout.RegularArray if `regulararray=True`.
       * #ak.layout.RecordArray if `recordarray=True`.

    See also #ak.to_numpy and #ak.from_cupy.
    """

    def recurse(array, mask):
        if regulararray and len(array.shape) > 1:
            return ak._v2.contents.RegularArray(
                recurse(array.reshape((-1,) + array.shape[2:]), mask),
                array.shape[1],
                array.shape[0],
            )

        if len(array.shape) == 0:
            array = array.reshape(1)

        if array.dtype.kind == "S":
            asbytes = array.reshape(-1)
            itemsize = asbytes.dtype.itemsize
            starts = numpy.arange(0, len(asbytes) * itemsize, itemsize, dtype=np.int64)
            stops = starts + numpy.char.str_len(asbytes)
            data = ak._v2.contents.ListArray64(
                ak._v2.index.Index64(starts),
                ak._v2.index.Index64(stops),
                ak._v2.contents.NumpyArray(
                    asbytes.view("u1"), parameters={"__array__": "byte"}, nplike=numpy
                ),
                parameters={"__array__": "bytestring"},
            )
            for i in range(len(array.shape) - 1, 0, -1):
                data = ak._v2.contents.RegularArray(
                    data, array.shape[i], array.shape[i - 1]
                )

        elif array.dtype.kind == "U":
            asbytes = numpy.char.encode(array.reshape(-1), "utf-8", "surrogateescape")
            itemsize = asbytes.dtype.itemsize
            starts = numpy.arange(0, len(asbytes) * itemsize, itemsize, dtype=np.int64)
            stops = starts + numpy.char.str_len(asbytes)
            data = ak._v2.contents.ListArray64(
                ak._v2.index.Index64(starts),
                ak._v2.index.Index64(stops),
                ak._v2.contents.NumpyArray(
                    asbytes.view("u1"), parameters={"__array__": "char"}, nplike=numpy
                ),
                parameters={"__array__": "string"},
            )
            for i in range(len(array.shape) - 1, 0, -1):
                data = ak._v2.contents.RegularArray(
                    data, array.shape[i], array.shape[i - 1]
                )

        else:
            data = ak._v2.contents.NumpyArray(array)

        if mask is None:
            return data

        elif mask is False or (isinstance(mask, np.bool_) and not mask):
            # NumPy's MaskedArray with mask == False is an UnmaskedArray
            if len(array.shape) == 1:
                return ak._v2.contents.UnmaskedArray(data)
            else:

                def attach(x):
                    if isinstance(x, ak._v2.contents.NumpyArray):
                        return ak._v2.contents.UnmaskedArray(x)
                    else:
                        return ak._v2.contents.RegularArray(
                            attach(x.content), x.size, len(x)
                        )

                return attach(data.toRegularArray())

        else:
            # NumPy's MaskedArray is a ByteMaskedArray with valid_when=False
            return ak._v2.contents.ByteMaskedArray(
                ak._v2.index.Index8(mask), data, valid_when=False
            )

    if isinstance(array, numpy.ma.MaskedArray):
        mask = numpy.ma.getmask(array)
        array = numpy.ma.getdata(array)
        if isinstance(mask, np.ndarray) and len(mask.shape) > 1:
            regulararray = True
            mask = mask.reshape(-1)
    else:
        mask = None

    if not recordarray or array.dtype.names is None:
        layout = recurse(array, mask)

    else:
        contents = []
        for name in array.dtype.names:
            contents.append(recurse(array[name], mask))
        layout = ak._v2.contents.RecordArray(contents, array.dtype.names)

    return ak._v2._util.wrap(layout, behavior, highlevel)
