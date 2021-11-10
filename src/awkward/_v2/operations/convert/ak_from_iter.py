# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


def from_iter(
    iterable, highlevel=True, behavior=None, allow_record=True, initial=1024, resize=1.5
):
    """
    Args:
        iterable (Python iterable): Data to convert into an Awkward Array.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        allow_record (bool): If True, the outermost element may be a record
            (returning #ak.Record or #ak.layout.Record type, depending on
            `highlevel`); if False, the outermost element must be an array.
        initial (int): Initial size (in bytes) of buffers used by
            #ak.layout.ArrayBuilder (see #ak.layout.ArrayBuilderOptions).
        resize (float): Resize multiplier for buffers used by
            #ak.layout.ArrayBuilder (see #ak.layout.ArrayBuilderOptions);
            should be strictly greater than 1.

    Converts Python data into an Awkward Array.

    Internally, this function uses #ak.layout.ArrayBuilder (see the high-level
    #ak.ArrayBuilder documentation for a more complete description), so it
    has the same flexibility and the same constraints. Any heterogeneous
    and deeply nested Python data can be converted, but the output will never
    have regular-typed array lengths.

    The following Python types are supported.

       * bool, including `np.bool_`: converted into #ak.layout.NumpyArray.
       * int, including `np.integer`: converted into #ak.layout.NumpyArray.
       * float, including `np.floating`: converted into #ak.layout.NumpyArray.
       * bytes: converted into #ak.layout.ListOffsetArray with parameter
         `"__array__"` equal to `"bytestring"` (unencoded bytes).
       * str: converted into #ak.layout.ListOffsetArray with parameter
         `"__array__"` equal to `"string"` (UTF-8 encoded string).
       * tuple: converted into #ak.layout.RecordArray without field names
         (i.e. homogeneously typed, uniform sized tuples).
       * dict: converted into #ak.layout.RecordArray with field names
         (i.e. homogeneously typed records with the same sets of fields).
       * iterable, including np.ndarray: converted into
         #ak.layout.ListOffsetArray.

    See also #ak.to_list.
    """
    if isinstance(iterable, dict):
        if allow_record:
            return from_iter(
                [iterable],
                highlevel=highlevel,
                behavior=behavior,
                initial=initial,
                resize=resize,
            )[0]
        else:
            raise ValueError(
                "cannot produce an array from a single dict (that would be a record)"
            )

    builder = ak.layout.ArrayBuilder(initial=initial, resize=resize)
    for x in iterable:
        builder.fromiter(x)

    formstr, length, buffers = builder.to_buffers()
    form = ak._v2.forms.from_json(formstr)

    return ak._v2.operations.convert.from_buffers(
        form, length, buffers, highlevel=highlevel
    )
