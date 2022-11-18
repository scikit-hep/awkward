# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from awkward_cpp.lib import _ext

import awkward as ak

np = ak.nplikes.NumpyMetadata.instance()


def from_iter(
    iterable, highlevel=True, behavior=None, allow_record=True, initial=1024, resize=1.5
):
    """
    Args:
        iterable (Python iterable): Data to convert into an Awkward Array.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        allow_record (bool): If True, the outermost element may be a record
            (returning #ak.Record or #ak.record.Record type, depending on
            `highlevel`); if False, the outermost element must be an array.
        initial (int): Initial size (in bytes) of buffers used by the
            [ak::ArrayBuilder](_static/classawkward_1_1ArrayBuilder.html).
        resize (float): Resize multiplier for buffers used by the
            [ak::ArrayBuilder](_static/classawkward_1_1ArrayBuilder.html);
            should be strictly greater than 1.

    Converts Python data into an Awkward Array.

    Internally, this function uses [ak::ArrayBuilder](_static/classawkward_1_1ArrayBuilder.html)
    (see the high-level #ak.ArrayBuilder documentation for a more complete description),
    so it has the same flexibility and the same constraints. Any heterogeneous
    and deeply nested Python data can be converted, but the output will never
    have regular-typed array lengths.

    The following Python types are supported.

       * bool, including `np.bool_`: converted into #ak.contents.NumpyArray.
       * int, including `np.integer`: converted into #ak.contents.NumpyArray.
       * float, including `np.floating`: converted into #ak.contents.NumpyArray.
       * bytes: converted into #ak.contents.ListOffsetArray with parameter
         `"__array__"` equal to `"bytestring"` (unencoded bytes).
       * str: converted into #ak.contents.ListOffsetArray with parameter
         `"__array__"` equal to `"string"` (UTF-8 encoded string).
       * tuple: converted into #ak.contents.RecordArray without field names
         (i.e. homogeneously typed, uniform sized tuples).
       * dict: converted into #ak.contents.RecordArray with field names
         (i.e. homogeneously typed records with the same sets of fields).
       * iterable, including np.ndarray: converted into
         #ak.contents.ListOffsetArray.

    See also #ak.to_list.
    """
    with ak._errors.OperationErrorContext(
        "ak.from_iter",
        dict(
            iterable=iterable,
            highlevel=highlevel,
            behavior=behavior,
            allow_record=allow_record,
            initial=initial,
            resize=resize,
        ),
    ):
        return _impl(iterable, highlevel, behavior, allow_record, initial, resize)


def _impl(iterable, highlevel, behavior, allow_record, initial, resize):
    if isinstance(iterable, dict):
        if allow_record:
            return _impl(
                [iterable],
                highlevel,
                behavior,
                False,
                initial,
                resize,
            )[0]
        else:
            raise ak._errors.wrap_error(
                ValueError(
                    "cannot produce an array from a single dict (that would be a record)"
                )
            )

    if isinstance(iterable, tuple):
        iterable = list(iterable)

    builder = _ext.ArrayBuilder(initial=initial, resize=resize)
    builder.fromiter(iterable)

    formstr, length, buffers = builder.to_buffers()
    form = ak.forms.from_json(formstr)

    return ak.operations.from_buffers(
        form, length, buffers, highlevel=highlevel, behavior=behavior
    )[0]
