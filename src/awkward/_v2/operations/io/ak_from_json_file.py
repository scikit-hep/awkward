# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE


import awkward as ak

np = ak.nplike.NumpyMetadata.instance()

# FIXME: implement and document the following:
#
# - accept newline as line delimiter '\n' (0x0A) as well as carriage return and newline '\r\n' (0x0D0A)
# - ignore empty lines '\n\n'
# - raise an error if JSON is nor parsable
#
# https://github.com/ndjson/ndjson-spec
#
#


def from_json_file(
    source,
    nan_string=None,
    infinity_string=None,
    minus_infinity_string=None,
    complex_record_fields=None,
    highlevel=False,
    behavior=None,
    initial=1024,
    resize=1.5,
    buffersize=65536,
):
    """
    Args:
        source (str): A filename with a JSON-formatted content to convert into an array.
        nan_string (None or str): If not None, strings with this value will be
            interpreted as floating-point NaN values.
        infinity_string (None or str): If not None, strings with this value will
            be interpreted as floating-point positive infinity values.
        minus_infinity_string (None or str): If not None, strings with this value
            will be interpreted as floating-point negative infinity values.
        complex_record_fields (None or (str, str)): If not None, defines a pair of
            field names to interpret records as complex numbers.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.
        initial (int): Initial size (in bytes) of buffers used by
            #ak.layout.ArrayBuilder (see #ak.layout.ArrayBuilderOptions).
        resize (float): Resize multiplier for buffers used by
            #ak.layout.ArrayBuilder (see #ak.layout.ArrayBuilderOptions);
            should be strictly greater than 1.
        buffersize (int): Size (in bytes) of the buffer used by the JSON
            parser.

    Converts content of a JSON file into an Awkward Array.

    Internally, this function uses #ak.layout.ArrayBuilder (see the high-level
    #ak.ArrayBuilder documentation for a more complete description), so it
    has the same flexibility and the same constraints. Any heterogeneous
    and deeply nested JSON can be converted, but the output will never have
    regular-typed array lengths.

    See also #ak.from_json_schema and #ak.to_json.
    """
    if complex_record_fields is None:
        complex_real_string = None
        complex_imag_string = None
    elif (
        isinstance(complex_record_fields, tuple)
        and len(complex_record_fields) == 2
        and isinstance(complex_record_fields[0], str)
        and isinstance(complex_record_fields[1], str)
    ):
        complex_real_string, complex_imag_string = complex_record_fields

    # FIXME: read blocks - need some changes in C++ code
    #
    # def read(file_path, out):
    #     with open(file_path, 'rb') as file_:
    #         while True:
    #             block = file_.read(block_size)
    #             if not block:
    #                 break
    #             out.send(block)
    #     out.close()

    with open(source, "rb") as f:
        builder = ak.layout.ArrayBuilder(initial=initial, resize=resize)
        # FIXME: for line in f:
        num = ak._ext.fromjson(
            f.read(),
            builder,
            nan_string=nan_string,
            infinity_string=infinity_string,
            minus_infinity_string=minus_infinity_string,
            buffersize=buffersize,
        )
        formstr, length, buffers = builder.to_buffers()
        form = ak._v2.forms.from_json(formstr)

        snapshot = ak._v2.operations.convert.from_buffers(
            form, length, buffers, highlevel=highlevel
        )
        # FIXME: the code is a copy from snapshot,
        # because this call returns v1: snapshot = builder.snapshot()
        layout = snapshot[0] if num == 1 else snapshot

    def action(recordnode, **kwargs):
        if isinstance(recordnode, ak._v2.contents.RecordArray):
            keys = recordnode.fields
            if complex_record_fields[0] in keys and complex_record_fields[1] in keys:
                real = recordnode[complex_record_fields[0]]
                imag = recordnode[complex_record_fields[1]]
                if (
                    isinstance(real, ak._v2.contents.NumpyArray)
                    and len(real.shape) == 1
                    and isinstance(imag, ak._v2.contents.NumpyArray)
                    and len(imag.shape) == 1
                ):
                    return ak._v2.contents.NumpyArray(
                        recordnode._nplike.asarray(real)
                        + recordnode._nplike.asarray(imag) * 1j
                    )
                else:
                    raise ValueError("Complex number fields must be numbers")
                return ak._v2.contents.NumpyArray(real + imag * 1j)
            else:
                return None
        else:
            return None

    if complex_imag_string is not None:
        layout = layout.recursively_apply(action)

    return ak._v2._util.wrap(layout, behavior, highlevel)
