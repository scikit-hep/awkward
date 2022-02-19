# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak
import json

np = ak.nplike.NumpyMetadata.instance()


def from_json(
    source,
    nan_string=None,
    infinity_string=None,
    minus_infinity_string=None,
    complex_record_fields=None,
    highlevel=True,
    behavior=None,
):
    """
    Args:
        source (str): JSON-formatted string to convert into an array. The string
            must comply with 'ndjson' specification: please, see
            https://github.com/ndjson/ndjson-spec for more details.
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

    Converts a JSON string into an Awkward Array.

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

    lines = source.splitlines()
    if len(lines) == 1:
        out = json.loads(source)
    else:
        non_empty_lines = [line for line in lines if line.strip() != ""]
        lines = ",".join(non_empty_lines)
        text = f"[{lines}]"
        out = json.loads(text)

    try:
        iter(out)
        layout = ak._v2.operations.convert.from_iter(out).layout
    except TypeError:
        return out

    def record_to_complex(node, **kwargs):
        if isinstance(node, ak._v2.contents.RecordArray):
            keys = node.fields
            if complex_record_fields[0] in keys and complex_record_fields[1] in keys:
                real = node[complex_record_fields[0]]
                imag = node[complex_record_fields[1]]
                if (
                    isinstance(real, ak._v2.contents.NumpyArray)
                    and len(real.shape) == 1
                    and isinstance(imag, ak._v2.contents.NumpyArray)
                    and len(imag.shape) == 1
                ):
                    return ak._v2.contents.NumpyArray(
                        node._nplike.asarray(real) + node._nplike.asarray(imag) * 1j
                    )
                else:
                    raise ValueError("Complex number fields must be numbers")
                return ak._v2.contents.NumpyArray(real + imag * 1j)

    layout = (
        layout
        if complex_imag_string is None
        else layout.recursively_apply(record_to_complex)
    )

    if highlevel:
        return ak._v2._util.wrap(layout, behavior, highlevel)
    else:
        return layout
