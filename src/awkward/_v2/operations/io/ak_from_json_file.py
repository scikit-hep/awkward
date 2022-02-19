# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak
import json

np = ak.nplike.NumpyMetadata.instance()


class NDJSONDecoder(json.JSONDecoder):
    """
    JSON decoder class to implement 'ndjson' specs:
    see https://github.com/ndjson/ndjson-spec

    - accept newline as line delimiter '\n' (0x0A) as well as carriage return and newline '\r\n' (0x0D0A)
    - ignore empty lines '\n\n'
    - raise an error if JSON is nor parsable
    """

    def decode(self, s, *args, **kwargs):
        lines = s.splitlines()
        non_empty_lines = [line for line in lines if line.strip() != ""]
        lines = ",".join(non_empty_lines)
        text = f"[{lines}]"
        return super().decode(text, *args, **kwargs)


def from_json_file(
    source,
    nan_string=None,
    infinity_string=None,
    minus_infinity_string=None,
    complex_record_fields=None,
    highlevel=False,
    behavior=None,
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
        buffersize (int): Size (in bytes) of the buffer used by the JSON
            parser.

    Converts content of a JSON file into an Awkward Array.

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

    with open(source, encoding="utf-8") as f:
        data = json.load(f, cls=NDJSONDecoder)

    layout = ak._v2.operations.convert.from_iter(data).layout

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

    nonfinite_dict = {}
    if nan_string is not None:
        nonfinite_dict[nan_string] = np.nan
    if infinity_string is not None:
        nonfinite_dict[infinity_string] = np.inf
    if minus_infinity_string is not None:
        nonfinite_dict[minus_infinity_string] = -np.inf

    def string_to_nonfinite(node, **kwargs):
        if isinstance(node, ak._v2.contents.ListOffsetArray):
            if node.parameter("__array__") == "string":
                return node._awkward_strings_to_nonfinite(nonfinite_dict)

    layout = layout.recursively_apply(string_to_nonfinite) if nonfinite_dict else layout

    layout = (
        layout.content
        if isinstance(layout, ak._v2.contents.ListOffsetArray)
        else layout
    )

    return ak._v2._util.wrap(layout, behavior, highlevel)
