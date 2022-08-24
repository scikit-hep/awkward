# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak
import awkward._v2._connect.tensorflow

np = ak.nplike.NumpyMetadata.instance()
numpy = ak.nplike.Numpy.instance()


def from_raggedtensor(tensor, string_encoding="UTF-8", highlevel=True, behavior=None):
    """
    Args:
        tensor: (`tensorflow.RaggedTensor` or `tensorflow.Tensor`) RaggedTensor to convert to an {class}`ak.Array`.
            Must be either a `RaggedTensor
        string_encoding (str): If not None, the encoding of string values. Otherwise, bytestrings are produced.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Perform a zero-copy (where possible) conversion from a `tensorflow.RaggedTensor` to an {class}`ak.Array`.

    RaggedArrays with primitive dtypes (int, bool, float) can be converted without a copy. Strings must first be
    decoded into UTF-8 codepoints before conversion.
    """
    with ak._v2._util.OperationErrorContext(
        "ak._v2.from_ragggedtensor",
        {
            "tensor": tensor,
            "string_encoding": string_encoding,
            "highlevel": highlevel,
            "behavior": behavior,
        },
    ):
        return _impl(
            tensor,
            string_encoding,
            highlevel,
            behavior,
        )


def _impl(
    tensor,
    string_encoding,
    highlevel,
    behavior,
):
    tensorflow = awkward._v2._connect.tensorflow.import_tensorflow()  # noqa: F401

    layout = tensor_to_layout(tensor, string_encoding, tensorflow)
    if highlevel:
        return ak._v2._util.wrap(layout, behavior, highlevel)
    else:
        return layout


def stops_to_offsets(stops):
    offsets = numpy.zeros(len(stops) + 1, dtype=np.int64)
    offsets[1:] = stops
    return offsets


def tensor_to_layout(tensor, string_encoding, tensorflow, parameters=None):
    if isinstance(tensor, tensorflow.RaggedTensor):
        offsets = stops_to_offsets(tensor.row_limits().numpy())
        return ak._v2.contents.ListOffsetArray(
            ak._v2.index.Index64(offsets),
            tensor_to_layout(tensor.values, string_encoding, tensorflow),
            parameters=parameters,
        )
    # Strings are not treated by TensorFlow as a ragged dimension, but instead a regular
    # dimension with a tensorflow.string dtype
    elif isinstance(tensor, tensorflow.Tensor):
        if tensor.dtype == tensorflow.string:
            # For encoded strings, transcode them to UTF8
            if string_encoding is not None:
                # Convert into UTF8
                raw_tensor = tensorflow.strings.unicode_transcode(
                    tensor, string_encoding, "UTF-8"
                )
                string_type = "string"
                char_type = "char"
            # Otherwise, read the raw bytes
            else:
                raw_tensor = tensor
                string_type = "bytestring"
                char_type = "byte"

            content_tensor = tensorflow.strings.bytes_split(tensor).values
            byte_count = tensorflow.strings.length(raw_tensor, unit="BYTE").numpy()
            offsets = stops_to_offsets(numpy.cumsum(byte_count))
            content = tensorflow.io.decode_raw(content_tensor, tensorflow.uint8)

            return ak._v2.contents.ListOffsetArray(
                ak._v2.index.Index64(offsets),
                ak._v2.contents.NumpyArray(
                    content,
                    parameters={"__array__": char_type},
                ),
                parameters={"__array__": string_type},
            )
        else:
            return ak._v2.contents.NumpyArray(tensor.numpy(), parameters=parameters)
    else:
        raise ak._v2._util.error(
            TypeError(
                f"expected `tensorflow.RaggedTensor`, or `tensorflow.Tensor` object, "
                f"not {type(tensor)}"
            )
        )
