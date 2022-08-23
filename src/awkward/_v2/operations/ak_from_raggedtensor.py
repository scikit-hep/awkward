# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak
import awkward._v2._connect.tensorflow

np = ak.nplike.NumpyMetadata.instance()
numpy = ak.nplike.Numpy.instance()


def from_raggedtensor(tensor, highlevel=True, behavior=None):
    """
    Args:
        tensor: (`tensorflow.RaggedTensor` or `tensorflow.Tensor`) RaggedTensor to convert to an {class}`ak.Array`.
            Must be either a `RaggedTensor
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Perform a zero-copy conversion from a `tensorflow.RaggedTensor` to an {class}`ak.Array`
    """
    with ak._v2._util.OperationErrorContext(
        "ak._v2.from_ragggedtensor",
        {"tensor": tensor},
    ):
        return _impl(
            tensor,
            highlevel,
            behavior,
        )


def _impl(
    tensor,
    highlevel,
    behavior,
):
    tensorflow = awkward._v2._connect.tensorflow.import_tensorflow()  # noqa: F401
    if not isinstance(tensor, tensorflow.RaggedTensor):
        raise ak._v2._util.error(
            TypeError(f"expected `tensorflow.RaggedTensor` object, not {type(tensor)}")
        )

    layout = _tensor_to_layout(tensor, tensorflow)
    if highlevel:
        return ak._v2._util.wrap(layout, behavior, highlevel)
    else:
        return layout


def _tensor_to_layout(tensor, tensorflow):
    if tensor.ragged_rank > 0:
        stops = tensor.row_limits().numpy()
        offsets = numpy.zeros(len(stops) + 1, dtype=np.int64)
        offsets[1:] = stops
        return ak._v2.contents.ListOffsetArray(
            ak._v2.index.Index64(offsets),
            _tensor_to_layout(tensor.values, tensorflow),
        )
    else:
        return ak._v2.contents.NumpyArray(tensor.numpy())
