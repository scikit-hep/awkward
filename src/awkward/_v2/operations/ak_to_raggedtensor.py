import awkward as ak


def to_raggedtensor(array):
    """
    Args:
        array: Data to convert to a RaggedTensor.

    Converts `array` into a `tensorflow.RaggedTensor` object.

    This function converts a subset of Awkward Array types into a RaggedTensor. Only
    list-types are supported: records, options, and unions, have no corresponding
    implementation in TensorFlow.
    """
    with ak._v2._util.OperationErrorContext(
        "ak._v2.to_raggedtensor",
        {"array": array},
    ):
        return _impl(array)


def _impl(array):
    layout = ak._v2.operations.to_layout(array, allow_record=False, allow_other=False)
    return layout.to_raggedtensor()
