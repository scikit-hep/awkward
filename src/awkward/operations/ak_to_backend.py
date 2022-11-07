# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak.nplikes.NumpyMetadata.instance()


def to_backend(array, backend, highlevel=True, behavior=None):
    """
    Args:
        array: Data to convert to a specified `backend` set.
        backend (`"cpu"` or `"cuda"`): If `"cpu"`, the array structure is
            recursively copied (if need be) to main memory for use with
            the default `libawkward-cpu-kernels.so`; if `"cuda"`, the
            structure is copied to the GPU(s) for use with
            `libawkward-cuda-kernels.so`.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Converts an array from `"cpu"`, `"cuda"`, or `"mixed"` kernels to `"cpu"`
    or `"cuda"`.

    An array is `"mixed"` if some components are set to use the `"cpu"` backend and
    others are set to use the `"cuda"` backend. Mixed arrays can't be used in any
    operations, and two arrays set to different backends can't be used in the
    same operation.

    Any components that are already in the desired backend are viewed,
    rather than copied, so this operation can be an inexpensive way to ensure
    that an array is ready for a particular library.

    To use `"cuda"`, the package
    [awkward-cuda-kernels](https://pypi.org/project/awkward-cuda-kernels)
    be installed, either by

        pip install awkward-cuda-kernels

    or as an optional dependency with

        pip install awkward[cuda] --upgrade

    It is only available for Linux as a binary wheel, and only supports Nvidia
    GPUs (it is written in CUDA).

    See #ak.kernels.
    """
    with ak._errors.OperationErrorContext(
        "ak.to_backend",
        dict(array=array, backend=backend, highlevel=highlevel, behavior=behavior),
    ):
        return _impl(array, backend, highlevel, behavior)


def _impl(array, backend, highlevel, behavior):
    layout = ak.operations.to_layout(
        array,
        allow_record=True,
        allow_other=True,
    )
    behavior = ak._util.behavior_of(array, behavior=behavior)
    backend_layout = layout.to_backend(backend)
    return ak._util.wrap(backend_layout, behavior, highlevel)
