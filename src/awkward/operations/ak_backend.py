# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak


def backend(*arrays):
    """
    Returns the names of the backend used by `arrays`. May be

       * `"cpu"` for `libawkward-cpu-kernels.so`;
       * `"cuda"` for `libawkward-cuda-kernels.so`;
       * `"mixed"` if any of the arrays have different labels within their
         structure or any arrays have different labels from each other;
       * None if the objects are not Awkward, NumPy, or CuPy arrays (e.g.
         Python numbers, booleans, strings).

    Mixed arrays can't be used in any operations, and two arrays on different
    devices can't be used in the same operation.

    To use `"cuda"`, the package
    [awkward-cuda-kernels](https://pypi.org/project/awkward-cuda-kernels)
    be installed, either by

        pip install awkward-cuda-kernels

    or as an optional dependency with

        pip install awkward[cuda] --upgrade

    It is only available for Linux as a binary wheel, and only supports Nvidia
    GPUs (it is written in CUDA).

    See #ak.to_backend.
    """
    with ak._v2._util.OperationErrorContext(
        "ak._v2.backend",
        {"*arrays": arrays},
    ):
        return _impl(arrays)


def _impl(arrays):
    backends = set()
    for array in arrays:
        layout = ak._v2.operations.to_layout(
            array,
            allow_record=True,
            allow_other=True,
        )
        if isinstance(layout, (ak._v2.contents.Content, ak._v2.index.Index)):
            if isinstance(layout.nplike, ak.nplike.Numpy):
                backends.add("cpu")
            elif isinstance(layout.nplike, ak.nplike.Cupy):
                backends.add("cuda")
            elif isinstance(layout.nplike, ak.nplike.Jax):
                backends.add("jax")
        elif isinstance(layout, ak.nplike.numpy.ndarray):
            backends.add("cpu")
        elif type(layout).__module__.startswith("cupy."):
            backends.add("cuda")
        elif type(layout).__module__.startswith("jaxlib."):
            backends.add("jax")

    if backends == set():
        return None
    elif backends == {"cpu"}:
        return "cpu"
    elif backends == {"cuda"}:
        return "cuda"
    elif backends == {"jax"}:
        return "jax"
    else:
        return "mixed"
