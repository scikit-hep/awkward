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
    with ak._errors.OperationErrorContext(
        "ak.backend",
        {"*arrays": arrays},
    ):
        return _impl(arrays)


def _impl(arrays):
    backends = set()
    for array in arrays:
        layout = ak.operations.to_layout(
            array,
            allow_record=True,
            allow_other=True,
        )
        # Find the nplike, if it is explicitly associated with this object
        nplike = ak.nplikes.nplike_of(layout, default=None)
        if nplike is None:
            continue
        if isinstance(nplike, ak.nplikes.Jax):
            backends.add("jax")
        elif isinstance(nplike, ak.nplikes.Cupy):
            backends.add("cuda")
        elif isinstance(nplike, ak.nplikes.Numpy):
            backends.add("cpu")

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
