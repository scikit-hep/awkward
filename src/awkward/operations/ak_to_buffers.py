# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak._nplikes.NumpyMetadata.instance()


def to_buffers(
    array,
    container=None,
    buffer_key="{form_key}-{attribute}",
    form_key="node{id}",
    *,
    id_start=0,
    backend=None,
):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        container (None or MutableMapping): The str \u2192 NumPy arrays (or
            Python buffers) that represent the decomposed Awkward Array. This
            `container` is only assumed to have a `__setitem__` method that
            accepts strings as keys.
        buffer_key (str or callable): Python format string containing
            `"{form_key}"` and/or `"{attribute}"` or a function that takes these
            (and/or `layout`) as keyword arguments and returns a string to use
            as a key for a buffer in the `container`. The `form_key` is the result
            of applying `form_key` (below), and the `attribute` is a hard-coded
            string representing the buffer's function (e.g. `"data"`, `"offsets"`,
            `"index"`).
        form_key (str, callable): Python format string containing
            `"{id}"` or a function that takes this (and/or `layout`) as a keyword
            argument and returns a string to use as a key for a Form node.
            Together, the `buffer_key` and `form_key` links attributes of each Form
            node to data in the `container`.
        id_start (int): Starting `id` to use in `form_key` and hence `buffer_key`.
            This integer increases in a depth-first walk over the `array` nodes and
            can be used to generate unique keys for each Form.
        backend (`"cpu"`, `"cuda"`, `"jax"`, None): Backend to use to
            generate values that are put into the `container`. The default,
            `"cpu"`, makes NumPy arrays, which are in main memory
            (e.g. not GPU) and satisfy Python's Buffer protocol. If all the
            buffers in `array` have the same `backend` as this, they won't be
            copied. If the backend is None, then the backend of the layout
            will be used to generate the buffers.

    Decomposes an Awkward Array into a Form and a collection of memory buffers,
    so that data can be losslessly written to file formats and storage devices
    that only map names to binary blobs (such as a filesystem directory).

    This function returns a 3-tuple:

        (form, length, container)

    where the `form` is a #ak.forms.Form (whose string representation is JSON),
    the `length` is an integer (`len(array)`), and the `container` is either
    the MutableMapping you passed in or a new dict containing the buffers (as
    NumPy arrays).

    These are also the first three arguments of #ak.from_buffers, so a full
    round-trip is

        >>> reconstituted = ak.from_buffers(*ak.to_buffers(original))

    The `container` argument lets you specify your own MutableMapping, which
    might be an interface to some storage format or device (e.g. h5py). It's
    okay if the `container` drops NumPy's `dtype` and `shape` information,
    leaving raw bytes, since `dtype` and `shape` can be reconstituted from
    the #ak.forms.NumpyForm.

    The `buffer_key` and `form_key` arguments let you configure the names of the
    buffers added to the `container` and string labels on each Form node, so that
    the two can be uniquely matched later. `buffer_key` and `form_key` are distinct
    arguments to allow for more indirection (buffer keys can differ from Form keys,
    as long as there's a way to map them to each other) and because some Form nodes,
    such as #ak.forms.ListForm and #ak.forms.UnionForm, have more than one attribute
    (`starts` and `stops` for #ak.forms.ListForm and `tags` and `index` for
    #ak.forms.UnionForm).

    Awkward 1.x also included partition numbers (`"part0-"`, `"part1-"`, ...) in
    the buffer keys. In version 2.x onward, partitioning is handled externally by
    Dask, but partition numbers can be emulated by prepending a fixed `"partN-"`
    string to the `buffer_key`. The `array` represents exactly one partition.

    Here is a simple example:

        >>> original = ak.Array([[1, 2, 3], [], [4, 5]])
        >>> form, length, container = ak.to_buffers(original)
        >>> print(form)
        {
            "class": "ListOffsetArray",
            "offsets": "i64",
            "content": {
                "class": "NumpyArray",
                "primitive": "int64",
                "form_key": "node1"
            },
            "form_key": "node0"
        }
        >>> length
        3
        >>> container
        {'node0-offsets': array([0, 3, 3, 5]), 'node1-data': array([1, 2, 3, 4, 5])}

    which may be read back with

        >>> ak.from_buffers(form, length, container)
        <Array [[1, 2, 3], [], [4, 5]] type='3 * var * int64'>

    If you intend to use this function for saving data, you may want to pack it
    first with #ak.to_packed.

    See also #ak.from_buffers and #ak.to_packed.
    """
    with ak._errors.OperationErrorContext(
        "ak.to_buffers",
        dict(
            array=array,
            container=container,
            buffer_key=buffer_key,
            form_key=form_key,
            id_start=id_start,
            backend=backend,
        ),
    ):
        return _impl(array, container, buffer_key, form_key, id_start, backend)


def _impl(array, container, buffer_key, form_key, id_start, backend):
    layout = ak.operations.to_layout(array, allow_record=False, allow_other=False)

    if backend is not None:
        backend = ak._backends.regularize_backend(backend)

    return ak._do.to_buffers(
        layout,
        container=container,
        buffer_key=buffer_key,
        form_key=form_key,
        id_start=id_start,
        backend=backend,
    )
