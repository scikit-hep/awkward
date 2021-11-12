# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import math

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()
numpy = ak.nplike.Numpy.instance()


def from_buffers(
    form,
    length,
    container,
    buffer_key="{form_key}-{attribute}",
    nplike=numpy,
    highlevel=True,
    behavior=None,
):
    """
    Args:
        form (#ak.forms.Form or str/dict equivalent): The form of the Awkward
            Array to reconstitute from named buffers.
        length (int): Length of the array. (The output of this function is always
            single-partition.)
        container (Mapping, such as dict): The str \u2192 Python buffers that
            represent the decomposed Awkward Array. This `container` is only
            assumed to have a `__getitem__` method that accepts strings as keys.
        buffer_key (str or callable): Python format string containing
            `"{form_key}"` and/or `"{attribute}"` or a function that takes these
            as keyword arguments and returns a string to use as a key for a buffer
            in the `container`.
        nplike (#ak.nplike.NumpyLike): Library to use to generate values that are
            put into the new array. The default, #ak.nplike.Numpy, makes NumPy
            arrays, which are in main memory (e.g. not GPU). If all the values in
            `container` have the same `nplike` as this, they won't be copied.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.layout.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Reconstitutes an Awkward Array from a Form, length, and a collection of memory
    buffers, so that data can be losslessly read from file formats and storage
    devices that only map names to binary blobs (such as a filesystem directory).

    The first three arguments of this function are the return values of
    #ak.to_buffers, so a full round-trip is

        >>> reconstituted = ak.from_buffers(*ak.to_buffers(original))

    The `container` argument lets you specify your own Mapping, which might be
    an interface to some storage format or device (e.g. h5py). It's okay if
    the `container` dropped NumPy's `dtype` and `shape` information, leaving
    raw bytes, since `dtype` and `shape` can be reconstituted from the
    #ak.forms.NumpyForm.

    The `buffer_key` should be the same as the one used in #ak.to_buffers.

    See #ak.to_buffers for examples.
    """
    if ak._v2._util.isstr(form):
        if ak._v2.types.numpytype.is_primitive(form):
            form = ak._v2.forms.NumpyForm(form)
        else:
            form = ak._v2.forms.from_json(form)
    elif isinstance(form, dict):
        form = ak._v2.forms.from_iter(form)

    if not (ak._v2._util.isint(length) and length >= 0):
        raise TypeError("'length' argument must be a non-negative integer")

    if not isinstance(form, ak._v2.forms.Form):
        raise TypeError(
            "'form' argument must be a Form or its Python dict/JSON string representation"
        )

    if ak._v2._util.isstr(buffer_key):

        def getkey(form, attribute):
            return buffer_key.format(form_key=form.form_key, attribute=attribute)

    elif callable(buffer_key):

        def getkey(form, attribute):
            return buffer_key(form_key=form.form_key, attribute=attribute, form=form)

    else:
        raise TypeError(
            "buffer_key must be a string or a callable, not {0}".format(
                type(buffer_key)
            )
        )

    out = reconstitute(form, length, container, getkey, nplike)
    return ak._v2._util.wrap(out, behavior, highlevel)


_index_to_dtype = {
    "i8": np.dtype("<i1"),
    "u8": np.dtype("<u1"),
    "i32": np.dtype("<i4"),
    "u32": np.dtype("<u4"),
    "i64": np.dtype("<i8"),
}


def reconstitute(form, length, container, getkey, nplike):
    if form.has_identifier:
        raise NotImplementedError("ak.from_buffers for an array with an Identifier")
    else:
        identifier = None

    if isinstance(form, ak._v2.forms.EmptyForm):
        if length != 0:
            raise ValueError(
                "EmptyForm node, but the expected length is {0}".format(length)
            )
        return ak._v2.contents.EmptyArray(identifier, form.parameters)

    elif isinstance(form, ak._v2.forms.NumpyForm):
        dtype = ak._v2.types.numpytype.primitive_to_dtype(form.primitive)
        raw_array = container[getkey(form, "data")]
        real_length = length
        for x in form.inner_shape:
            real_length *= x
        data = nplike.frombuffer(raw_array, dtype=dtype, count=real_length)
        if form.inner_shape != ():
            if len(data) == 0:
                data = data.reshape((length,) + form.inner_shape)
            else:
                data = data.reshape((-1,) + form.inner_shape)
        return ak._v2.contents.NumpyArray(data, identifier, form.parameters, nplike)

    elif isinstance(form, ak._v2.forms.UnmaskedForm):
        content = reconstitute(form.content, length, container, getkey, nplike)
        return ak._v2.contents.UnmaskedArray(content, identifier, form.parameters)

    elif isinstance(form, ak._v2.forms.BitMaskedForm):
        raw_array = container[getkey(form, "mask")]
        excess_length = int(math.ceil(length / 8.0))
        mask = nplike.frombuffer(
            raw_array, dtype=_index_to_dtype[form.mask], count=excess_length
        )
        return ak._v2.contents.BitMaskedArray(
            ak._v2.index.Index(mask),
            reconstitute(form.content, length, container, getkey, nplike),
            form.valid_when,
            length,
            form.lsb_order,
            identifier,
            form.parameters,
        )

    elif isinstance(form, ak._v2.forms.ByteMaskedForm):
        raw_array = container[getkey(form, "mask")]
        mask = nplike.frombuffer(
            raw_array, dtype=_index_to_dtype[form.mask], count=length
        )
        return ak._v2.contents.ByteMaskedArray(
            ak._v2.index.Index(mask),
            reconstitute(form.content, length, container, getkey, nplike),
            form.valid_when,
            identifier,
            form.parameters,
        )

    elif isinstance(form, ak._v2.forms.IndexedOptionForm):
        raw_array = container[getkey(form, "index")]
        index = nplike.frombuffer(
            raw_array, dtype=_index_to_dtype[form.index], count=length
        )
        next_length = 0 if len(index) == 0 else max(0, nplike.max(index) + 1)
        return ak._v2.contents.IndexedOptionArray(
            ak._v2.index.Index(index),
            reconstitute(form.content, next_length, container, getkey, nplike),
            identifier,
            form.parameters,
        )

    elif isinstance(form, ak._v2.forms.IndexedForm):
        raw_array = container[getkey(form, "index")]
        index = nplike.frombuffer(
            raw_array, dtype=_index_to_dtype[form.index], count=length
        )
        next_length = 0 if len(index) == 0 else nplike.max(index) + 1
        return ak._v2.contents.IndexedArray(
            ak._v2.index.Index(index),
            reconstitute(form.content, next_length, container, getkey, nplike),
            identifier,
            form.parameters,
        )

    elif isinstance(form, ak._v2.forms.ListForm):
        raw_array1 = container[getkey(form, "starts")]
        raw_array2 = container[getkey(form, "stops")]
        starts = nplike.frombuffer(
            raw_array1, dtype=_index_to_dtype[form.starts], count=length
        )
        stops = nplike.frombuffer(
            raw_array2, dtype=_index_to_dtype[form.stops], count=length
        )
        reduced_stops = stops[starts != stops]
        next_length = 0 if len(starts) == 0 else nplike.max(reduced_stops)
        return ak._v2.contents.ListArray(
            ak._v2.index.Index(starts),
            ak._v2.index.Index(stops),
            reconstitute(form.content, next_length, container, getkey, nplike),
            identifier,
            form.parameters,
        )

    elif isinstance(form, ak._v2.forms.ListOffsetForm):
        raw_array = container[getkey(form, "offsets")]
        offsets = nplike.frombuffer(
            raw_array, dtype=_index_to_dtype[form.offsets], count=length + 1
        )
        next_length = 0 if len(offsets) == 1 else offsets[-1]
        return ak._v2.contents.ListOffsetArray(
            ak._v2.index.Index(offsets),
            reconstitute(form.content, next_length, container, getkey, nplike),
            identifier,
            form.parameters,
        )

    elif isinstance(form, ak._v2.forms.RegularForm):
        next_length = length * form.size
        return ak._v2.contents.RegularArray(
            reconstitute(form.content, next_length, container, getkey, nplike),
            form.size,
            length,
            identifier,
            form.parameters,
        )

    elif isinstance(form, ak._v2.forms.RecordForm):
        return ak._v2.contents.RecordArray(
            [
                reconstitute(content, length, container, getkey, nplike)
                for content in form.contents
            ],
            None if form.is_tuple else form.fields,
            length,
            identifier,
            form.parameters,
        )

    elif isinstance(form, ak._v2.forms.UnionForm):
        raw_array1 = container[getkey(form, "tags")]
        raw_array2 = container[getkey(form, "index")]
        tags = nplike.frombuffer(
            raw_array1, dtype=_index_to_dtype[form.tags], count=length
        )
        index = nplike.frombuffer(
            raw_array2, dtype=_index_to_dtype[form.index], count=length
        )
        lengths = []
        for tag in range(len(form.contents)):
            selected_index = index[tags == tag]
            if len(selected_index) == 0:
                lengths.append(0)
            else:
                lengths.append(nplike.max(selected_index) + 1)
        return ak._v2.contents.UnionArray(
            ak._v2.index.Index(tags),
            ak._v2.index.Index(index),
            [
                reconstitute(content, lengths[i], container, getkey, nplike)
                for i, content in enumerate(form.contents)
            ],
            identifier,
            form.parameters,
        )

    else:
        raise AssertionError("unexpected form node type: " + str(type(form)))
