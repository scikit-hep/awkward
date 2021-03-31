# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import types
import awkward as ak
import numbers

np = ak.nplike.NumpyMetadata.instance()
numpy = ak.nplike.Numpy.instance()


def _find_dataptrs(layout):
    def find_nparray_ptrs(node, depth, data_ptrs):
        if isinstance(node, ak.layout.NumpyArray):
            data_ptrs.append(node.ptr)

    data_ptrs = []
    ak._util.recursive_walk(layout, find_nparray_ptrs, args=(data_ptrs,))

    return data_ptrs


def _find_dataptrs_and_map(layout, jaxtracers):
    if isinstance(layout, ak.layout.Content):
        data_ptrs = _find_dataptrs(layout)
        assert len(jaxtracers) == len(data_ptrs)
        map_ptrs_to_tracers = dict(zip(data_ptrs, jaxtracers))
    else:
        # Layout is a scalar
        data_ptrs = None
        map_ptrs_to_tracers = None

    return data_ptrs, map_ptrs_to_tracers


def _jaxtracers_getitem(layout, where, jaxtracers, data_ptrs, map_ptrs_to_tracers):
    if not isinstance(layout, ak.layout.Content):
        raise TypeError("Cannot slice a scalar")

    out = layout[where]

    def find_nparray_node_newptr(layout, outlayout):
        def find_nparray_node(node, depth, fieldloc, shape, nodenum, nodenum_index):
            if isinstance(node, ak.layout.NumpyArray):
                if (
                    node.identities.fieldloc == fieldloc
                    and np.asarray(node.identities).shape[1] == shape[1]
                ):
                    nodenum_index = nodenum
                    return
                else:
                    nodenum = nodenum + 1

        outlayout_fieldloc = outlayout.identities.fieldloc
        nodenum_index = -1
        ak._util.recursive_walk(
            layout,
            find_nparray_node,
            args=(
                outlayout_fieldloc,
                np.asarray(outlayout.identities).shape,
                0,
                nodenum_index,
            ),
        )
        if nodenum_index == -1:
            raise ValueError("Couldn't find the node in new slice")
        return nodenum_index

    if not isinstance(out, ak.layout.Content):

        def recurse(outlayout, recurse_where):
            if isinstance(recurse_where, numbers.Integral) or isinstance(
                recurse_where, str
            ):
                if isinstance(outlayout, ak.layout.NumpyArray):
                    if outlayout.ptr in map_ptrs_to_tracers:
                        tracer = map_ptrs_to_tracers[outlayout.ptr]
                    else:
                        tracer = jaxtracers[find_nparray_node_newptr(layout, outlayout)]
                    return tracer[recurse_where]
            elif isinstance(where, tuple):
                return recurse(outlayout[where[:-1]], where[len(where) - 1])
            else:
                raise ValueError("Can't slice the array with {0}".format(where))

        child = [recurse(out, where)]
        return ak.Array.set_jaxtracers(out, child)

    else:

        def fetch_indices_and_fieldloc_layout(outlayout):
            if isinstance(outlayout, ak.layout.NumpyArray):
                return [
                    (
                        (
                            outlayout.identities.fieldloc,
                            numpy.asarray(outlayout.identities).shape[1],
                        ),
                        numpy.asarray(outlayout.identities),
                    )
                ]
            elif isinstance(outlayout, ak._util.listtypes):
                return fetch_indices_and_fieldloc_layout(outlayout.content)
            elif isinstance(outlayout, ak._util.indexedtypes):
                return fetch_indices_and_fieldloc_layout(outlayout.project())
            elif isinstance(outlayout, ak._util.uniontypes):
                raise ValueError(
                    "Can't differntiate an UnionArray type {0}".format(outlayout)
                )
            elif isinstance(outlayout, ak._util.recordtypes):
                indices = []
                for content in outlayout.contents:
                    indices = indices + fetch_indices_and_fieldloc_layout(content)
                return indices
            elif isinstance(outlayout, ak._util.indexedtypes):
                return fetch_indices_and_fieldloc_layout(outlayout.content)
            elif isinstance(outlayout, ak._util.indexedoptiontypes):
                return fetch_indices_and_fieldloc_layout(outlayout.content)
            elif isinstance(
                outlayout,
                (
                    ak.layout.BitMaskedArray,
                    ak.layout.ByteMaskedArray,
                    ak.layout.UnmaskedArray,
                ),
            ):
                return fetch_indices_and_fieldloc_layout(outlayout.content)
            else:
                raise NotImplementedError

        def fetch_children_tracer(outlayout, preslice_identities, children=[]):
            if isinstance(outlayout, ak.layout.NumpyArray):

                def find_intersection_indices(
                    postslice_identities, preslice_identities
                ):
                    multiplier = numpy.append(
                        numpy.cumprod(
                            (numpy.max(preslice_identities, axis=0) + 1)[::-1]
                        )[-2::-1],
                        1,
                    )
                    haystack = numpy.sum(preslice_identities * multiplier, axis=1)
                    needle = numpy.sum(postslice_identities * multiplier, axis=1)
                    return numpy.searchsorted(haystack, needle)

                def find_corresponding_identity(
                    postslice_identities, preslice_identities
                ):
                    for identity in preslice_identities:
                        if identity[0] == postslice_identities:
                            return identity[1]
                    raise ValueError(
                        "Couldn't find postslice identities in preslice identities"
                    )

                indices = find_intersection_indices(
                    numpy.asarray(outlayout.identities),
                    find_corresponding_identity(
                        (
                            outlayout.identities.fieldloc,
                            numpy.asarray(outlayout.identities).shape[1],
                        ),
                        preslice_identities,
                    ),
                )
                if outlayout.ptr in map_ptrs_to_tracers:
                    tracer = map_ptrs_to_tracers[outlayout.ptr]
                else:
                    tracer = jaxtracers[find_nparray_node_newptr(layout, outlayout)]
                children.append(jax.numpy.take(tracer, indices))
                return children

            elif isinstance(outlayout, ak._util.listtypes):
                return fetch_children_tracer(outlayout.content, preslice_identities)
            elif isinstance(outlayout, ak._util.uniontypes):
                raise ValueError(
                    "Can't differntiate an UnionArray type {0}".format(layout)
                )
            elif isinstance(outlayout, ak._util.recordtypes):
                children = []
                for content in outlayout.contents:
                    children = children + fetch_children_tracer(
                        content, preslice_identities
                    )
                return children
            elif isinstance(outlayout, ak._util.indexedtypes):
                return fetch_children_tracer(outlayout.content, preslice_identities)
            elif isinstance(outlayout, ak._util.indexedoptiontypes):
                return fetch_children_tracer(outlayout.content, preslice_identities)
            elif isinstance(
                outlayout,
                (
                    ak.layout.BitMaskedArray,
                    ak.layout.ByteMaskedArray,
                    ak.layout.UnmaskedArray,
                ),
            ):
                return fetch_children_tracer(outlayout.content, preslice_identities)
            else:
                raise NotImplementedError(
                    "fetch_children_tracer not completely implemented yet for {0}".format(
                        outlayout
                    )
                )

        children = fetch_children_tracer(out, fetch_indices_and_fieldloc_layout(layout))
        out = out.deep_copy()
        out.setidentities()
        return ak.Array.set_jaxtracers(out, children)


def array_ufunc(array, ufunc, method, inputs, kwargs):
    for x in inputs:
        if isinstance(x, ak.Array) and hasattr(x, "_tracers"):
            assert len(x._tracers) == len(array._tracers)

    # ak.Array __add__, etc. map to the NumPy functions, switch to JAX
    import numpy

    for name, np_ufunc in numpy.core.umath.__dict__.items():
        if ufunc is np_ufunc:
            ufunc = getattr(jax.numpy, name)

    # need to apply the ufunc to the same argument list for each tracer separately
    nexttracers = []
    for i in range(len(array._tracers)):
        nextinputs = [
            x._tracers[i] if isinstance(x, ak.Array) and hasattr(x, "_tracers") else x
            for x in inputs
        ]
        nexttracers.append(getattr(ufunc, method)(*nextinputs, **kwargs))

    outarray = ak._connect._numpy.array_ufunc(ufunc, method, inputs, kwargs)

    return ak.Array.set_jaxtracers(outarray, nexttracers)


class AuxData(object):
    def __init__(self, layout):
        self.layout = layout

    def __eq__(self, other):
        print(self.form, other.form)
        return True


def special_flatten(array):
    if isinstance(array, ak.Array) and hasattr(array, "_tracers"):
        aux_data, children = AuxData(array.layout), array._tracers
    elif isinstance(array, ak.Array):

        def create_databuffers(node, depth, databuffers):
            if isinstance(node, ak.layout.NumpyArray):
                databuffers.append(node)

        databuffers = []
        ak._util.recursive_walk(array.layout, create_databuffers, args=(databuffers,))

        array.layout.setidentities()
        aux_data = AuxData(array.layout)
        children = [jax.numpy.asarray(x) for x in databuffers]
    else:
        raise ValueError(
            "Can only differentiate Awkward Arrays, recieved array of type {0}".format(
                type(array)
            )
        )

    return children, aux_data


def special_unflatten(aux_data, children):
    if any(isinstance(x, jax.core.Tracer) for x in children):
        return ak.Array.set_jaxtracers(aux_data.layout, children)
    elif all(child is None for child in children):
        return None
    else:
        if aux_data.layout is None:
            assert len(children) == 1
            return np.ndarray.item(np.asarray(children[0]))

        def function(layout, num=0):
            if isinstance(layout, ak.layout.NumpyArray):
                num = num + 1
                return lambda: ak.layout.NumpyArray(children[num - 1])

        arr = ak._util.recursively_apply(aux_data.layout, function, pass_depth=False)
        return ak.Array(arr)


jax.tree_util.register_pytree_node(ak.Array, special_flatten, special_unflatten)
