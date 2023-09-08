# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
__all__ = ("argcartesian",)
from collections.abc import Mapping

import awkward as ak
from awkward._backends.dispatch import backend_of
from awkward._backends.numpy import NumpyBackend
from awkward._behavior import behavior_of
from awkward._dispatch import high_level_function
from awkward._layout import wrap_layout
from awkward._nplikes.numpylike import NumpyMetadata
from awkward._regularize import regularize_axis

np = NumpyMetadata.instance()
cpu = NumpyBackend.instance()


@high_level_function()
def argcartesian(
    arrays,
    axis=1,
    *,
    nested=None,
    parameters=None,
    with_name=None,
    highlevel=True,
    behavior=None,
):
    """
    Args:
        arrays (dict or iterable of arrays): Each value in this dict or iterable
            can be any array-like data that #ak.to_layout recognizes.
        axis (int): The dimension at which this operation is applied. The
            outermost dimension is `0`, followed by `1`, etc., and negative
            values count backward from the innermost: `-1` is the innermost
            dimension, `-2` is the next level up, etc.
        nested (None, True, False, or iterable of str or int): If None or
            False, all combinations of elements from the `arrays` are
            produced at the same level of nesting; if True, they are grouped
            in nested lists by combinations that share a common item from
            each of the `arrays`; if an iterable of str or int, group common
            items for a chosen set of keys from the `array` dict or slots
            of the `array` iterable.
        parameters (None or dict): Parameters for the new
            #ak.contents.RecordArray node that is created by this operation.
        with_name (None or str): Assigns a `"__record__"` name to the new
            #ak.contents.RecordArray node that is created by this operation
            (overriding `parameters`, if necessary).
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Computes a Cartesian product (i.e. cross product) of data from a set of
    `arrays`, like #ak.cartesian, but returning integer indexes for
    #ak.Array.__getitem__.

    For example, the Cartesian product of

        >>> one = ak.Array([1.1, 2.2, 3.3])
        >>> two = ak.Array(["a", "b"])

    is

        >>> ak.cartesian([one, two], axis=0).show()
        [(1.1, 'a'),
         (1.1, 'b'),
         (2.2, 'a'),
         (2.2, 'b'),
         (3.3, 'a'),
         (3.3, 'b')]

    But with argcartesian, only the indexes are returned.

        >>> ak.argcartesian([one, two], axis=0).show()
        [(0, 0),
         (0, 1),
         (1, 0),
         (1, 1),
         (2, 0),
         (2, 1)]

    These are the indexes that can select the items that go into the actual
    Cartesian product.

        >>> one_index, two_index = ak.unzip(ak.argcartesian([one, two], axis=0))
        >>> one[one_index]
        <Array [1.1, 1.1, 2.2, 2.2, 3.3, 3.3] type='6 * float64'>
        >>> two[two_index]
        <Array ['a', 'b', 'a', 'b', 'a', 'b'] type='6 * string'>

    All of the parameters for #ak.cartesian apply equally to #ak.argcartesian,
    so see the #ak.cartesian documentation for a more complete description.
    """
    # Dispatch
    if isinstance(arrays, Mapping):
        yield arrays.values()
    else:
        yield arrays

    # Implementation
    return _impl(arrays, axis, nested, parameters, with_name, highlevel, behavior)


def _impl(arrays, axis, nested, parameters, with_name, highlevel, behavior):
    axis = regularize_axis(axis)

    if isinstance(arrays, dict):
        behavior = behavior_of(*arrays.values(), behavior=behavior)
        backend = backend_of(*arrays.values(), default=cpu, coerce_to_common=True)
        layouts = {
            n: ak._do.local_index(
                ak.operations.to_layout(x, allow_record=False, allow_other=False),
                axis,
            ).to_backend(backend)
            for n, x in arrays.items()
        }
    else:
        arrays = list(arrays)
        behavior = behavior_of(*arrays, behavior=behavior)
        backend = backend_of(*arrays, default=cpu, coerce_to_common=True)
        layouts = [
            ak._do.local_index(
                ak.operations.to_layout(x, allow_record=False, allow_other=False),
                axis,
            ).to_backend(backend)
            for x in arrays
        ]

    if with_name is not None:
        if parameters is None:
            parameters = {}
        else:
            parameters = dict(parameters)
        parameters["__record__"] = with_name

    result = ak.operations.cartesian(
        layouts,
        axis=axis,
        nested=nested,
        parameters=parameters,
        highlevel=False,
        behavior=behavior,
    )

    return wrap_layout(result, behavior, highlevel)
