# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

__all__ = ("index_in",)


import awkward as ak
from awkward._backends.dispatch import backend_of
from awkward._backends.numpy import NumpyBackend
from awkward._backends.typetracer import TypeTracerBackend
from awkward._behavior import behavior_of
from awkward._dispatch import high_level_function
from awkward._layout import wrap_layout

cpu = NumpyBackend.instance()
typetracer = TypeTracerBackend.instance()


@high_level_function(module="ak.str")
def index_in(array, value_set, *, skip_nones=False, highlevel=True, behavior=None):
    """
    Args:
        array: Array-like data (anything #ak.to_layout recognizes).
        value_set: Array-like data (anything #ak.to_layout recognizes), set of
            values to search for in `array`.
        skip_nones (bool): If True, None values in `array` are not matched
            against `value_set`; otherwise, None is considered a legal value.
        highlevel (bool): If True, return an #ak.Array; otherwise, return
            a low-level #ak.contents.Content subclass.
        behavior (None or dict): Custom #ak.behavior for the output array, if
            high-level.

    Returns the index of the first pattern in `value_set` that each string in
    `array` matches. If the string is not found within `value_set`, then the
    index is set to None.

    Note: this function does not raise an error if the `array` does not
    contain any string or bytestring data.

    Requires the pyarrow library and calls
    [pyarrow.compute.index_in](https://arrow.apache.org/docs/python/generated/pyarrow.compute.index_in.html).
    """
    # Dispatch
    yield (array, value_set)

    # Implementation
    return _impl(array, value_set, skip_nones, highlevel, behavior)


def _is_maybe_optional_list_of_string(layout):
    if layout.is_list and layout.parameter("__array__") in {"string", "bytestring"}:
        return True
    elif layout.is_option or layout.is_indexed:
        return _is_maybe_optional_list_of_string(layout.content)
    else:
        return False


def _impl(array, value_set, skip_nones, highlevel, behavior):
    from awkward._connect.pyarrow import import_pyarrow_compute

    pc = import_pyarrow_compute("ak.str.index_in")

    behavior = behavior_of(array, value_set, behavior=behavior)
    backend = backend_of(array, value_set, coerce_to_common=True, default=cpu)

    layout = ak.to_layout(array, allow_record=False).to_backend(backend)
    value_set_layout = ak.to_layout(value_set, allow_record=False).to_backend(backend)

    if not _is_maybe_optional_list_of_string(value_set_layout):
        raise TypeError("`value_set` must be 1D array of (possibly missing) strings")

    def apply(layout, **kwargs):
        if _is_maybe_optional_list_of_string(layout):
            if layout.backend is typetracer:
                return ak.from_arrow(
                    pc.index_in(
                        ak.to_arrow(
                            layout.form.length_zero_array(highlevel=False),
                            extensionarray=False,
                        ),
                        ak.to_arrow(
                            value_set_layout.form.length_zero_array(highlevel=False),
                            extensionarray=False,
                        ),
                        skip_nulls=skip_nones,
                    ),
                    highlevel=False,
                    generate_bitmasks=True,
                ).to_typetracer(forget_length=True)
            else:
                return ak.from_arrow(
                    pc.index_in(
                        ak.to_arrow(layout, extensionarray=False),
                        ak.to_arrow(value_set_layout, extensionarray=False),
                        skip_nulls=skip_nones,
                    ),
                    highlevel=False,
                    generate_bitmasks=True,
                )

    out = ak._do.recursively_apply(layout, apply, behavior=behavior)

    return wrap_layout(out, highlevel=highlevel, behavior=behavior)
