# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

# This set of string-manipulation routines is strongly inspired by Arrow:

# string transforms
# https://arrow.apache.org/docs/python/api/compute.html#string-transforms
from awkward.operations.str.akstr_capitalize import *

# string padding
# https://arrow.apache.org/docs/python/api/compute.html#string-padding
from awkward.operations.str.akstr_center import *

# containment tests
# https://arrow.apache.org/docs/python/api/compute.html#containment-tests
from awkward.operations.str.akstr_count_substring import *
from awkward.operations.str.akstr_count_substring_regex import *
from awkward.operations.str.akstr_distinct_counts import *
from awkward.operations.str.akstr_ends_with import *

# string component extraction
# https://arrow.apache.org/docs/python/api/compute.html#string-component-extraction
from awkward.operations.str.akstr_extract_regex import *
from awkward.operations.str.akstr_find_substring import *
from awkward.operations.str.akstr_find_substring_regex import *
from awkward.operations.str.akstr_index_in import *

# string predicates
# https://arrow.apache.org/docs/python/api/compute.html#string-predicates
from awkward.operations.str.akstr_is_alnum import *
from awkward.operations.str.akstr_is_alpha import *
from awkward.operations.str.akstr_is_ascii import *
from awkward.operations.str.akstr_is_decimal import *
from awkward.operations.str.akstr_is_digit import *
from awkward.operations.str.akstr_is_in import *
from awkward.operations.str.akstr_is_lower import *
from awkward.operations.str.akstr_is_numeric import *
from awkward.operations.str.akstr_is_printable import *
from awkward.operations.str.akstr_is_space import *
from awkward.operations.str.akstr_is_title import *
from awkward.operations.str.akstr_is_upper import *

# string joining
# https://arrow.apache.org/docs/python/api/compute.html#string-joining
from awkward.operations.str.akstr_join import *
from awkward.operations.str.akstr_join_element_wise import *
from awkward.operations.str.akstr_length import *
from awkward.operations.str.akstr_lower import *
from awkward.operations.str.akstr_lpad import *

# string trimming
# https://arrow.apache.org/docs/python/api/compute.html#string-trimming
from awkward.operations.str.akstr_ltrim import *
from awkward.operations.str.akstr_ltrim_whitespace import *
from awkward.operations.str.akstr_match_like import *
from awkward.operations.str.akstr_match_substring import *
from awkward.operations.str.akstr_match_substring_regex import *
from awkward.operations.str.akstr_repeat import *
from awkward.operations.str.akstr_replace_slice import *
from awkward.operations.str.akstr_replace_substring import *
from awkward.operations.str.akstr_replace_substring_regex import *
from awkward.operations.str.akstr_reverse import *
from awkward.operations.str.akstr_rpad import *
from awkward.operations.str.akstr_rtrim import *
from awkward.operations.str.akstr_rtrim_whitespace import *

# string slicing
# https://arrow.apache.org/docs/python/api/compute.html#string-slicing
from awkward.operations.str.akstr_slice import *
from awkward.operations.str.akstr_split_pattern import *
from awkward.operations.str.akstr_split_pattern_regex import *

# string splitting
# https://arrow.apache.org/docs/python/api/compute.html#string-splitting
from awkward.operations.str.akstr_split_whitespace import *
from awkward.operations.str.akstr_starts_with import *
from awkward.operations.str.akstr_swapcase import *
from awkward.operations.str.akstr_title import *

# dictionary-encoding (exclusively for strings)
# https://arrow.apache.org/docs/python/api/compute.html#associative-transforms
from awkward.operations.str.akstr_to_categorical import *
from awkward.operations.str.akstr_trim import *
from awkward.operations.str.akstr_trim_whitespace import *
from awkward.operations.str.akstr_uniques import *
from awkward.operations.str.akstr_upper import *

__all__ = [k for k in locals() if not k.startswith("akstr_")]


def __dir__():
    return __all__


def _drop_option_preserving_form(layout, ensure_empty_mask: bool = False):
    from awkward._do import recursively_apply
    from awkward.contents import UnmaskedArray, IndexedOptionArray, IndexedArray

    def action(_, continuation, **kwargs):
        this = continuation()

        if not this.is_option:
            return this
        elif isinstance(this, UnmaskedArray):
            return this.content
        else:
            nplike = this.backend.nplike
            assert not (
                ensure_empty_mask
                and nplike.known_data
                and nplike.any(this.mask_as_bool(valid_when=False))
            ), "did not expect option type, but arrow returned a non-erasable option"
            # Re-write indexed options as indexed
            if isinstance(this, IndexedOptionArray):
                return IndexedArray(
                    this.index, this.content, parameters=this._parameters
                )
            else:
                return this.content

    return recursively_apply(layout, action, return_simplified=False)


def _apply_through_arrow(
    operation,
    /,
    *args,
    generate_bitmasks=False,
    expect_option_type=False,
    string_to32=False,
    bytestring_to32=False,
    ensure_empty_mask=False,
    **kwargs,
):
    from awkward._backends.dispatch import backend_of
    from awkward.contents.content import Content
    from awkward.operations.ak_from_arrow import from_arrow
    from awkward.operations.ak_to_arrow import to_arrow
    from awkward._backends.typetracer import TypeTracerBackend

    backend = backend_of(*args)

    typetracer = TypeTracerBackend.instance()
    if backend is typetracer:
        converted_args = [
            to_arrow(
                x.form.length_zero_array(),
                extensionarray=False,
                string_to32=string_to32,
                bytestring_to32=bytestring_to32,
            )
            if isinstance(x, Content)
            else x
            for x in args
        ]
        out = from_arrow(
            operation(*converted_args, **kwargs),
            generate_bitmasks=generate_bitmasks,
            highlevel=False,
        ).to_typetracer(forget_length=True)

    else:
        converted_args = [
            to_arrow(
                x,
                extensionarray=False,
                string_to32=string_to32,
                bytestring_to32=bytestring_to32,
            )
            if isinstance(x, Content)
            else x
            for x in args
        ]
        out = from_arrow(
            operation(*converted_args, **kwargs),
            generate_bitmasks=generate_bitmasks,
            highlevel=False,
        )

    if expect_option_type:
        return out
    else:
        return _drop_option_preserving_form(out, ensure_empty_mask=ensure_empty_mask)


def _get_ufunc_action(
    utf8_function,
    ascii_function,
    *args,
    generate_bitmasks=False,
    bytestring_to_string=False,
    expect_option_type=False,
    **kwargs,
):
    def action(layout, **absorb):
        if layout.is_list and layout.parameter("__array__") == "string":
            return _apply_through_arrow(
                utf8_function,
                layout,
                *args,
                generate_bitmasks=generate_bitmasks,
                expect_option_type=expect_option_type,
                **kwargs,
            )

        elif layout.is_list and layout.parameter("__array__") == "bytestring":
            if bytestring_to_string:
                out = _apply_through_arrow(
                    ascii_function,
                    layout.copy(
                        content=layout.content.copy(parameters={"__array__": "char"}),
                        parameters={"__array__": "string"},
                    ),
                    *args,
                    generate_bitmasks=generate_bitmasks,
                    expect_option_type=expect_option_type,
                    **kwargs,
                )
                if out.is_list and out.parameter("__array__") == "string":
                    out = out.copy(
                        content=out.content.copy(parameters={"__array__": "byte"}),
                        parameters={"__array__": "bytestring"},
                    )
                return out

            else:
                return _apply_through_arrow(
                    ascii_function,
                    layout,
                    *args,
                    generate_bitmasks=generate_bitmasks,
                    expect_option_type=expect_option_type,
                    **kwargs,
                )

    return action


def _get_split_action(
    utf8_function,
    ascii_function,
    *args,
    generate_bitmasks=False,
    bytestring_to_string=False,
    **kwargs,
):
    def action(layout, **_):
        if layout.is_list and layout.parameter("__array__") == "string":
            return _drop_option_preserving_form(
                _apply_through_arrow(
                    utf8_function,
                    layout,
                    *args,
                    generate_bitmasks=generate_bitmasks,
                    **kwargs,
                )
            )

        elif layout.is_list and layout.parameter("__array__") == "bytestring":
            if bytestring_to_string:
                out = _drop_option_preserving_form(
                    _apply_through_arrow(
                        ascii_function,
                        layout.copy(
                            content=layout.content.copy(
                                parameters={"__array__": "char"}
                            ),
                            parameters={"__array__": "string"},
                        ),
                        *args,
                        generate_bitmasks=generate_bitmasks,
                        **kwargs,
                    )
                )
                assert out.is_list

                assert (
                    out.content.is_list
                    and out.content.parameter("__array__") == "string"
                )
                return out.copy(
                    content=out.content.copy(
                        content=out.content.content.copy(
                            parameters={"__array__": "byte"}
                        ),
                        parameters={"__array__": "bytestring"},
                    ),
                )

            else:
                return _drop_option_preserving_form(
                    _apply_through_arrow(
                        ascii_function,
                        layout,
                        *args,
                        generate_bitmasks=generate_bitmasks,
                        **kwargs,
                    )
                )

    return action


def _is_string(layout):
    return layout.is_list and layout.parameter("__array__") in {"string", "bytestring"}


def _is_maybe_optional_string(layout):
    if _is_string(layout):
        return True
    elif layout.is_option or layout.is_indexed:
        return _is_maybe_optional_string(layout.content)
    else:
        return False


def _string_parameters(layout):
    while not _is_string(layout):
        layout = layout.content
    return layout._parameters


def _without_categorical(layout):
    # plain strings avoid arrow dictionaries and give the same form on every backend
    if _is_string(layout):
        return layout
    elif layout.parameter("__array__") == "categorical":
        return layout.copy(parameters=None)
    else:
        return layout.copy(content=_without_categorical(layout.content))


def _distinct_per_list(list_array, with_counts, function_name):
    from awkward._connect.pyarrow import import_pyarrow, import_pyarrow_compute
    from awkward._nplikes.numpy import Numpy

    pa = import_pyarrow(function_name)
    pc = import_pyarrow_compute(function_name)
    numpy = Numpy.instance()

    # missing lists never get here so every list is valid
    assert list_array.null_count == 0

    table = pa.table(
        {
            "parent": pc.list_parent_indices(list_array),
            "values": pc.list_flatten(list_array),
        }
    )
    # without threads the groups come out in order of first appearance
    groups = (
        table.drop_null()
        .group_by(["parent", "values"], use_threads=False)
        .aggregate([([], "count_all")])
    )

    # groups are sorted by parent so each list starts at its first group
    offsets = numpy.searchsorted(
        groups["parent"].to_numpy(), numpy.arange(len(list_array) + 1)
    )
    values = groups["values"].combine_chunks()
    if with_counts:
        values = pa.StructArray.from_arrays(
            [values, groups["count_all"].combine_chunks()],
            names=["values", "counts"],
        )

    return pa.LargeListArray.from_arrays(pa.array(offsets, type=pa.int64()), values)


def _distinct_strings(array, with_counts, function_name, highlevel, behavior, attrs):
    from awkward._connect.pyarrow import import_pyarrow_compute
    from awkward._do import recursively_apply
    from awkward._layout import HighLevelContext
    from awkward._nplikes.numpy_like import NumpyMetadata
    from awkward.contents import ListOffsetArray
    from awkward.index import Index64

    np = NumpyMetadata.instance()
    pc = import_pyarrow_compute(function_name)

    def with_string_parameters(out, parameters):
        # arrow drops any custom parameters on the strings
        if with_counts:
            values = out.content("values").copy(parameters=parameters)
            return out.copy(contents=[values, out.content("counts")])
        else:
            return out.copy(parameters=parameters)

    def action(layout, **kwargs):
        if layout.is_list and _is_maybe_optional_string(layout.content):
            out = _apply_through_arrow(
                _distinct_per_list,
                layout.copy(content=_without_categorical(layout.content)),
                with_counts,
                function_name,
            )

            offsets = out.offsets
            if not layout.backend.nplike.known_data:
                # the typetracer forgets the length but it matches the input
                offsets = Index64(
                    layout.backend.nplike.empty(layout.length + 1, dtype=np.int64)
                )

            return ListOffsetArray(
                offsets,
                with_string_parameters(out.content, _string_parameters(layout.content)),
                parameters=layout._parameters,
            )

        elif _is_string(layout):
            raise TypeError(
                "strings inside a record or union are not supported because "
                "the result would have a different length"
            )

    with HighLevelContext(behavior=behavior, attrs=attrs) as ctx:
        layout = ctx.unwrap(array, allow_record=False)

    if _is_maybe_optional_string(layout):
        # a flat array of strings is one big list
        kernel = pc.value_counts if with_counts else pc.unique
        out = _apply_through_arrow(
            lambda strings: kernel(pc.drop_null(strings)),
            _without_categorical(layout),
        )
        out = with_string_parameters(out, _string_parameters(layout))
    else:
        out = recursively_apply(layout, action, numpy_to_regular=False)

    return ctx.wrap(out, highlevel=highlevel)
