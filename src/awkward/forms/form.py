# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import itertools
import json
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from fnmatch import fnmatchcase
from glob import escape as escape_glob

import awkward as ak
from awkward._backends.numpy import NumpyBackend
from awkward._meta.meta import Meta
from awkward._nplikes.numpy_like import NumpyMetadata
from awkward._nplikes.shape import ShapeItem, unknown_length
from awkward._parameters import (
    parameters_are_equal,
    parameters_union,
    type_parameters_equal,
)
from awkward._typing import (
    Any,
    DType,
    Final,
    Iterator,
    JSONMapping,
    Self,
)

__all__ = ("from_dict", "from_type", "from_json", "reserved_nominal_parameters", "Form")

np = NumpyMetadata.instance()
numpy_backend = NumpyBackend.instance()


reserved_nominal_parameters: Final = frozenset(
    {
        ("__array__", "string"),
        ("__array__", "bytestring"),
        ("__array__", "char"),
        ("__array__", "byte"),
        ("__array__", "sorted_map"),
        ("__array__", "categorical"),
    }
)


def from_dict(input: Mapping) -> Form:
    assert input is not None
    if isinstance(input, str):
        return ak.forms.NumpyForm(primitive=input)

    assert isinstance(input, Mapping)
    parameters = input.get("parameters", None)
    form_key = input.get("form_key", None)

    if input["class"] == "NumpyArray":
        primitive = input["primitive"]
        inner_shape = tuple(
            unknown_length if item is None else item
            for item in input.get("inner_shape", [])
        )
        return ak.forms.NumpyForm(
            primitive, inner_shape, parameters=parameters, form_key=form_key
        )

    elif input["class"] == "EmptyArray":
        return ak.forms.EmptyForm(parameters=parameters, form_key=form_key)

    elif input["class"] == "RegularArray":
        return ak.forms.RegularForm(
            content=from_dict(input["content"]),
            size=unknown_length if input["size"] is None else input["size"],
            parameters=parameters,
            form_key=form_key,
        )

    elif input["class"] in ("ListArray", "ListArray32", "ListArrayU32", "ListArray64"):
        return ak.forms.ListForm(
            starts=input["starts"],
            stops=input["stops"],
            content=from_dict(input["content"]),
            parameters=parameters,
            form_key=form_key,
        )

    elif input["class"] in (
        "ListOffsetArray",
        "ListOffsetArray32",
        "ListOffsetArrayU32",
        "ListOffsetArray64",
    ):
        return ak.forms.ListOffsetForm(
            offsets=input["offsets"],
            content=from_dict(input["content"]),
            parameters=parameters,
            form_key=form_key,
        )

    elif input["class"] == "RecordArray":
        # New serialisation
        if "fields" in input:
            if isinstance(input["contents"], Mapping):
                raise TypeError("new-style RecordForm contents must not be mappings")
            contents = [from_dict(content) for content in input["contents"]]
            fields = input["fields"]
        # Old style record
        elif isinstance(input["contents"], Mapping):
            contents = []
            fields = []
            for key, content in input["contents"].items():
                contents.append(from_dict(content))
                fields.append(key)
        # Old style tuple
        else:
            contents = [from_dict(content) for content in input["contents"]]
            fields = None
        return ak.forms.RecordForm(
            contents=contents,
            fields=fields,
            parameters=parameters,
            form_key=form_key,
        )

    elif input["class"] in (
        "IndexedArray",
        "IndexedArray32",
        "IndexedArrayU32",
        "IndexedArray64",
    ):
        return ak.forms.IndexedForm(
            index=input["index"],
            content=from_dict(input["content"]),
            parameters=parameters,
            form_key=form_key,
        )

    elif input["class"] in (
        "IndexedOptionArray",
        "IndexedOptionArray32",
        "IndexedOptionArray64",
    ):
        return ak.forms.IndexedOptionForm(
            index=input["index"],
            content=from_dict(input["content"]),
            parameters=parameters,
            form_key=form_key,
        )

    elif input["class"] == "ByteMaskedArray":
        return ak.forms.ByteMaskedForm(
            mask=input["mask"],
            content=from_dict(input["content"]),
            valid_when=input["valid_when"],
            parameters=parameters,
            form_key=form_key,
        )

    elif input["class"] == "BitMaskedArray":
        return ak.forms.BitMaskedForm(
            mask=input["mask"],
            content=from_dict(input["content"]),
            valid_when=input["valid_when"],
            lsb_order=input["lsb_order"],
            parameters=parameters,
            form_key=form_key,
        )

    elif input["class"] == "UnmaskedArray":
        return ak.forms.UnmaskedForm(
            content=from_dict(input["content"]),
            parameters=parameters,
            form_key=form_key,
        )

    elif input["class"] in (
        "UnionArray",
        "UnionArray8_32",
        "UnionArray8_U32",
        "UnionArray8_64",
    ):
        return ak.forms.UnionForm(
            tags=input["tags"],
            index=input["index"],
            contents=[from_dict(content) for content in input["contents"]],
            parameters=parameters,
            form_key=form_key,
        )

    elif input["class"] == "VirtualArray":
        raise ValueError("Awkward 1.x VirtualArrays are not supported")

    else:
        raise ValueError(
            "input class: {} was not recognised".format(repr(input["class"]))
        )


def from_json(input: str) -> Form:
    return from_dict(json.loads(input))


def from_type(type_: ak.types.Type) -> Form:
    # Categorical types are reintroduced into forms using metadata
    if type_.parameter("__categorical__"):
        # Drop categorical placeholder parameter
        if type_._parameters is None:
            next_parameters = None
        else:
            next_parameters = type_._parameters.copy()
            next_parameters.pop("__categorical__")

        if isinstance(type_, ak.types.OptionType):
            next_content = from_type(type_.content)
            return ak.forms.IndexedOptionForm(
                "i64",
                next_content,
                parameters=parameters_union(
                    next_parameters, {"__array__": "categorical"}
                ),
            )
        else:
            next_content = from_type(type_.copy(parameters=next_parameters))
            return ak.forms.IndexedForm(
                "i64", next_content, parameters={"__array__": "categorical"}
            )

    if isinstance(type_, ak.types.NumpyType):
        return ak.forms.NumpyForm(type_.primitive, parameters=type_._parameters)
    elif isinstance(type_, ak.types.ListType):
        return ak.forms.ListOffsetForm(
            "i64", from_type(type_.content), parameters=type_._parameters
        )
    elif isinstance(type_, ak.types.RegularType):
        return ak.forms.RegularForm(
            from_type(type_.content),
            size=type_.size,
            parameters=type_._parameters,
        )
    elif isinstance(type_, ak.types.OptionType):
        return ak.forms.IndexedOptionForm(
            "i64",
            from_type(type_.content),
            parameters=type_._parameters,
        )
    elif isinstance(type_, ak.types.RecordType):
        return ak.forms.RecordForm(
            [from_type(c) for c in type_.contents],
            type_.fields,
            parameters=type_._parameters,
        )
    elif isinstance(type_, ak.types.UnionType):
        return ak.forms.UnionForm(
            "i8",
            "i64",
            [from_type(c) for c in type_.contents],
            parameters=type_._parameters,
        )
    elif isinstance(type_, ak.types.UnknownType):
        return ak.forms.EmptyForm(parameters=type_._parameters)
    elif isinstance(type_, (ak.types.ArrayType, ak.types.ScalarType)):
        raise TypeError(
            "High-level types (ak.types.ArrayType, ak.types.ScalarType) do not have representations as Awkward forms. "
            "Instead the low level type should be used."
        )
    else:
        raise TypeError(f"unsupported type {type_!r}")


def _expand_braces(text, seen=None):
    if seen is None:
        seen = set()

    spans = [m.span() for m in re.finditer(r"\{[^\{\}]*\}", text)][::-1]
    alts = [text[start + 1 : stop - 1].split(",") for start, stop in spans]

    if len(spans) == 0:
        if text not in seen:
            yield text
        seen.add(text)

    else:
        for combo in itertools.product(*alts):
            replaced = list(text)
            for (start, stop), replacement in zip(spans, combo):
                replaced[start:stop] = replacement
            yield from _expand_braces("".join(replaced), seen)


class _SpecifierMatcher:
    def __init__(
        self, specifiers: Iterable[list[str]], *, match_if_empty: bool = False
    ):
        # We'll build two sets of unique fixed-strings and patterns
        fixed_strings = set()
        patterns = set()
        # And then map these unique strings to their child specifiers
        match_to_next_specifiers: defaultdict[str, list[list[str]]] = defaultdict(list)

        # For each specifier, categorise it as a fixed-string or pattern,
        # and build the next-specifier table
        parent: str
        child: list[str]
        for item in specifiers:
            parent, *child = item

            if escape_glob(parent) == parent:
                fixed_strings.add(parent)
            else:
                patterns.add(parent)

            # Only include child specifier list if it is non-empty
            if child:
                match_to_next_specifiers[parent].append(child)

        self._match_to_next_specifiers = match_to_next_specifiers
        self._fixed_strings = fixed_strings
        self._patterns = patterns
        self._match_if_empty = match_if_empty

    @property
    def is_empty(self) -> bool:
        return not (self._patterns or self._fixed_strings)

    def __call__(self, field: str, *, next_match_if_empty: bool = False) -> Self | None:
        has_matched = False

        # Fixed-strings are an O(log n) lookup
        next_specifiers = []
        if field in self._fixed_strings:
            has_matched = True
            next_specifiers.extend(self._match_to_next_specifiers[field])

        # Fixed-strings are an O(n) lookup
        for pattern in self._patterns:
            if fnmatchcase(field, pattern):
                has_matched = True
                next_specifiers.extend(self._match_to_next_specifiers[pattern])

        if has_matched:
            return type(self)(next_specifiers, match_if_empty=next_match_if_empty)
        elif self.is_empty and self._match_if_empty:
            return self
        else:
            return None


def regularize_buffer_key(buffer_key: str | Callable) -> Callable[[Form, str], str]:
    if isinstance(buffer_key, str):

        def getkey(form, attribute):
            return buffer_key.format(form_key=form.form_key, attribute=attribute)

        return getkey

    elif callable(buffer_key):

        def getkey(form, attribute):
            return buffer_key(form_key=form.form_key, attribute=attribute, form=form)

        return getkey

    else:
        raise TypeError(
            f"buffer_key must be a string or a callable, not {type(buffer_key)}"
        )


index_to_dtype: Final[dict[str, DType]] = {
    "i8": np.dtype("<i1"),
    "u8": np.dtype("<u1"),
    "i32": np.dtype("<i4"),
    "u32": np.dtype("<u4"),
    "i64": np.dtype("<i8"),
}


class Form(Meta):
    def _init(self, *, parameters: JSONMapping | None, form_key: str | None):
        if parameters is not None and not isinstance(parameters, dict):
            raise TypeError(
                f"{type(self).__name__} 'parameters' must be of type dict or None, not {parameters!r}"
            )
        if form_key is not None and not isinstance(form_key, str):
            raise TypeError(
                f"{type(self).__name__} 'form_key' must be of type string or None, not {form_key!r}"
            )

        self._parameters = parameters
        self._form_key = form_key

    @property
    def form_key(self):
        return self._form_key

    @form_key.setter
    def form_key(self, value):
        if value is not None and not isinstance(value, str):
            raise TypeError("form_key must be None or a string")
        self._form_key = value

    def __str__(self):
        return json.dumps(self.to_dict(verbose=False), indent=4)

    def to_dict(self, verbose=True):
        return self._to_dict_part(verbose, toplevel=True)

    def _to_dict_extra(self, out, verbose):
        if verbose or (self._parameters is not None and len(self._parameters) > 0):
            out["parameters"] = self.parameters
        if verbose or self._form_key is not None:
            out["form_key"] = self._form_key
        return out

    def to_json(self):
        return json.dumps(self.to_dict(verbose=True))

    def _repr_args(self):
        out = []
        if self._parameters is not None and len(self._parameters) > 0:
            out.append("parameters=" + repr(self._parameters))
        if self._form_key is not None:
            out.append("form_key=" + repr(self._form_key))
        return out

    @property
    def type(self):
        raise NotImplementedError

    def columns(self, list_indicator=None, column_prefix=()):
        output = []
        self._columns(column_prefix, output, list_indicator)
        return output

    def select_columns(
        self, specifier, expand_braces=True, *, prune_unions_and_records: bool = True
    ):
        if isinstance(specifier, str):
            specifier = {specifier}

        # Only take unique specifiers
        for item in specifier:
            if not isinstance(item, str):
                raise TypeError(
                    "a column-selection specifier must be a list of non-empty strings"
                )
            if not item:
                raise ValueError(
                    "a column-selection specifier must be a list of non-empty strings"
                )

        if expand_braces:
            next_specifier = []
            for item in specifier:
                for result in _expand_braces(item):
                    next_specifier.append(result)
            specifier = next_specifier

        specifier = [[] if item == "" else item.split(".") for item in set(specifier)]
        match_specifier = _SpecifierMatcher(specifier, match_if_empty=False)
        selection = self._select_columns(match_specifier)
        assert selection is not None, "top-level selections always return a Form"

        if prune_unions_and_records:
            return selection._prune_columns(False)
        else:
            return selection

    def column_types(self):
        return self._column_types()

    def _columns(self, path, output, list_indicator):
        raise NotImplementedError

    def _prune_columns(self, is_inside_record_or_union: bool) -> Form | None:
        raise NotImplementedError

    def _select_columns(self, match_specifier: _SpecifierMatcher) -> Form | None:
        raise NotImplementedError

    def _column_types(self):
        raise NotImplementedError

    def _to_dict_part(self, verbose, toplevel):
        raise NotImplementedError

    def length_zero_array(
        self, *, backend=numpy_backend, highlevel=False, behavior=None
    ):
        if highlevel:
            raise ValueError(
                "The `highlevel=True` variant of `Form.length_zero_array` has been removed. "
                "Please use `ak.Array(form.length_zero_array(...), behavior=...)` if an `ak.Array` is required.",
            )

        return ak.operations.ak_from_buffers._impl(
            form=self,
            length=0,
            container={"": b"\x00\x00\x00\x00\x00\x00\x00\x00"},
            buffer_key="",
            backend=backend,
            byteorder=ak._util.native_byteorder,
            highlevel=False,
            behavior=behavior,
            attrs=None,
            simplify=False,
        )

    def length_one_array(
        self, *, backend=numpy_backend, highlevel=False, behavior=None
    ):
        if highlevel:
            raise ValueError(
                "The `highlevel=True` variant of `Form.length_one_array` has been removed. "
                "Please use `ak.Array(form.length_one_array(...), behavior=...)` if an `ak.Array` is required.",
            )

        # The naive implementation of a length-1 array requires that we have a sufficiently
        # large buffer to be able to build _any_ subtree.
        def max_prefer_unknown(this: ShapeItem, that: ShapeItem) -> ShapeItem:
            if this is unknown_length:
                return this
            if that is unknown_length:
                return that
            return max(this, that)

        container = {}

        def prepare(form, multiplier):
            form_key = f"node-{len(container)}"

            if isinstance(form, (ak.forms.BitMaskedForm, ak.forms.ByteMaskedForm)):
                if form.valid_when:
                    container[form_key] = b"\x00" * multiplier
                else:
                    container[form_key] = b"\xff" * multiplier
                return form.copy(form_key=form_key)  # DO NOT RECURSE

            elif isinstance(form, ak.forms.IndexedOptionForm):
                container[form_key] = b"\xff\xff\xff\xff\xff\xff\xff\xff"  # -1
                return form.copy(form_key=form_key)  # DO NOT RECURSE

            elif isinstance(form, ak.forms.EmptyForm):
                # no error if protected by non-recursing node type
                raise TypeError(
                    "cannot generate a length_one_array from a Form with an "
                    "unknowntype that cannot be hidden (EmptyForm not within "
                    "BitMaskedForm, ByteMaskedForm, or IndexedOptionForm)"
                )

            elif isinstance(form, ak.forms.UnmaskedForm):
                return form.copy(content=prepare(form.content, multiplier))

            elif isinstance(form, (ak.forms.IndexedForm, ak.forms.ListForm)):
                container[form_key] = b"\x00" * (8 * multiplier)
                return form.copy(
                    content=prepare(form.content, multiplier), form_key=form_key
                )

            elif isinstance(form, ak.forms.ListOffsetForm):
                # offsets length == array length + 1
                container[form_key] = b"\x00" * (8 * (multiplier + 1))
                return form.copy(
                    content=prepare(form.content, multiplier), form_key=form_key
                )

            elif isinstance(form, ak.forms.RegularForm):
                size = form.size

                # https://github.com/scikit-hep/awkward/pull/2499#discussion_r1220503454
                if size is unknown_length:
                    size = 1

                return form.copy(content=prepare(form.content, multiplier * size))

            elif isinstance(form, ak.forms.NumpyForm):
                dtype = ak.types.numpytype.primitive_to_dtype(form._primitive)
                size = multiplier * dtype.itemsize
                for x in form.inner_shape:
                    if x is not unknown_length:
                        size *= x

                container[form_key] = b"\x00" * size
                return form.copy(form_key=form_key)

            elif isinstance(form, ak.forms.RecordForm):
                return form.copy(
                    # recurse down all contents
                    contents=[prepare(x, multiplier) for x in form.contents]
                )

            elif isinstance(form, ak.forms.UnionForm):
                # both tags and index will get this buffer, but index is 8 bytes
                container[form_key] = b"\x00" * (8 * multiplier)
                return form.copy(
                    # only recurse down contents[0] because all index == 0
                    contents=(
                        [prepare(form.contents[0], multiplier)] + form.contents[1:]
                    ),
                    form_key=form_key,
                )

            else:
                raise AssertionError(f"not a Form: {form!r}")

        return ak.operations.ak_from_buffers._impl(
            form=prepare(self, 1),
            length=1,
            container=container,
            buffer_key="{form_key}",
            backend=backend,
            byteorder=ak._util.native_byteorder,
            highlevel=False,
            behavior=behavior,
            attrs=None,
            simplify=False,
        )

    def _expected_from_buffers(
        self, getkey: Callable[[Form, str], str], recursive: bool
    ) -> Iterator[tuple[str, DType]]:
        raise NotImplementedError

    def expected_from_buffers(
        self, buffer_key="{form_key}-{attribute}", recursive=True
    ):
        """
        Args:
            buffer_key (str or callable): Python format string containing
                `"{form_key}"` and/or `"{attribute}"` or a function that takes these
                as keyword arguments and returns a string to use as a key for a buffer
                in the `container`.
            recursive (bool): If True, recurse into subforms; otherwise, yield
                only the (buffer_key, dtype) pairs for this form object.

        Yield (buffer_key, dtype) pairs describing the expected buffer keys,
        and their corresponding dtypes, that a call to #ak.from_buffers would
        be expected to find from the `container` object.
        """
        getkey = regularize_buffer_key(buffer_key)
        return dict(self._expected_from_buffers(getkey, recursive))

    def is_equal_to(
        self, other: Any, *, all_parameters: bool = False, form_key: bool = False
    ) -> bool:
        return self._is_equal_to(other, all_parameters, form_key)

    __eq__ = is_equal_to

    def _is_equal_to(self, other: Any, all_parameters: bool, form_key: bool) -> bool:
        raise NotImplementedError

    def _is_equal_to_generic(
        self, other: Any, all_parameters: bool, form_key: bool
    ) -> bool:
        compare_parameters = (
            parameters_are_equal if all_parameters else type_parameters_equal
        )

        return (
            isinstance(other, type(self))
            and not (form_key and self._form_key != other._form_key)
            and compare_parameters(self._parameters, other._parameters)
        )
