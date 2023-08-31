# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

__all__ = ("RegularForm",)
from collections.abc import Callable, Iterator

import awkward as ak
from awkward._nplikes.numpylike import NumpyMetadata
from awkward._nplikes.shape import unknown_length
from awkward._parameters import type_parameters_equal
from awkward._regularize import is_integer
from awkward._typing import JSONSerializable, final
from awkward._util import UNSET
from awkward.forms.form import Form

np = NumpyMetadata.instance()


@final
class RegularForm(Form):
    is_list = True
    is_regular = True

    def __init__(self, content, size, *, parameters=None, form_key=None):
        if not isinstance(content, Form):
            raise TypeError(
                "{} all 'contents' must be Form subclasses, not {}".format(
                    type(self).__name__, repr(content)
                )
            )
        if not (size is unknown_length or (is_integer(size) and size >= 0)):
            raise TypeError(
                "{} 'size' must be a non-negative int or None, not {}".format(
                    type(self).__name__, repr(size)
                )
            )

        self._content = content
        self._size = size
        self._init(parameters=parameters, form_key=form_key)

    @property
    def content(self):
        return self._content

    @property
    def size(self):
        return self._size

    def copy(self, content=UNSET, size=UNSET, *, parameters=UNSET, form_key=UNSET):
        return RegularForm(
            self._content if content is UNSET else content,
            self._size if size is UNSET else size,
            parameters=self._parameters if parameters is UNSET else parameters,
            form_key=self._form_key if form_key is UNSET else form_key,
        )

    @classmethod
    def simplified(cls, content, size, *, parameters=None, form_key=None):
        return cls(content, size, parameters=parameters, form_key=form_key)

    def __repr__(self):
        args = [repr(self._content), repr(self._size), *self._repr_args()]
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def _to_dict_part(self, verbose, toplevel):
        return self._to_dict_extra(
            {
                "class": "RegularArray",
                "size": None if self._size is unknown_length else self._size,
                "content": self._content._to_dict_part(verbose, toplevel=False),
            },
            verbose,
        )

    @property
    def type(self):
        return ak.types.RegularType(
            self._content.type,
            self._size,
            parameters=self._parameters,
        )

    def __eq__(self, other):
        if isinstance(other, RegularForm):
            return (
                self._form_key == other._form_key
                and self._size == other._size
                and type_parameters_equal(self._parameters, other._parameters)
                and self._content == other._content
            )
        else:
            return False

    def purelist_parameters(self, *keys: str) -> JSONSerializable:
        if self._parameters is not None:
            for key in keys:
                if key in self._parameters:
                    return self._parameters[key]

        return self._content.purelist_parameters(*keys)

    @property
    def purelist_isregular(self):
        return self._content.purelist_isregular

    @property
    def purelist_depth(self):
        if self.parameter("__array__") in ("string", "bytestring"):
            return 1
        else:
            return self._content.purelist_depth + 1

    @property
    def is_identity_like(self):
        return False

    @property
    def minmax_depth(self):
        if self.parameter("__array__") in ("string", "bytestring"):
            return (1, 1)
        else:
            mindepth, maxdepth = self._content.minmax_depth
            return (mindepth + 1, maxdepth + 1)

    @property
    def branch_depth(self):
        if self.parameter("__array__") in ("string", "bytestring"):
            return (False, 1)
        else:
            branch, depth = self._content.branch_depth
            return (branch, depth + 1)

    @property
    def fields(self):
        return self._content.fields

    @property
    def is_tuple(self):
        return self._content.is_tuple

    @property
    def dimension_optiontype(self):
        return False

    def _columns(self, path, output, list_indicator):
        if (
            self.parameter("__array__") not in ("string", "bytestring")
            and list_indicator is not None
        ):
            path = (*path, list_indicator)
        self._content._columns(path, output, list_indicator)

    def _prune_columns(self, is_inside_record_or_union: bool):
        next_content = self._content._prune_columns(is_inside_record_or_union)
        if next_content is None:
            return None
        else:
            return RegularForm(
                next_content,
                self._size,
                parameters=self._parameters,
                form_key=self._form_key,
            )

    def _select_columns(self, match_specifier):
        return RegularForm(
            self._content._select_columns(match_specifier),
            self._size,
            parameters=self._parameters,
            form_key=self._form_key,
        )

    def _column_types(self):
        if self.parameter("__array__") in ("string", "bytestring"):
            return ("string",)
        else:
            return self._content._column_types()

    def __setstate__(self, state):
        if isinstance(state, dict):
            # read data pickled in Awkward 2.x
            self.__dict__.update(state)
        else:
            # read data pickled in Awkward 1.x

            # https://github.com/scikit-hep/awkward/blob/main-v1/src/python/forms.cpp#L692-L698
            has_identities, parameters, form_key, content, size = state

            if form_key is not None:
                form_key = "part0-" + form_key  # only the first partition

            self.__init__(content, size, parameters=parameters, form_key=form_key)

    def _expected_from_buffers(
        self, getkey: Callable[[Form, str], str]
    ) -> Iterator[tuple[str, np.dtype]]:
        yield from self._content._expected_from_buffers(getkey)
