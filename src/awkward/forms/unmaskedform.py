# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

__all__ = ("UnmaskedForm",)
from collections.abc import Callable, Iterator

import awkward as ak
from awkward._nplikes.numpylike import NumpyMetadata
from awkward._parameters import parameters_union, type_parameters_equal
from awkward._typing import JSONSerializable, Self, final
from awkward._util import UNSET
from awkward.forms.form import Form

np = NumpyMetadata.instance()


@final
class UnmaskedForm(Form):
    is_option = True

    def __init__(
        self,
        content,
        *,
        parameters=None,
        form_key=None,
    ):
        if not isinstance(content, Form):
            raise TypeError(
                "{} all 'contents' must be Form subclasses, not {}".format(
                    type(self).__name__, repr(content)
                )
            )

        self._content = content
        self._init(parameters=parameters, form_key=form_key)

    @property
    def content(self):
        return self._content

    def copy(
        self,
        content=UNSET,
        *,
        parameters=UNSET,
        form_key=UNSET,
    ):
        return UnmaskedForm(
            self._content if content is UNSET else content,
            parameters=self._parameters if parameters is UNSET else parameters,
            form_key=self._form_key if form_key is UNSET else form_key,
        )

    @classmethod
    def simplified(
        cls,
        content,
        *,
        parameters=None,
        form_key=None,
    ):
        if content.is_union:
            return content.copy(
                contents=[cls.simplified(x) for x in content.contents],
                parameters=parameters_union(content._parameters, parameters),
            )
        elif content.is_indexed or content.is_option:
            return content.copy(
                parameters=parameters_union(content._parameters, parameters)
            )
        else:
            return cls(content, parameters=parameters, form_key=form_key)

    def __repr__(self):
        args = [repr(self._content), *self._repr_args()]
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def _to_dict_part(self, verbose, toplevel):
        return self._to_dict_extra(
            {
                "class": "UnmaskedArray",
                "content": self._content._to_dict_part(verbose, toplevel=False),
            },
            verbose,
        )

    @property
    def type(self):
        return ak.types.OptionType(
            self._content.type,
            parameters=self._parameters,
        ).simplify_option_union()

    def __eq__(self, other):
        if isinstance(other, UnmaskedForm):
            return (
                self._form_key == other._form_key
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
        return self._content.purelist_depth

    @property
    def is_identity_like(self):
        return self._content.is_identity_like

    @property
    def minmax_depth(self):
        return self._content.minmax_depth

    @property
    def branch_depth(self):
        return self._content.branch_depth

    @property
    def fields(self):
        return self._content.fields

    @property
    def is_tuple(self):
        return self._content.is_tuple

    @property
    def dimension_optiontype(self):
        return True

    def _columns(self, path, output, list_indicator):
        self._content._columns(path, output, list_indicator)

    def _prune_columns(self, is_inside_record_or_union: bool) -> Self | None:
        next_content = self._content._prune_columns(is_inside_record_or_union)
        if next_content is None:
            return None
        else:
            return UnmaskedForm(
                next_content,
                parameters=self._parameters,
                form_key=self._form_key,
            )

    def _select_columns(self, match_specifier):
        return UnmaskedForm(
            self._content._select_columns(match_specifier),
            parameters=self._parameters,
            form_key=self._form_key,
        )

    def _column_types(self):
        return self._content._column_types()

    def __setstate__(self, state):
        if isinstance(state, dict):
            # read data pickled in Awkward 2.x
            self.__dict__.update(state)
        else:
            # read data pickled in Awkward 1.x

            # https://github.com/scikit-hep/awkward/blob/main-v1/src/python/forms.cpp#L792-L797
            has_identities, parameters, form_key, content = state

            if form_key is not None:
                form_key = "part0-" + form_key  # only the first partition

            self.__init__(content, parameters=parameters, form_key=form_key)

    def _expected_from_buffers(
        self, getkey: Callable[[Form, str], str]
    ) -> Iterator[tuple[str, np.dtype]]:
        yield from self._content._expected_from_buffers(getkey)
