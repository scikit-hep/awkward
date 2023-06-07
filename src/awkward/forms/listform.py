# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
import awkward as ak
from awkward._parameters import type_parameters_equal
from awkward._typing import final
from awkward._util import UNSET
from awkward.forms.form import Form


@final
class ListForm(Form):
    is_list = True

    def __init__(
        self,
        starts,
        stops,
        content,
        *,
        parameters=None,
        form_key=None,
    ):
        if not isinstance(starts, str):
            raise TypeError(
                "{} 'starts' must be of type str, not {}".format(
                    type(self).__name__, repr(starts)
                )
            )
        if not isinstance(stops, str):
            raise TypeError(
                "{} 'starts' must be of type str, not {}".format(
                    type(self).__name__, repr(starts)
                )
            )
        if not isinstance(content, Form):
            raise TypeError(
                "{} all 'contents' must be Form subclasses, not {}".format(
                    type(self).__name__, repr(content)
                )
            )

        self._starts = starts
        self._stops = stops
        self._content = content
        self._init(parameters=parameters, form_key=form_key)

    @property
    def starts(self):
        return self._starts

    @property
    def stops(self):
        return self._stops

    @property
    def content(self):
        return self._content

    def copy(
        self,
        starts=UNSET,
        stops=UNSET,
        content=UNSET,
        *,
        parameters=UNSET,
        form_key=UNSET,
    ):
        return ListForm(
            self._starts if starts is UNSET else starts,
            self._stops if stops is UNSET else stops,
            self._content if content is UNSET else content,
            parameters=self._parameters if parameters is UNSET else parameters,
            form_key=self._form_key if form_key is UNSET else form_key,
        )

    @classmethod
    def simplified(
        cls,
        starts,
        stops,
        content,
        *,
        parameters=None,
        form_key=None,
    ):
        return cls(starts, stops, content, parameters=parameters, form_key=form_key)

    def __repr__(self):
        args = [
            repr(self._starts),
            repr(self._stops),
            repr(self._content),
            *self._repr_args(),
        ]
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def _to_dict_part(self, verbose, toplevel):
        return self._to_dict_extra(
            {
                "class": "ListArray",
                "starts": self._starts,
                "stops": self._stops,
                "content": self._content._to_dict_part(verbose, toplevel=False),
            },
            verbose,
        )

    @property
    def type(self):
        return ak.types.ListType(
            self._content.type,
            parameters=self._parameters,
        )

    def __eq__(self, other):
        if isinstance(other, ListForm):
            return (
                self._form_key == other._form_key
                and self._starts == other._starts
                and self._stops == other._stops
                and type_parameters_equal(self._parameters, other._parameters)
                and self._content == other._content
            )
        else:
            return False

    def purelist_parameter(self, key):
        if self._parameters is None or key not in self._parameters:
            return self._content.purelist_parameter(key)
        else:
            return self._parameters[key]

    @property
    def purelist_isregular(self):
        return False

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

    def _select_columns(self, index, specifier, matches, output):
        return ListForm(
            self._starts,
            self._stops,
            self._content._select_columns(index, specifier, matches, output),
            parameters=self._parameters,
            form_key=self._form_key,
        )

    def _column_types(self):
        if self.parameter("__array__") in ("string", "bytestring"):
            return ("string",)
        else:
            return self._content._column_types()
