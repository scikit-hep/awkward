# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from collections.abc import Iterable

import awkward as ak
from awkward._util import unset
from awkward.forms.form import Form, _parameters_equal


class UnionForm(Form):
    is_union = True

    def __init__(
        self,
        tags,
        index,
        contents,
        *,
        parameters=None,
        form_key=None,
    ):
        if not isinstance(tags, str):
            raise ak._errors.wrap_error(
                TypeError(
                    "{} 'tags' must be of type str, not {}".format(
                        type(self).__name__, repr(tags)
                    )
                )
            )
        if not isinstance(index, str):
            raise ak._errors.wrap_error(
                TypeError(
                    "{} 'index' must be of type str, not {}".format(
                        type(self).__name__, repr(index)
                    )
                )
            )
        if not isinstance(contents, Iterable):
            raise ak._errors.wrap_error(
                TypeError(
                    "{} 'contents' must be iterable, not {}".format(
                        type(self).__name__, repr(contents)
                    )
                )
            )
        for content in contents:
            if not isinstance(content, Form):
                raise ak._errors.wrap_error(
                    TypeError(
                        "{} all 'contents' must be Form subclasses, not {}".format(
                            type(self).__name__, repr(content)
                        )
                    )
                )

        self._tags = tags
        self._index = index
        self._contents = list(contents)
        self._init(parameters, form_key)

    @property
    def tags(self):
        return self._tags

    @property
    def index(self):
        return self._index

    @property
    def contents(self):
        return self._contents

    def copy(
        self,
        tags=unset,
        index=unset,
        contents=unset,
        *,
        parameters=unset,
        form_key=unset,
    ):
        return UnionForm(
            self._tags if tags is unset else tags,
            self._index if index is unset else index,
            self._contents if contents is unset else contents,
            parameters=self._parameters if parameters is unset else parameters,
            form_key=self._form_key if form_key is unset else form_key,
        )

    @classmethod
    def simplified(
        cls,
        tags,
        index,
        contents,
        *,
        parameters=None,
        form_key=None,
    ):
        return ak.contents.UnionArray.simplified(
            ak.index._form_to_zero_length(tags),
            ak.index._form_to_zero_length(index),
            [x.length_zero_array(highlevel=False) for x in contents],
            parameters=parameters,
        ).form

    def _union_of_optionarrays(self, index, parameters):
        return (
            self.length_zero_array(highlevel=False)
            ._union_of_optionarrays(ak.index._form_to_zero_length(index), parameters)
            .form
        )

    def content(self, index):
        return self._contents[index]

    def __repr__(self):
        args = [
            repr(self._tags),
            repr(self._index),
            repr(self._contents),
        ] + self._repr_args()
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def _to_dict_part(self, verbose, toplevel):
        return self._to_dict_extra(
            {
                "class": "UnionArray",
                "tags": self._tags,
                "index": self._index,
                "contents": [
                    content._to_dict_part(verbose, toplevel=False)
                    for content in self._contents
                ],
            },
            verbose,
        )

    def _type(self, typestrs):
        return ak.types.UnionType(
            [x._type(typestrs) for x in self._contents],
            parameters=self._parameters,
            typestr=ak._util.gettypestr(self._parameters, typestrs),
        )

    def __eq__(self, other):
        if (
            isinstance(other, UnionForm)
            and self._form_key == other._form_key
            and self._tags == other._tags
            and self._index == other._index
            and len(self._contents) == len(other._contents)
            and _parameters_equal(
                self._parameters, other._parameters, only_array_record=True
            )
        ):
            return self._contents == other._contents

        return False

    def purelist_parameter(self, key):
        if self._parameters is None or key not in self._parameters:
            out = self._contents[0].purelist_parameter(key)
            for content in self._contents[1:]:
                tmp = content.purelist_parameter(key)
                if out != tmp:
                    return None
            return out
        else:
            return self._parameters[key]

    @property
    def purelist_isregular(self):
        for content in self._contents:
            if not content.purelist_isregular:
                return False
        return True

    @property
    def purelist_depth(self):
        out = None
        for content in self._contents:
            if out is None:
                out = content.purelist_depth
            elif out != content.purelist_depth:
                return -1
        return out

    @property
    def is_identity_like(self):
        return False

    @property
    def minmax_depth(self):
        if len(self._contents) == 0:
            return (0, 0)
        mins, maxs = [], []
        for content in self._contents:
            mindepth, maxdepth = content.minmax_depth
            mins.append(mindepth)
            maxs.append(maxdepth)
        return (min(mins), max(maxs))

    @property
    def branch_depth(self):
        if len(self._contents) == 0:
            return (False, 1)
        anybranch = False
        mindepth = None
        for content in self._contents:
            branch, depth = content.branch_depth
            if mindepth is None:
                mindepth = depth
            if branch or mindepth != depth:
                anybranch = True
            if mindepth > depth:
                mindepth = depth
        return (anybranch, mindepth)

    @property
    def fields(self):
        fieldslists = [cont.fields for cont in self._contents]
        return list(set.intersection(*[set(x) for x in fieldslists]))

    @property
    def is_tuple(self):
        return all(x.is_tuple for x in self._contents) and (len(self._contents) > 0)

    @property
    def dimension_optiontype(self):
        for content in self._contents:
            if content.dimension_optiontype:
                return True
        return False

    def _columns(self, path, output, list_indicator):
        for content, field in zip(self._contents, self.fields):
            content._columns(path + (field,), output, list_indicator)

    def _select_columns(self, index, specifier, matches, output):
        contents = []
        for content in self._contents:
            len_output = len(output)
            next_content = content._select_columns(index, specifier, matches, output)
            if len_output != len(output):
                contents.append(next_content)

        if len(contents) == 0:
            return ak.forms.EmptyForm(
                parameters=self._parameters, form_key=self._form_key
            )
        elif len(contents) == 1:
            return contents[0]
        else:
            return UnionForm(
                self._tags,
                self._index,
                contents,
                parameters=self._parameters,
                form_key=self._form_key,
            )

    def _column_types(self):
        return sum((x._column_types() for x in self._contents), ())
