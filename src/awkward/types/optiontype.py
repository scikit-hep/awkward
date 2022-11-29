# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak
from awkward.forms.form import _parameters_equal
from awkward.types.listtype import ListType
from awkward.types.regulartype import RegularType
from awkward.types.type import Type
from awkward.types.uniontype import UnionType


class OptionType(Type):
    def __init__(self, content, *, parameters=None, typestr=None):
        if not isinstance(content, Type):
            raise ak._errors.wrap_error(
                TypeError(
                    "{} 'content' must be a Type subclass, not {}".format(
                        type(self).__name__, repr(content)
                    )
                )
            )
        if parameters is not None and not isinstance(parameters, dict):
            raise ak._errors.wrap_error(
                TypeError(
                    "{} 'parameters' must be of type dict or None, not {}".format(
                        type(self).__name__, repr(parameters)
                    )
                )
            )
        if typestr is not None and not isinstance(typestr, str):
            raise ak._errors.wrap_error(
                TypeError(
                    "{} 'typestr' must be of type string or None, not {}".format(
                        type(self).__name__, repr(typestr)
                    )
                )
            )
        self._content = content
        self._parameters = parameters
        self._typestr = typestr

    @property
    def content(self):
        return self._content

    def _str(self, indent, compact):
        head = []
        tail = []
        if self._typestr is not None:
            content_out = [self._typestr]

        else:
            content_out = self._content._str(indent, compact)
            params = self._str_parameters()
            if params is None:
                if isinstance(
                    self._content, (RegularType, ListType)
                ) and not self._content.parameter("__array__") in (
                    "string",
                    "bytestring",
                    "char",
                    "byte",
                ):
                    head = ["option["]
                    tail = ["]"]
                else:
                    head = ["?"]

            else:
                head = ["option["]
                tail = [f", {params}]"]

        return (
            head
            + [self._str_categorical_begin()]
            + content_out
            + [self._str_categorical_end()]
            + tail
        )

    def __repr__(self):
        args = [repr(self._content)] + self._repr_args()
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def __eq__(self, other):
        if isinstance(other, OptionType):
            return (
                _parameters_equal(
                    self._parameters, other._parameters, only_array_record=True
                )
                and self._content == other._content
            )
        else:
            return False

    def simplify_option_union(self):
        if isinstance(self._content, UnionType):
            contents = []
            for content in self._content.contents:
                if isinstance(content, OptionType):
                    typestr = (
                        content._typestr if self._typestr is None else self._typestr
                    )
                    contents.append(
                        OptionType(
                            content.content,
                            parameters=ak._util.merge_parameters(
                                self._parameters, content._parameters
                            ),
                            typestr=typestr,
                        )
                    )

                else:
                    contents.append(
                        OptionType(
                            content, parameters=self._parameters, typestr=self._typestr
                        )
                    )

            return UnionType(
                contents,
                parameters=self._content.parameters,
                typestr=self._content.typestr,
            )

        else:
            return self
