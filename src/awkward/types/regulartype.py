# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from awkward._nplikes.shape import unknown_length
from awkward._parameters import parameters_are_equal, type_parameters_equal
from awkward._regularize import is_integer
from awkward._typing import final
from awkward.types.type import Type


@final
class RegularType(Type):
    def __init__(self, content, size, *, parameters=None, typestr=None):
        if not isinstance(content, Type):
            raise TypeError(
                "{} 'content' must be a Type subtype, not {}".format(
                    type(self).__name__, repr(content)
                )
            )
        if not (size is unknown_length or (is_integer(size) and size >= 0)):
            raise ValueError(
                "{} 'size' must be a non-negative int or None, not {}".format(
                    type(self).__name__, repr(size)
                )
            )
        if parameters is not None and not isinstance(parameters, dict):
            raise TypeError(
                "{} 'parameters' must be of type dict or None, not {}".format(
                    type(self).__name__, repr(parameters)
                )
            )
        if typestr is not None and not isinstance(typestr, str):
            raise TypeError(
                "{} 'typestr' must be of type string or None, not {}".format(
                    type(self).__name__, repr(typestr)
                )
            )
        self._content = content
        self._size = size
        self._parameters = parameters
        self._typestr = typestr

    @property
    def content(self):
        return self._content

    @property
    def size(self):
        return self._size

    def _str(self, indent, compact):
        if self._typestr is not None:
            out = [self._typestr]

        elif self.parameter("__array__") == "string":
            out = [f"string[{self._size}]"]

        elif self.parameter("__array__") == "bytestring":
            out = [f"bytes[{self._size}]"]

        else:
            params = self._str_parameters()

            if params is None:
                out = [str(self._size), " * ", *self._content._str(indent, compact)]
            else:
                out = [
                    "[",
                    str(self._size),
                    " * ",
                    *self._content._str(indent, compact),
                ] + [", ", params, "]"]

        return [self._str_categorical_begin(), *out] + [self._str_categorical_end()]

    def __repr__(self):
        args = [repr(self._content), repr(self._size), *self._repr_args()]
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def _is_equal_to(self, other, all_parameters: bool) -> bool:
        compare_parameters = (
            parameters_are_equal if all_parameters else type_parameters_equal
        )
        return (
            isinstance(other, type(self))
            and compare_parameters(self._parameters, other._parameters)
            and (self._size == other._size)
            and self._content._is_equal_to(other._content, all_parameters)
        )
