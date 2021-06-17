# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import
from awkward._v2.types.type import Type


class ArrayType(object):
    def __init__(self, content, length):
        if not isinstance(content, Type):
            raise TypeError(
                "{0} all 'contents' must be Type subclasses, not {1}".format(
                    type(self).__name__, repr(content)
                )
            )
        if not isinstance(length, int) or length < 0:
            raise ValueError(
                "{0} 'size' must be of a positive integer, not {1}".format(
                    type(self).__name__, repr(length)
                )
            )
        self._content = content
        self._length = length

    @property
    def content(self):
        return self._content

    @property
    def length(self):
        return self._length

    def __str__(self):
        return "{0} * {1}".format(repr(self._length), self._content.typestr)

    def __repr__(self):
        return "ArrayType({0}, {1})".format(repr(self._content), repr(self._length))
