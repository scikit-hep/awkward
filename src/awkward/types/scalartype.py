# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import sys

import awkward as ak


class ScalarType:
    def __init__(self, content):
        if not isinstance(content, ak.types.Type):
            raise TypeError(
                "{} all 'contents' must be Type subclasses, not {}".format(
                    type(self).__name__, repr(content)
                )
            )
        self._content = content

    @property
    def content(self):
        return self._content

    def __str__(self):
        return "".join(self._str("", True))

    def show(self, stream=sys.stdout):
        stream.write("".join([*self._str("", False), "\n"]))

    def _str(self, indent, compact):
        return self._content._str(indent, compact)

    def __repr__(self):
        return f"{type(self).__name__}({self._content!r})"

    def __eq__(self, other):
        if isinstance(other, ScalarType):
            return self._content == other._content
        else:
            return False
