# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

import sys

import awkward as ak


class ScalarType:
    def __init__(self, content, behavior=None):
        if not isinstance(content, ak.types.Type):
            raise TypeError(
                "{} all 'contents' must be Type subclasses, not {}".format(
                    type(self).__name__, repr(content)
                )
            )
        self._content = content
        self._behavior = behavior

    @property
    def content(self):
        return self._content

    @property
    def behavior(self):
        return self._behavior

    def __str__(self):
        return "".join(self._str("", True))

    def show(self, stream=sys.stdout):
        stream.write("".join([*self._str("", False), "\n"]))

    def _str(self, indent, compact):
        return self._content._str(
            indent,
            compact,
            self._behavior,
        )

    def __repr__(self):
        return f"{type(self).__name__}({self._content!r}, {self._behavior!r})"

    def is_equal_to(self, other, *, all_parameters: bool = False) -> bool:
        return isinstance(other, type(self)) and self._content.is_equal_to(
            other._content, all_parameters=all_parameters
        )

    __eq__ = is_equal_to
