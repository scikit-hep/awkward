# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak
from awkward._v2.contents.content import Content

np = ak.nplike.NumpyMetadata.instance()


class NumpyArray(Content):
    def __init__(self, data):
        self._nplike = ak.nplike.of(data)
        self._data = self._nplike.asarray(data)

    @property
    def nplike(self):
        return self._nplike

    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<NumpyArray dtype="]
        out.append(repr(str(self.dtype)))
        if len(self._data.shape) == 1:
            out.append(" len=")
            out.append(repr(str(len(self))))
        else:
            out.append(" shape=")
            out.append(repr(str(self._data.shape)))

        arraystr_lines = self._nplike.array_str(self._data, max_line_width=30).split(
            "\n"
        )
        if len(arraystr_lines) > 1:  # if or this array has an Identifier
            arraystr_lines = self._nplike.array_str(
                self._data, max_line_width=max(80 - len(indent) - 4, 40)
            ).split("\n")
            if len(arraystr_lines) > 5:
                arraystr_lines = arraystr_lines[:2] + [" ..."] + arraystr_lines[-2:]
            out.append(">\n" + indent + "    ")
            out.append(("\n" + indent + "    ").join(arraystr_lines))
            out.append("\n" + indent + "</NumpyArray>")
        else:
            if len(arraystr_lines) > 5:
                arraystr_lines = arraystr_lines[:2] + [" ..."] + arraystr_lines[-2:]
            out.append(">")
            out.append(arraystr_lines[0])
            out.append("</NumpyArray>")

        out.append(post)
        return "".join(out)

    def _getitem_at(self, where):
        out = self._data[where]
        if hasattr(out, "shape") and len(out.shape) != 0:
            return NumpyArray(out)
        else:
            return out

    def _getitem_range(self, where):
        return NumpyArray(self._data[where])

    def _getitem_field(self, where):
        raise IndexError("field " + repr(where) + " not found")

    def _getitem_fields(self, where):
        raise IndexError("fields " + repr(where) + " not found")
