# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak
from awkward._v2.contents.content import Content

np = ak.nplike.NumpyMetadata.instance()


class NumpyArray(Content):
    def __init__(self, data):
        self._nplike = ak.nplike.of(data)
        assert isinstance(self._nplike, (ak.nplike.Numpy, ak.nplike.Cupy))
        self._data = data

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<NumpyArray>\n"]
        if hasattr(self._data, "shape") and len(self._data.shape) != 0:
            out.append(indent + "    <dtype>" + str(self._data.dtype) + "</dtype>\n")
            out.append(indent + "    <shape>" + repr(self._data.shape) + "</shape>\n")
            out.append(indent + "    <array>" + "\n    " + indent)
            out.append(
                self._nplike.array_str(self._data, max_line_width=30).replace(
                    "\n", "\n" + indent + "    "
                )
            )
            out.append("\n" + indent + "    </array>")
        # FIXME print option for lists pretty print?
        else:
            out.append(indent + "    <length>" + str(len(self._data)) + "</length>\n")
            out.append(indent + "    <array>" + "\n    " + indent)
            out.append(str(self._data).replace("\n", "\n" + indent + "    "))
            out.append("\n" + indent + "    </array>")
        out.append("\n" + indent)
        out.append("</NumpyArray>")
        out.append(post)
        return "".join(out)

    def __len__(self):
        return len(self._data)

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
