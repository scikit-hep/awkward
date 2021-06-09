# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak
from awkward._v2.content import Content

np = ak.nplike.NumpyMetadata.instance()


class NumpyArray(Content):
    def __init__(self, data):
        self._nplike = ak.nplike.of(data)
        assert isinstance(self._nplike, (ak.nplike.Numpy, ak.nplike.Cupy))
        self._data = data

    def __repr__(self):
        return (
            "NumpyArray("
            + self._nplike.array_str(self._data, max_line_width=30)
            + ", "
            + repr(self._data.shape)
            + " )"
        )

    def __len__(self):
        return len(self._data)

    def _getitem_at(self, where):
        out = self._data[where]
        # FIXME if array is of any nplike, will instance fail when it's not specifically numpy?
        if isinstance(out, np.ndarray) and len(out.shape) != 0:
            return NumpyArray(out)
        else:
            return out

    def _getitem_range(self, where):
        return NumpyArray(self._data[where])

    def _getitem_field(self, where):
        raise IndexError("field " + repr(where) + " not found")

    def _getitem_fields(self, where):
        raise IndexError("fields " + repr(where) + " not found")
