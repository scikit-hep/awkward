# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


class Index(object):
    _dtype_to_form = {
        np.dtype(np.int8): "i8",
        np.dtype(np.uint8): "u8",
        np.dtype(np.int32): "i32",
        np.dtype(np.uint32): "u32",
        np.dtype(np.int64): "i64",
    }

    def __init__(self, data):
        self._nplike = ak.nplike.of(data)

        self._data = self._nplike.asarray(data, order="C")
        if len(self._data.shape) != 1:
            raise TypeError("Index data must be one-dimensional")

        self._dtype = self._data.dtype
        if self._dtype not in self._dtype_to_form:
            raise TypeError("Index data must be int8, uint8, int32, uint32, int64")

    @classmethod
    def zeros(cls, length, nplike, dtype):
        return Index(nplike.zeros(length, dtype=dtype))

    @classmethod
    def empty(cls, length, nplike, dtype):
        return Index(nplike.empty(length, dtype=dtype))

    @property
    def data(self):
        return self._data

    @property
    def nplike(self):
        return self._nplike

    @property
    def dtype(self):
        return self._dtype

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<Index T="]
        out.append(repr(self._dtype_to_form[self._data.dtype]))
        out.append(" length=")
        out.append(repr(str(len(self._data))))
        out.append(" at=")
        out.append(repr(hex(self._data.ctypes.data)))
        out.append(">")
        out.append(self._nplike.array_str(self._data, max_line_width=30))
        out.append("</Index>")
        out.append(post)
        return "".join(out)

    def form(self):
        return self._dtype_to_form[self._data.dtype]

    def __getitem__(self, where):
        return self._data[where]

    def __setitem__(self, where, what):
        self._data[where] = what

    def to64(self):
        return Index(self._data.astype(np.int64))

    def iscontiguous(self):
        return self._data.strides == (self._data.itemsize,)

    def __copy__(self):
        return Index(self._data.copy())

    def convert_to(self, nplike):
        return Index(nplike.asarray(self._data))
