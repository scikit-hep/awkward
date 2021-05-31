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

        self._T = self._data.dtype
        if self._T not in self._dtype_to_form:
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

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        # FIXME
        return self._nplike.array_str(self._data)

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
