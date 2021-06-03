# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


class Identifier(object):
    _numrefs = 0

    @staticmethod
    def newref():
        out = Identifier._numrefs
        Identifier._numrefs += 1
        return out

    def __init__(self, ref, fieldloc, data):
        self._ref = ref
        self._fieldloc = fieldloc

        self._nplike = ak.nplike.of(data)

        self._data = self._nplike.asarray(data, order="C")
        if len(self._data.shape) != 2:
            raise TypeError("Identifer data must be 2-dimensional")

        # TypeError for unsupported types?
        # self._T = self._data.dtype
        # if self._T not in [np.dtype(np.int32), np.dtype(np.int64)]:
        #     raise TypeError("Identifier data must be int32, int64")

    @classmethod
    # cpp takes width, length?
    def zeros(cls, ref, fieldloc, length, width, nplike, dtype):
        return Identifier(ref, fieldloc, nplike.zeros((length, width), dtype=dtype))

    @classmethod
    def empty(cls, ref, fieldloc, length, width, nplike, dtype):
        return Identifier(ref, fieldloc, nplike.empty((length, width), dtype=dtype))

    @property
    def ref(self):
        return self._ref

    @property
    def filedloc(self):
        return self._fieldloc

    @property
    def data(self):
        return self._data

    @property
    def nplike(self):
        return self._nplike

    def __len__(self):
        return len(self._data)

    def width(self):
        return self._data.shape[1]

    def to64(self):
        return Identifier(self._ref, self._fieldloc, self._data.astype(np.int64))

    def __getitem__(self, where):
        return self._data[where]

    def __copy__(self):
        return Identifier(self._ref, self._fieldloc, self._data.copy())

    def __repr__(self):
        return self._nplike.array_str(self._data, max_line_width=30)

    def _repr(self, indent, pre, post):
        if self._data.dtype == np.dtype(np.int32):
            name = "Identities32"
        elif self._data.dtype == np.dtype(np.int64):
            name = "Identities64"
        else:
            name = "Unrecognized Identities"
        out = indent + pre + "<" + name + ' ref="' + str(self._ref) + '" fieldloc="'
        for elem in self._fieldloc:
            out += str(elem[0]) + ": " + str(elem[1]) + " "
        out = (
            out[:-1]
            + '" width="'
            + str(self._data.shape[1])
            + '" length="'
            + str(len(self._data))
            + '" at="'
            + str(hex(id(self._data)))
            + '"/>'
            + post
        )
        return out

    def convert_to(self, nplike):
        return Identifier(self._ref, self._fieldloc, nplike.asarray(self._data))

    def referentially_equal(self, other):
        return (
            self._ref == other._ref
            and self._fieldloc == other._fieldloc
            and self._data.ctypes.data == other._data.ctypes.data
            and self._data.shape == other._data.shape
            and self._data.strides == other._data.strides
            and self._data.dtype == other._data.dtype
        )
