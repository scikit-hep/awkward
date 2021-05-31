# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import
import awkward as ak

np = ak.nplike.NumpyMetadata.instance()

class Identifier(Object):

    numrefs = 0

    def newref():
        numrefs += 1
        return numrefs

    def __init__(self, ref, filedloc, data):
        self._ref = ref
        self._fieldloc = filedloc
        self._data = data
        if len(self._data.shape) != 2:
            #FIXME
            raise TypeError("Data is not two dimensional.")
        self.nplike = ak.nplike.of(data)


    @classmethod
    def zeros(cls, ref, fieldloc, length, width, nplike, dtype):
        return nplike.zeros((length,width), dtype=dtype)???

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
        return  self._nplike

    @property
    def  __len__(self):
        return len(self._data)

    @property
    def  __width__(self):
        return self._data.shape[1]

    def to64(self):
        return Identifier(self._data.astype(np.int64))

    #is this retrieving just one dimension?
    def __getitem__(self, where):
        return self._data[where]

    def __copy__(self):
        return Identifier(self._data.copy())

    def __repr__(self):
        # FIXME
        return self._nplike.array_str(self._data)

    def convert_to(self, nplike):
        return Identifier(nplike.asarray(self._data))

    def referentially_equal (self):
        return self._ref is self._filedloc is self._data
