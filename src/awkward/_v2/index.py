import awkward as ak

class Index:

    def __init__(self, data):
        self._nplike = ak.nplike.of(data)
        self._data = self._nplike.asarray(data, order=”C”)
        self._T = self._data.dtype
        if self._data == None:
            self._len = 0
        else:
            self._len = len(self.data)
        if self._T not in (self._nplike.int8, self._nplike.uint8, self._nplike.int32, self._nplike.uint32, self._nplike.int64, object):
            raise TypeError(...)

    #the default for this is float64 so maybe type should also be mentioned
    def __zeros__(length, nplike):
        return nplike.zeros(length)

    #the default for this is float64 so maybe type should also be mentioned
    def __empty__(length, nplike):
        return nplike.empty(length)

    def __data__(self):
        return self._data

    def __nplike__(self):
        return  self._nplike

    def  __len__(self):
        return self._len

    #array_str needs to be included in nplike
    def __repr__(self):
        return self._nplike.array_str(self._data)

    def __form__(self):
        type = str(self._T)
        return type[0] + type.split('t')[1]

    def __getitem__(self, index):
        return self._data[index]

    def __setitem__(self,index, value):
        self._data[index] = value

    #this will return an empty array if start and stop are not in range
    def __getitem_range__(self, start, stop):
        if  start < stop and stop < self._len:
            return self._data[start:stop]
        else:
            print("Illegal start:stop for this length")

    def __to64__(self):
        self._data = self._data.astype('int64')

    def __iscontiguous__(self):
        return self._data.strides[0] == self._data.itemsize

    def __copy__(self):
        return self._data.copy()


    def __covert__(self):
        cp = ak.nplike.Cupy()
        np = ak.nplike.Numpy()
        if str(self._nplike) == 'Numpy':
            self._data = self._nplike.asarray(self._data)
            self._nplike = cp.ak.nplike.instance()
        else:
            self._data = self._nplike.asarray(self._data)
            self._nplike = np.ak.nplike.instance()


