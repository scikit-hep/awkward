# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


class Reducer:
    def __init__(self, reducername, nplike, *args, **params):
        self.__name__ = reducername
        self._nplike = nplike

    def _apply(self, array, parents, outlength):
        reducer_name = self.__name__
        reducer = getattr(self, reducer_name, None)
        return reducer(array, parents, outlength)

    def argmin(self, array, parents, outlength):
        nplike = self._nplike
        result = ak._v2.index.Index64.zeros(outlength, nplike)
        array._handle_error(
            nplike[
                "awkward_reduce_argmin",
                result.dtype.type,
                array.data.dtype.type,
                parents.dtype.type,
            ](
                result.to(nplike),
                array.data,
                parents.to(nplike),
                len(parents),
                outlength,
            )
        )
        return ak._v2.contents.NumpyArray(result)

    def argmax(self, array, parents, outlength):
        nplike = self._nplike
        result = ak._v2.index.Index64.zeros(outlength, nplike)
        array._handle_error(
            nplike[
                "awkward_reduce_argmax",
                result.dtype.type,
                array._data.dtype.type,
                parents.dtype.type,
            ](
                result.to(nplike),
                array._data,
                parents.to(nplike),
                len(parents),
                outlength,
            )
        )
        return ak._v2.contents.NumpyArray(result)

    def count(self, array, parents, outlength):
        nplike = self._nplike
        result = ak._v2.index.Index64.zeros(outlength, nplike)
        array._handle_error(
            nplike["awkward_reduce_count_64", result.dtype.type, parents.dtype.type](
                result.to(nplike),
                parents.to(nplike),
                len(parents),
                outlength,
            )
        )
        return ak._v2.contents.NumpyArray(result)

    def count_nonzero(self, array, parents, outlength):
        nplike = self._nplike
        result = ak._v2.index.Index64.zeros(outlength, nplike)
        array._handle_error(
            nplike[
                "awkward_reduce_countnonzero",
                result.dtype.type,
                array._data.dtype.type,
                parents.dtype.type,
            ](
                result.to(nplike),
                array._data,
                parents.to(nplike),
                len(parents),
                outlength,
            )
        )
        return ak._v2.contents.NumpyArray(result)

    def sum(self, array, parents, outlength):
        nplike = self._nplike
        result = ak._v2.contents.NumpyArray(nplike.zeros(outlength))
        array._handle_error(
            nplike[
                "awkward_reduce_sum",
                result.dtype.type,
                array.dtype.type,
                parents.dtype.type,
            ](
                result._data,
                array._data,
                parents.to(nplike),
                len(parents),
                outlength,
            )
        )
        return ak._v2.contents.NumpyArray(result)

    def prod(self, array, parents, outlength):
        nplike = self._nplike
        result = ak._v2.contents.NumpyArray(nplike.zeros(outlength))
        array._handle_error(
            nplike[
                "awkward_reduce_prod",
                result.dtype.type,
                array._data.dtype.type,
                parents.dtype.type,
            ](
                result._data,
                array._data,
                parents.to(nplike),
                len(parents),
                outlength,
            )
        )
        return ak._v2.contents.NumpyArray(result)

    def any(self, array, parents, outlength):
        nplike = self._nplike
        result = ak._v2.contents.NumpyArray(nplike.zeros(outlength, dtype=bool))
        array._handle_error(
            nplike[
                "awkward_reduce_sum_bool",
                result.dtype.type,
                array._data.dtype.type,
                parents.dtype.type,
            ](
                result._data,
                array._data,
                parents.to(nplike),
                len(parents),
                outlength,
            )
        )
        return ak._v2.contents.NumpyArray(result)

    def all(self, array, parents, outlength):
        nplike = self._nplike
        result = ak._v2.contents.NumpyArray(nplike.zeros(outlength, dtype=np.bool_))
        array._handle_error(
            nplike[
                "awkward_reduce_prod_bool",
                result.dtype.type,
                array._data.dtype.type,
                parents.dtype.type,
            ](
                result._data,
                array._data,
                parents.to(nplike),
                len(parents),
                outlength,
            )
        )
        return ak._v2.contents.NumpyArray(result)

    def min(self, array, parents, outlength):
        nplike = self._nplike
        result = ak._v2.contents.NumpyArray(nplike.zeros(outlength, dtype=array.dtype))
        array._handle_error(
            nplike[
                "awkward_reduce_min",
                result.dtype.type,
                array.dtype.type,
                parents.dtype.type,
            ](
                result._data,
                array._data,
                parents.to(nplike),
                len(parents),
                outlength,
                True,
            )
        )
        return ak._v2.contents.NumpyArray(result)

    def max(self, array, parents, outlength):
        nplike = self._nplike
        result = ak._v2.contents.NumpyArray(nplike.zeros(outlength, dtype=array.dtype))
        array._handle_error(
            nplike[
                "awkward_reduce_max",
                result.dtype.type,
                array.dtype.type,
                parents.dtype.type,
            ](
                result._data,
                array._data,
                parents.to(nplike),
                len(parents),
                outlength,
                True,
            )
        )
        return ak._v2.contents.NumpyArray(result)
