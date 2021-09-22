# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


class Reducer:
    def __init__(self, name: str, *args, **params):
        self.__name__ = name

    def _argmin(self, array, parents, outlength):
        result = ak._v2.index.Index64.empty(outlength, array.nplike, dtype=np.int64)
        array._handle_error(
            array.nplike[
                "awkward_reduce_argmin",
                result.dtype.type,
                array.data.dtype.type,
                parents.dtype.type,
            ](
                result.to(array.nplike),
                array.data,
                parents.to(array.nplike),
                len(parents),
                outlength,
            )
        )
        return ak._v2.contents.NumpyArray(result)

    def _argmax(self, array, parents, outlength):
        result = ak._v2.index.Index64.empty(outlength, array.nplike)
        array._handle_error(
            array.nplike[
                "awkward_reduce_argmax",
                result.dtype.type,
                array._data.dtype.type,
                parents.dtype.type,
            ](
                result.to(array.nplike),
                array._data,
                parents.to(array.nplike),
                len(parents),
                outlength,
            )
        )
        return ak._v2.contents.NumpyArray(result)

    def _count(self, array, parents, outlength):
        result = ak._v2.index.Index64.empty(outlength, array.nplike)
        array._handle_error(
            array.nplike[
                "awkward_reduce_count_64", result.dtype.type, parents.dtype.type
            ](
                result.to(array.nplike),
                parents.to(array.nplike),
                len(parents),
                outlength,
            )
        )
        return ak._v2.contents.NumpyArray(result)

    def _count_nonzero(self, array, parents, outlength):
        result = ak._v2.index.Index64.empty(outlength, array.nplike)
        array._handle_error(
            array.nplike[
                "awkward_reduce_countnonzero",
                result.dtype.type,
                array._data.dtype.type,
                parents.dtype.type,
            ](
                result.to(array.nplike),
                array._data,
                parents.to(array.nplike),
                len(parents),
                outlength,
            )
        )
        return ak._v2.contents.NumpyArray(result)

    def _sum(self, array, parents, outlength):
        result = ak._v2.contents.NumpyArray(array.nplike.empty(outlength))
        array._handle_error(
            array.nplike[
                "awkward_reduce_sum",
                result.dtype.type,
                array.dtype.type,
                parents.dtype.type,
            ](
                result._data,
                array._data,
                parents.to(array.nplike),
                len(parents),
                outlength,
            )
        )
        return ak._v2.contents.NumpyArray(result)

    def _prod(self, array, parents, outlength):
        result = ak._v2.contents.NumpyArray(
            array.nplike.empty(outlength, dtype=array.dtype)
        )
        array._handle_error(
            array.nplike[
                "awkward_reduce_prod",
                result.dtype.type,
                array._data.dtype.type,
                parents.dtype.type,
            ](
                result._data,
                array._data,
                parents.to(array.nplike),
                len(parents),
                outlength,
            )
        )
        return ak._v2.contents.NumpyArray(result)

    def _any(self, array, parents, outlength):
        result = ak._v2.contents.NumpyArray(array.nplike.empty(outlength, dtype=bool))
        array._handle_error(
            array.nplike[
                "awkward_reduce_sum_bool",
                result.dtype.type,
                array._data.dtype.type,
                parents.dtype.type,
            ](
                result._data,
                array._data,
                parents.to(array.nplike),
                len(parents),
                outlength,
            )
        )
        return ak._v2.contents.NumpyArray(result)

    def _all(self, array, parents, outlength):
        result = ak._v2.contents.NumpyArray(
            array.nplike.empty(outlength, dtype=np.bool_)
        )
        array._handle_error(
            array.nplike[
                "awkward_reduce_prod_bool",
                result.dtype.type,
                array._data.dtype.type,
                parents.dtype.type,
            ](
                result._data,
                array._data,
                parents.to(array.nplike),
                len(parents),
                outlength,
            )
        )
        return ak._v2.contents.NumpyArray(result)

    def _min(self, array, parents, outlength):
        result = ak._v2.contents.NumpyArray(
            array.nplike.empty(outlength, dtype=array.dtype)
        )
        array._handle_error(
            array.nplike[
                "awkward_reduce_min",
                result.dtype.type,
                array.dtype.type,
                parents.dtype.type,
            ](
                result._data,
                array._data,
                parents.to(array.nplike),
                len(parents),
                outlength,
                True,
            )
        )
        return ak._v2.contents.NumpyArray(result)

    def _max(self, array, parents, outlength):
        result = ak._v2.contents.NumpyArray(
            array.nplike.empty(outlength, dtype=array.dtype)
        )
        array._handle_error(
            array.nplike[
                "awkward_reduce_max",
                result.dtype.type,
                array.dtype.type,
                parents.dtype.type,
            ](
                result._data,
                array._data,
                parents.to(array.nplike),
                len(parents),
                outlength,
                True,
            )
        )
        return ak._v2.contents.NumpyArray(result)

    def _apply(self, array, parents, outlength):
        do = f"_{self.__name__}"
        if hasattr(self, do) and callable(func := getattr(self, do)):
            return func(array, parents, outlength)
