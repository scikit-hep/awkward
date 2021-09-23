# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


class Reducer:
    def __init__(self, name, *args, **params):
        self.__name__ = name

    def _return_dtype(self, given_dtype):
        # FIXME: if defined _MSC_VER || defined __i386__
        if given_dtype == np.int8 or given_dtype == np.int16 or given_dtype == np.int32:
            return np.int64

        if (
            given_dtype == np.uint8
            or given_dtype == np.uint16
            or given_dtype == np.uint32
        ):
            return np.uint64

        return given_dtype

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
        result = ak._v2.index.Index64.empty(outlength, array.nplike, dtype=np.int64)
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
        # reducer.preferred_dtype is float64
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
        # reducer.preferred_dtype is float64
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
        # reducer.preferred_dtype is float64
        dtype = array.dtype
        kernel_name = "awkward_reduce_sum"
        if dtype == np.bool_:
            kernel_name = kernel_name + "_bool"
        result = ak._v2.contents.NumpyArray(
            array.nplike.empty(outlength, dtype=self._return_dtype(array.dtype))
        )
        array._handle_error(
            array.nplike[
                kernel_name,
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
        # reducer.preferred_dtype is int64
        result = ak._v2.contents.NumpyArray(
            array.nplike.empty(outlength, dtype=self._return_dtype(array.dtype))
        )
        kernel_name = "awkward_reduce_prod"
        if array.dtype == np.bool_:
            kernel_name = kernel_name + "_bool"
        array._handle_error(
            array.nplike[
                kernel_name,
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
        # reducer.preferred_dtype is boolean
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
        # reducer.preferred_dtype is boolean
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
        dtype = array.dtype
        result = ak._v2.contents.NumpyArray(array.nplike.empty(outlength, dtype))

        if array.dtype == np.bool_:
            array._handle_error(
                array.nplike[
                    "awkward_reduce_prod_bool",
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
        else:
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
                    np.inf,  # FIXME: set initial
                )
            )
        return ak._v2.contents.NumpyArray(result)

    def _max(self, array, parents, outlength):
        dtype = array.dtype
        result = ak._v2.contents.NumpyArray(array.nplike.empty(outlength, dtype))

        if array.dtype == np.bool_:
            array._handle_error(
                array.nplike[
                    "awkward_reduce_sum_bool",
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
        else:
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
                    -np.inf,  # FIXME: set initial
                )
            )
        return ak._v2.contents.NumpyArray(result)

    def _apply(self, array, parents, outlength):
        do = "_" + self.__name__  # f"_{self.__name__}"
        if hasattr(self, do):
            func = getattr(self, do)
            if callable(func):
                return func(array, parents, outlength)
