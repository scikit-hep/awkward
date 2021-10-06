# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()


class Reducer(object):
    needs_position = False

    @classmethod
    def return_dtype(cls, given_dtype):
        if (
            given_dtype == np.bool_
            or given_dtype == np.int8
            or given_dtype == np.int16
            or given_dtype == np.int32
        ):
            if ak._util.win or ak._util.bits32:
                return np.int32
            else:
                return np.int64

        if (
            given_dtype == np.uint8
            or given_dtype == np.uint16
            or given_dtype == np.uint32
        ):
            if ak._util.win or ak._util.bits32:
                return np.uint32
            else:
                return np.uint64

        return given_dtype


class ArgMin(Reducer):
    name = "argmin"

    needs_position = True
    preferred_dtype = np.int64

    @classmethod
    def apply(cls, array, parents, outlength):
        result = ak._v2.index.Index64.empty(
            outlength, array.nplike, dtype=cls.preferred_dtype
        )
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


class ArgMax(Reducer):
    name = "argmax"

    needs_position = True
    preferred_dtype = np.int64

    @classmethod
    def apply(cls, array, parents, outlength):
        result = ak._v2.index.Index64.empty(
            outlength, array.nplike, dtype=cls.preferred_dtype
        )
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


class Count(Reducer):
    name = "count"
    preferred_dtype = np.float64

    @classmethod
    def apply(cls, array, parents, outlength):
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


class CountNonzero(Reducer):
    name = "count_nonzero"
    preferred_dtype = np.float64

    @classmethod
    def apply(cls, array, parents, outlength):
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


class Sum(Reducer):
    name = "sum"
    preferred_dtype = np.float64

    @classmethod
    def apply(cls, array, parents, outlength):
        result = ak._v2.contents.NumpyArray(
            array.nplike.empty(outlength, dtype=cls.return_dtype(array.dtype))
        )
        if array.dtype == np.bool_:
            if result.dtype == np.int64 or result.dtype == np.uint64:
                array._handle_error(
                    array.nplike[
                        "awkward_reduce_sum_int64_bool_64",
                        np.int64,
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
            elif result.dtype == np.int32 or result.dtype == np.uint32:
                array._handle_error(
                    array.nplike[
                        "awkward_reduce_sum_int32_bool_64",
                        np.int32,
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
                raise NotImplementedError
        else:
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


class Prod(Reducer):
    name = "prod"
    preferred_dtype = np.int64

    @classmethod
    def apply(cls, array, parents, outlength):
        result = ak._v2.contents.NumpyArray(
            array.nplike.empty(outlength, dtype=cls.return_dtype(array.dtype))
        )
        if array.dtype == np.bool_:
            array._handle_error(
                array.nplike[
                    "awkward_reduce_prod_bool",
                    array.dtype.type,
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
                    "awkward_reduce_prod",
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


class Any(Reducer):
    name = "any"
    preferred_dtype = np.bool_

    @classmethod
    def apply(cls, array, parents, outlength):
        result = ak._v2.contents.NumpyArray(
            array.nplike.empty(outlength, dtype=cls.preferred_dtype)
        )
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


class All(Reducer):
    name = "all"
    preferred_dtype = np.bool_

    @classmethod
    def apply(cls, array, parents, outlength):
        result = ak._v2.contents.NumpyArray(
            array.nplike.empty(outlength, dtype=cls.preferred_dtype)
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


class Min(Reducer):
    name = "min"
    preferred_dtype = np.float64
    initial = None

    def __init__(self, initial):
        type(self).initial = initial

    def __del__(self):
        type(self).initial = None

    @staticmethod
    def _min_initial(initial, type):
        if initial is None:
            if (
                type == np.int8
                or type == np.int16
                or type == np.int32
                or type == np.int64
                or type == np.uint8
                or type == np.uint16
                or type == np.uint32
                or type == np.uint64
            ):
                return np.iinfo(type).max
            else:
                return np.inf

        return initial

    @classmethod
    def apply(cls, array, parents, outlength):
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
                    cls._min_initial(cls.initial, dtype.type),
                )
            )

        return ak._v2.contents.NumpyArray(result)


class Max(Reducer):
    name = "max"
    preferred_dtype = np.float64
    initial = None

    def __init__(self, initial):
        type(self).initial = initial

    def __del__(self):
        type(self).initial = None

    @staticmethod
    def _max_initial(initial, type):
        if initial is None:
            if (
                type == np.int8
                or type == np.int16
                or type == np.int32
                or type == np.int64
                or type == np.uint8
                or type == np.uint16
                or type == np.uint32
                or type == np.uint64
            ):
                return np.iinfo(type).min
            else:
                return -np.inf

        return initial

    @classmethod
    def apply(cls, array, parents, outlength):
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
                    cls._max_initial(cls.initial, dtype.type),
                )
            )
        return ak._v2.contents.NumpyArray(result)
