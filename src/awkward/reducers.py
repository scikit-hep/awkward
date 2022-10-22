# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak

np = ak.nplikes.NumpyMetadata.instance()


class Reducer:
    needs_position = False

    @classmethod
    def highlevel_function(cls):
        return getattr(ak.operations, cls.name)

    @classmethod
    def return_dtype(cls, given_dtype):
        if given_dtype in (np.bool_, np.int8, np.int16, np.int32):
            return np.int32 if ak._util.win or ak._util.bits32 else np.int64

        if given_dtype in (np.uint8, np.uint16, np.uint32):
            return np.uint32 if ak._util.win or ak._util.bits32 else np.uint64

        return given_dtype

    @classmethod
    def maybe_double_length(cls, type, length):
        return 2 * length if type in (np.complex128, np.complex64) else length

    @classmethod
    def maybe_other_type(cls, dtype):
        type = np.int64 if dtype.kind.upper() == "M" else dtype.type
        if dtype == np.complex128:
            type = np.float64
        if dtype == np.complex64:
            type = np.float32
        return type

    def apply(self, array, parents, outlength):
        raise ak._errors.wrap_error(NotImplementedError)

    def apply_parts(self, arrays):
        raise ak._errors.wrap_error(NotImplementedError)


class ArgMin(Reducer):
    name = "argmin"
    needs_position = True
    preferred_dtype = np.int64

    @classmethod
    def return_dtype(cls, given_dtype):
        return np.int64

    def apply(self, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)
        dtype = self.maybe_other_type(array.dtype)
        result = array.nplike.empty(outlength, dtype=np.int64)
        if array.dtype.type in (np.complex128, np.complex64):
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_argmin_complex",
                    result.dtype.type,
                    dtype,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )
        else:
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_argmin",
                    result.dtype.type,
                    dtype,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )
        return ak.contents.NumpyArray(result)


class ArgMax(Reducer):
    name = "argmax"
    needs_position = True
    preferred_dtype = np.int64

    @classmethod
    def return_dtype(cls, given_dtype):
        return np.int64

    def apply(self, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)
        dtype = self.maybe_other_type(array.dtype)
        result = array.nplike.empty(outlength, dtype=np.int64)
        if array.dtype.type in (np.complex128, np.complex64):
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_argmax_complex",
                    result.dtype.type,
                    dtype,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )
        else:
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_argmax",
                    result.dtype.type,
                    dtype,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )
        return ak.contents.NumpyArray(result)


class Count(Reducer):
    name = "count"
    preferred_dtype = np.int64

    @classmethod
    def return_dtype(cls, given_dtype):
        return np.int64

    def apply(self, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)
        result = array.nplike.empty(outlength, dtype=np.int64)
        assert parents.nplike is array.nplike
        array._handle_error(
            array.nplike[
                "awkward_reduce_count_64", result.dtype.type, parents.dtype.type
            ](
                result,
                parents.data,
                parents.length,
                outlength,
            )
        )
        return ak.contents.NumpyArray(result)


class CountNonzero(Reducer):
    name = "count_nonzero"
    preferred_dtype = np.float64

    @classmethod
    def return_dtype(cls, given_dtype):
        return np.int64

    def apply(self, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)
        dtype = np.dtype(np.int64) if array.dtype.kind.upper() == "M" else array.dtype
        result = array.nplike.empty(outlength, dtype=np.int64)
        if array.dtype.type in (np.complex128, np.complex64):
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_countnonzero_complex",
                    result.dtype.type,
                    np.float64 if array.dtype.type == np.complex128 else np.float32,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )
        else:
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_countnonzero",
                    result.dtype.type,
                    dtype.type,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )
        return ak.contents.NumpyArray(result)


class Sum(Reducer):
    name = "sum"
    preferred_dtype = np.float64

    def apply(self, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)
        if array.dtype.kind == "M":
            raise ak._errors.wrap_error(
                ValueError(f"cannot compute the sum (ak.sum) of {array.dtype!r}")
            )
        else:
            dtype = self.maybe_other_type(array.dtype)
        result = array.nplike.empty(
            self.maybe_double_length(array.dtype.type, outlength),
            dtype=self.return_dtype(dtype),
        )

        if array.dtype == np.bool_:
            if result.dtype in (np.int64, np.uint64):
                assert parents.nplike is array.nplike
                array._handle_error(
                    array.nplike[
                        "awkward_reduce_sum_int64_bool_64",
                        np.int64,
                        array.dtype.type,
                        parents.dtype.type,
                    ](
                        result,
                        array.data,
                        parents.data,
                        parents.length,
                        outlength,
                    )
                )
            elif result.dtype in (np.int32, np.uint32):
                assert parents.nplike is array.nplike
                array._handle_error(
                    array.nplike[
                        "awkward_reduce_sum_int32_bool_64",
                        np.int32,
                        array.dtype.type,
                        parents.dtype.type,
                    ](
                        result,
                        array.data,
                        parents.data,
                        parents.length,
                        outlength,
                    )
                )
            else:
                raise ak._errors.wrap_error(NotImplementedError)
        elif array.dtype.type in (np.complex128, np.complex64):
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_sum_complex",
                    np.float64 if array.dtype.type == np.complex128 else np.float32,
                    np.float64 if array.dtype.type == np.complex128 else np.float32,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )
        else:
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_sum",
                    result.dtype.type,
                    np.int64 if array.dtype.kind == "m" else array.dtype.type,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )

        if array.dtype.kind == "m":
            return ak.contents.NumpyArray(array.nplike.asarray(result, array.dtype))
        elif array.dtype.type in (np.complex128, np.complex64):
            return ak.contents.NumpyArray(result.view(array.dtype))
        else:
            return ak.contents.NumpyArray(result)


class Prod(Reducer):
    name = "prod"
    preferred_dtype = np.int64

    def apply(self, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)
        if array.dtype.kind.upper() == "M":
            raise ak._errors.wrap_error(
                ValueError(f"cannot compute the product (ak.prod) of {array.dtype!r}")
            )
        if array.dtype == np.bool_:
            result = array.nplike.empty(
                self.maybe_double_length(array.dtype.type, outlength),
                dtype=np.dtype(np.bool_),
            )
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_prod_bool",
                    result.dtype.type,
                    array.dtype.type,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )
            result = result.astype(self.return_dtype(array.dtype))
        elif array.dtype.type in (np.complex128, np.complex64):
            result = array.nplike.empty(
                self.maybe_double_length(array.dtype.type, outlength),
                dtype=self.return_dtype(array.dtype),
            )
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_prod_complex",
                    np.float64 if array.dtype.type == np.complex128 else np.float32,
                    np.float64 if array.dtype.type == np.complex128 else np.float32,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )
        else:
            result = array.nplike.empty(
                self.maybe_double_length(array.dtype.type, outlength),
                dtype=self.return_dtype(array.dtype),
            )
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_prod",
                    result.dtype.type,
                    array.dtype.type,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )
        if array.dtype.type in (np.complex128, np.complex64):
            return ak.contents.NumpyArray(result.view(array.dtype))
        else:
            return ak.contents.NumpyArray(result)


class Any(Reducer):
    name = "any"
    preferred_dtype = np.bool_

    @classmethod
    def return_dtype(cls, given_dtype):
        return np.bool_

    def apply(self, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)
        dtype = self.maybe_other_type(array.dtype)
        result = array.nplike.empty(outlength, dtype=np.bool_)
        if array.dtype.type in (np.complex128, np.complex64):
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_sum_bool_complex",
                    result.dtype.type,
                    dtype,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )
        else:
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_sum_bool",
                    result.dtype.type,
                    dtype,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )
        return ak.contents.NumpyArray(result)


class All(Reducer):
    name = "all"
    preferred_dtype = np.bool_

    @classmethod
    def return_dtype(cls, given_dtype):
        return np.bool_

    def apply(self, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)
        dtype = self.maybe_other_type(array.dtype)
        result = array.nplike.empty(outlength, dtype=np.bool_)
        if array.dtype.type in (np.complex128, np.complex64):
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_prod_bool_complex",
                    result.dtype.type,
                    dtype,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )
        else:
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_prod_bool",
                    result.dtype.type,
                    dtype,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )
        return ak.contents.NumpyArray(result)


class Min(Reducer):
    name = "min"
    preferred_dtype = np.float64

    def __init__(self, initial):
        self._initial = initial

    @property
    def initial(self):
        return self._initial

    @staticmethod
    def _min_initial(initial, type):
        if initial is None:
            if type in (
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ):
                return np.iinfo(type).max
            else:
                return np.inf

        return initial

    def apply(self, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)
        dtype = self.maybe_other_type(array.dtype)
        result = array.nplike.empty(
            self.maybe_double_length(array.dtype.type, outlength), dtype=dtype
        )
        if array.dtype == np.bool_:
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_prod_bool",
                    result.dtype.type,
                    array.dtype.type,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )
        elif array.dtype.type in (np.complex128, np.complex64):
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_min_complex",
                    result.dtype.type,
                    dtype,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                    self._min_initial(self.initial, dtype),
                )
            )
        else:
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_min",
                    result.dtype.type,
                    dtype,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                    self._min_initial(self.initial, dtype),
                )
            )
        if array.dtype.type in (np.complex128, np.complex64):
            return ak.contents.NumpyArray(
                array.nplike.array(result.view(array.dtype), array.dtype)
            )
        else:
            return ak.contents.NumpyArray(array.nplike.array(result, array.dtype))


class Max(Reducer):
    name = "max"
    preferred_dtype = np.float64

    def __init__(self, initial):
        self._initial = initial

    @property
    def initial(self):
        return self._initial

    @staticmethod
    def _max_initial(initial, type):
        if initial is None:
            if type in (
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ):
                return np.iinfo(type).min
            else:
                return -np.inf

        return initial

    def apply(self, array, parents, outlength):
        assert isinstance(array, ak.contents.NumpyArray)
        dtype = self.maybe_other_type(array.dtype)
        result = array.nplike.empty(
            self.maybe_double_length(array.dtype.type, outlength), dtype=dtype
        )
        if array.dtype == np.bool_:
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_sum_bool",
                    result.dtype.type,
                    array.dtype.type,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                )
            )
        elif array.dtype.type in (np.complex128, np.complex64):
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_max_complex",
                    result.dtype.type,
                    dtype,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                    self._max_initial(self.initial, dtype),
                )
            )
        else:
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_reduce_max",
                    result.dtype.type,
                    dtype,
                    parents.dtype.type,
                ](
                    result,
                    array.data,
                    parents.data,
                    parents.length,
                    outlength,
                    self._max_initial(self.initial, dtype),
                )
            )
        if array.dtype.type in (np.complex128, np.complex64):
            return ak.contents.NumpyArray(
                array.nplike.array(result.view(array.dtype), array.dtype)
            )
        else:
            return ak.contents.NumpyArray(array.nplike.array(result, array.dtype))
