import awkward as ak

np = ak.nplikes.NumpyMetadata.instance()


class Transformer:
    needs_position = False
    maintain_none_position = True
    through_record = False
    preferred_dtype = None

    @property
    def name(self):
        raise ak._errors.wrap_error(NotImplementedError)

    @property
    def highlevel_function(self):
        return getattr(ak.operations, self.name)

    def return_dtype(self, given_dtype):
        if given_dtype in (np.bool_, np.int8, np.int16, np.int32):
            return np.int32 if ak._util.win or ak._util.bits32 else np.int64

        if given_dtype in (np.uint8, np.uint16, np.uint32):
            return np.uint32 if ak._util.win or ak._util.bits32 else np.uint64

        return given_dtype

    def maybe_double_length(self, type, length):
        return 2 * length if type in (np.complex128, np.complex64) else length

    def maybe_other_type(self, dtype):
        type = np.int64 if dtype.kind.upper() == "M" else dtype.type
        if dtype == np.complex128:
            type = np.float64
        if dtype == np.complex64:
            type = np.float32
        return type

    def apply(
        self,
        array: "ak.contents.NumpyArray",
        offsets: "ak.index.Index",
        parents: "ak.index.Index",
    ):
        """Apply the transform function to the given array.

        Args:
            array: array to transform
            offsets: the indices of the start and stop positions for contiguous sublists in parents
            parents: the identity of the sublist associated with each value in array

        Returns:

        """
        raise NotImplementedError


class Sort(Transformer):
    through_record = True
    maintain_none_position = False

    def __init__(self, ascending: bool, stable: bool):
        self._ascending = ascending
        self._stable = stable

    @property
    def ascending(self):
        return self._ascending

    @property
    def stable(self):
        return self._stable

    def apply(self, array, offsets, parents):
        dtype = self.maybe_other_type(array.dtype)
        out = array.nplike.empty(array.length, dtype)
        assert offsets.nplike is array.nplike
        array._handle_error(
            array.nplike[  # noqa: E231
                "awkward_sort",
                dtype,
                dtype,
                offsets.dtype.type,
            ](
                out,
                array.data,
                array.shape[0],
                offsets.data,
                offsets.length,
                parents.length,
                self._ascending,
                self._stable,
            )
        )
        return ak.contents.NumpyArray(
            array.nplike.asarray(out, array.dtype), None, None, array.nplike
        )


class ArgSort(Transformer):
    needs_position = True
    maintain_none_position = False
    preferred_dtype = np.int64

    def __init__(self, ascending: bool, stable: bool):
        self._ascending = ascending
        self._stable = stable

    @property
    def ascending(self):
        return self._ascending

    @property
    def stable(self):
        return self._stable

    def apply(self, array, offsets, parents):
        dtype = self.maybe_other_type(array.dtype)
        nextcarry = ak.index.Index64.empty(len(array), array.nplike)
        assert offsets.nplike is array.nplike
        array._handle_error(
            array.nplike[  # noqa: E231
                "awkward_argsort",
                nextcarry.dtype.type,
                dtype,
                offsets.dtype.type,
            ](
                nextcarry.data,
                array.data,
                len(array),
                offsets.data,
                offsets.length,
                self._ascending,
                self._stable,
            )
        )
        return ak.contents.NumpyArray(
            array.nplike.asarray(nextcarry, nextcarry.dtype), None, None, array.nplike
        )


class CumSum(Transformer):
    def apply(self, array, offsets, parents):
        assert isinstance(array, ak.contents.NumpyArray)
        if array.dtype.kind == "M":
            raise ak._errors.wrap_error(
                ValueError(f"cannot compute the sum (ak.sum) of {array.dtype!r}")
            )
        else:
            dtype = self.maybe_other_type(array.dtype)
        result = array.nplike.empty(
            self.maybe_double_length(array.dtype.type, array.length),
            dtype=self.return_dtype(dtype),
        )

        if array.dtype.type in (np.complex128, np.complex64):
            raise NotImplementedError
        else:
            assert parents.nplike is array.nplike
            array._handle_error(
                array.nplike[
                    "awkward_transform_cumsum",
                    result.dtype.type,
                    np.int64 if array.dtype.kind == "m" else array.dtype.type,
                    offsets.dtype.type,
                ](
                    result,
                    array.data,
                    offsets.data,
                    offsets.length,
                )
            )
        if array.dtype.kind == "m":
            return ak.contents.NumpyArray(array.nplike.asarray(result, array.dtype))
        elif array.dtype.type in (np.complex128, np.complex64):
            return ak.contents.NumpyArray(result.view(array.dtype))
        else:
            return ak.contents.NumpyArray(result)
