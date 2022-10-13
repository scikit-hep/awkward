import awkward as ak

np = ak.nplikes.NumpyMetadata.instance()


class Transformer:
    needs_position = False
    through_record = False

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
