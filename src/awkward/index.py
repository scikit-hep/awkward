# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import copy

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()

_dtype_to_form = {
    np.dtype(np.int8): "i8",
    np.dtype(np.uint8): "u8",
    np.dtype(np.int32): "i32",
    np.dtype(np.uint32): "u32",
    np.dtype(np.int64): "i64",
}


class Index:
    _expected_dtype = None

    def __init__(self, data, metadata=None, nplike=None):
        if nplike is None:
            nplike = ak.nplike.of(data)
        self._nplike = nplike
        if metadata is not None and not isinstance(metadata, dict):
            raise ak._v2._util.error(TypeError("Index metadata must be None or a dict"))
        self._metadata = metadata
        self._data = self._nplike.index_nplike.asarray(
            data, dtype=self._expected_dtype, order="C"
        )
        if len(self._data.shape) != 1:
            raise ak._v2._util.error(TypeError("Index data must be one-dimensional"))

        if issubclass(self._data.dtype.type, np.longlong):
            assert (
                np.dtype(np.longlong).itemsize == 8
            ), "longlong is always 64-bit, right?"

            self._data = self._data.view(np.int64)

        if self._expected_dtype is None:
            if self._data.dtype == np.dtype(np.int8):
                self.__class__ = Index8
            elif self._data.dtype == np.dtype(np.uint8):
                self.__class__ = IndexU8
            elif self._data.dtype == np.dtype(np.int32):
                self.__class__ = Index32
            elif self._data.dtype == np.dtype(np.uint32):
                self.__class__ = IndexU32
            elif self._data.dtype == np.dtype(np.int64):
                self.__class__ = Index64
            else:
                raise ak._v2._util.error(
                    TypeError(
                        "Index data must be int8, uint8, int32, uint32, int64, not "
                        + repr(self._data.dtype)
                    )
                )
        else:
            if self._data.dtype != self._expected_dtype:
                # self._data = self._data.astype(self._expected_dtype)   # copy/convert
                raise ak._v2._util.error(
                    NotImplementedError(
                        "while developing, we want to catch these errors"
                    )
                )

    @classmethod
    def zeros(cls, length, nplike, dtype=None):
        if dtype is None:
            dtype = cls._expected_dtype
        return Index(nplike.index_nplike.zeros(length, dtype=dtype), nplike=nplike)

    @classmethod
    def empty(cls, length, nplike, dtype=None):
        if dtype is None:
            dtype = cls._expected_dtype
        return Index(nplike.index_nplike.empty(length, dtype=dtype), nplike=nplike)

    @property
    def data(self):
        return self._data

    @property
    def nplike(self):
        return self._nplike

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def metadata(self):
        if self._metadata is None:
            self._metadata = {}
        return self._metadata

    @property
    def ptr(self):
        return self._data.ctypes.data

    @property
    def length(self):
        return self._data.shape[0]

    def forget_length(self):
        tt = ak._v2._typetracer.TypeTracer.instance()
        if isinstance(self._nplike, type(tt)):
            data = self._data
        else:
            data = self.raw(tt)
        return type(self)(data.forget_length(), self._metadata, tt)

    def raw(self, nplike):
        return self.nplike.index_nplike.raw(self.data, nplike.index_nplike)

    def __len__(self):
        return self.length

    def __array__(self, *args, **kwargs):
        return self._nplike.index_nplike.asarray(self._data, *args, **kwargs)

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<Index dtype="]
        out.append(repr(str(self.dtype)))
        out.append(" len=")
        out.append(repr(str(self._data.shape[0])))

        arraystr_lines = self._nplike.index_nplike.array_str(
            self._data, max_line_width=30
        ).split("\n")
        if len(arraystr_lines) > 1 or self._metadata is not None:
            arraystr_lines = self._nplike.index_nplike.array_str(
                self._data, max_line_width=max(80 - len(indent) - 4, 40)
            ).split("\n")
            if len(arraystr_lines) > 5:
                arraystr_lines = arraystr_lines[:2] + [" ..."] + arraystr_lines[-2:]
            out.append(">\n" + indent + "    ")
            if self._metadata is not None:
                for k, v in self._metadata.items():
                    out.append(
                        f"<metadata key={k!r}>{v!r}</metadata>\n" + indent + "    "
                    )
            out.append(("\n" + indent + "    ").join(arraystr_lines))
            out.append("\n" + indent + "</Index>")
        else:
            if len(arraystr_lines) > 5:
                arraystr_lines = arraystr_lines[:2] + [" ..."] + arraystr_lines[-2:]
            out.append(">")
            out.append(arraystr_lines[0])
            out.append("</Index>")

        out.append(post)
        return "".join(out)

    @property
    def form(self):
        return _dtype_to_form[self._data.dtype]

    def __getitem__(self, where):
        out = self._data[where]

        if hasattr(out, "shape") and len(out.shape) != 0:
            return Index(out, metadata=self.metadata, nplike=self.nplike)
        elif (ak.nplike.is_jax_buffer(out) or ak.nplike.is_cupy_buffer(out)) and len(
            out.shape
        ) == 0:
            return out.item()
        else:
            return out

    def __setitem__(self, where, what):
        self._data[where] = what

    def to64(self):
        return Index(self._data.astype(np.int64))

    def __copy__(self):
        return type(self)(self._data, self._metadata, self._nplike)

    def __deepcopy__(self, memo):
        return type(self)(
            copy.deepcopy(self._data, memo),
            copy.deepcopy(self._metadata, memo),
            self._nplike,
        )

    def _nbytes_part(self):
        return self.data.nbytes

    def to_backend(self, backend):
        if self.nplike is ak._v2._util.regularize_backend(backend):
            return self
        else:
            return self._to_nplike(ak._v2._util.regularize_backend(backend))

    def _to_nplike(self, nplike):
        # if isinstance(nplike, ak.nplike.Jax):
        #     print("YES OFFICER, this nplike right here")
        return Index(self.raw(nplike), metadata=self.metadata, nplike=nplike)

    def layout_equal(self, other, index_dtype=True, numpyarray=True):
        if index_dtype:
            return (
                self.nplike.index_nplike.array_equal(self.data, other.data)
                and self._data.dtype == other.data.dtype
            )
        else:
            return self.nplike.index_nplike.array_equal(self.data, other.data)


class Index8(Index):
    _expected_dtype = np.dtype(np.int8)


class IndexU8(Index):
    _expected_dtype = np.dtype(np.uint8)


class Index32(Index):
    _expected_dtype = np.dtype(np.int32)


class IndexU32(Index):
    _expected_dtype = np.dtype(np.uint32)


class Index64(Index):
    _expected_dtype = np.dtype(np.int64)
