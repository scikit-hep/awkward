# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

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
        self._metadata = metadata
        self._data = self._nplike.asarray(data, dtype=self._expected_dtype, order="C")
        if len(self._data.shape) != 1:
            raise TypeError("Index data must be one-dimensional")

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
                raise TypeError(
                    "Index data must be int8, uint8, int32, uint32, int64, not "
                    + repr(self._data.dtype)
                )
        else:
            if self._data.dtype != self._expected_dtype:
                # self._data = self._data.astype(self._expected_dtype)   # copy/convert
                raise NotImplementedError(
                    "while developing, we want to catch these errors"
                )

    @classmethod
    def zeros(cls, length, nplike, dtype=None):
        if dtype is None:
            dtype = cls._expected_dtype
        return Index(nplike.zeros(length, dtype=dtype))

    @classmethod
    def empty(cls, length, nplike, dtype=None):
        if dtype is None:
            dtype = cls._expected_dtype
        return Index(nplike.empty(length, dtype=dtype), nplike=nplike)

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
        return self.nplike.raw(self.data, nplike)

    def __len__(self):
        return self.length

    def __array__(self, *args, **kwargs):
        return self._nplike.asarray(self._data, *args, **kwargs)

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<Index dtype="]
        out.append(repr(str(self.dtype)))
        out.append(" len=")
        out.append(repr(str(self._data.shape[0])))

        arraystr_lines = self._nplike.array_str(self._data, max_line_width=30).split(
            "\n"
        )
        if len(arraystr_lines) > 1 or self._metadata is not None:
            arraystr_lines = self._nplike.array_str(
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
            return type(self)(out)
        elif type(out).__module__.startswith("cupy.") and len(out.shape) == 0:
            return out.item()
        else:
            return out

    def __setitem__(self, where, what):
        self._data[where] = what

    def to64(self):
        return Index(self._data.astype(np.int64))

    def __copy__(self):
        return Index(self._data.copy())

    def _nbytes_part(self):
        return self.data.nbytes

    def to_backend(self, backend):
        if self.nplike is ak._v2._util.regularize_backend(backend):
            return self
        else:
            return self._to_nplike(ak._v2._util.regularize_backend(backend))

    def _to_nplike(self, nplike):
        return Index(self.raw(nplike), metadata=self.metadata, nplike=nplike)


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
