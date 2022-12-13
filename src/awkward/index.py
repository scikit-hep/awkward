# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
from __future__ import annotations

import copy

import awkward as ak
from awkward.typing import Self

np = ak._nplikes.NumpyMetadata.instance()
numpy = ak._nplikes.Numpy.instance()


_dtype_to_form = {
    np.dtype(np.int8): "i8",
    np.dtype(np.uint8): "u8",
    np.dtype(np.int32): "i32",
    np.dtype(np.uint32): "u32",
    np.dtype(np.int64): "i64",
}


def _form_to_zero_length(form):
    if form == "i8":
        return Index8(numpy.zeros(0, dtype=np.int8))
    elif form == "u8":
        return IndexU8(numpy.zeros(0, dtype=np.uint8))
    elif form == "i32":
        return Index32(numpy.zeros(0, dtype=np.int32))
    elif form == "u32":
        return IndexU32(numpy.zeros(0, dtype=np.uint32))
    elif form == "i64":
        return Index64(numpy.zeros(0, dtype=np.int64))
    else:
        raise ak._errors.wrap_error(
            AssertionError(f"unrecognized Index form: {form!r}")
        )


class Index:
    _expected_dtype = None

    def __init__(self, data, *, metadata=None, nplike=None):
        if nplike is None:
            nplike = ak._nplikes.nplike_of(data)
        self._nplike = nplike
        if metadata is not None and not isinstance(metadata, dict):
            raise ak._errors.wrap_error(
                TypeError("Index metadata must be None or a dict")
            )
        self._metadata = metadata
        self._data = self._nplike.asarray(data, dtype=self._expected_dtype, order="C")
        if len(self._data.shape) != 1:
            raise ak._errors.wrap_error(TypeError("Index data must be one-dimensional"))

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
                raise ak._errors.wrap_error(
                    TypeError(
                        "Index data must be int8, uint8, int32, uint32, int64, not "
                        + repr(self._data.dtype)
                    )
                )
        else:
            if self._data.dtype != self._expected_dtype:
                # self._data = self._data.astype(self._expected_dtype)   # copy/convert
                raise ak._errors.wrap_error(
                    NotImplementedError(
                        "while developing, we want to catch these errors"
                    )
                )

    @classmethod
    def zeros(cls, length, nplike, dtype=None):
        if dtype is None:
            dtype = cls._expected_dtype
        return Index(nplike.zeros(length, dtype=dtype), nplike=nplike)

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
        tt = ak._typetracer.TypeTracer.instance()
        if isinstance(self._nplike, type(tt)):
            data = self._data
        else:
            data = self.raw(tt)
        return type(self)(data.forget_length(), metadata=self._metadata, nplike=tt)

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
            return Index(out, metadata=self.metadata, nplike=self._nplike)
        elif (
            ak._nplikes.Jax.is_own_array(out) or ak._nplikes.Cupy.is_own_array(out)
        ) and len(out.shape) == 0:
            return out.item()
        else:
            return out

    def __setitem__(self, where, what):
        self._data[where] = what

    def to64(self):
        return Index(self._data.astype(np.int64))

    def __copy__(self):
        return type(self)(self._data, metadata=self._metadata, nplike=self._nplike)

    def __deepcopy__(self, memo):
        return type(self)(
            copy.deepcopy(self._data, memo),
            metadata=copy.deepcopy(self._metadata, memo),
            nplike=self._nplike,
        )

    def _nbytes_part(self):
        return self.data.nbytes

    def to_nplike(self, nplike: ak._nplikes.NumpyLike) -> Self:
        return type(self)(self.raw(nplike), metadata=self.metadata, nplike=nplike)

    def is_equal_to(self, other, index_dtype=True, numpyarray=True):
        if index_dtype:
            return (
                self.nplike.array_equal(self.data, other.data)
                and self._data.dtype == other.data.dtype
            )
        else:
            return self.nplike.array_equal(self.data, other.data)


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
