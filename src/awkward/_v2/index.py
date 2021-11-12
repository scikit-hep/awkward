# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak

np = ak.nplike.NumpyMetadata.instance()

_dtype_to_form = {
    np.dtype(np.int8): "i8",
    np.dtype(np.uint8): "u8",
    np.dtype(np.int32): "i32",
    np.dtype(np.uint32): "u32",
    np.dtype(np.int64): "i64",
}


class Index(object):
    _expected_dtype = None

    def __init__(self, data, metadata=None):
        self._nplike = ak.nplike.of(data)
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
        return Index(nplike.empty(length, dtype=dtype))

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

    def __len__(self):
        return len(self._data)

    def to(self, nplike):
        return nplike.asarray(self._data)

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
                        "<metadata key={0}>{1}</metadata>\n".format(repr(k), repr(v))
                        + indent
                        + "    "
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
