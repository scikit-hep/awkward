# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import copy

from awkward._nplikes import to_nplike
from awkward._nplikes.array_like import ArrayLike
from awkward._nplikes.cupy import Cupy
from awkward._nplikes.dispatch import nplike_of_obj
from awkward._nplikes.jax import Jax
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpy_like import NumpyLike, NumpyMetadata
from awkward._nplikes.shape import ShapeItem
from awkward._nplikes.typetracer import TypeTracer
from awkward._nplikes.virtual import VirtualArray, materialize_if_virtual
from awkward._slicing import normalize_slice
from awkward._typing import Any, DType, Final, Self, cast

np: Final = NumpyMetadata.instance()
numpy: Final = Numpy.instance()


_dtype_to_form: Final[dict[DType, str]] = {
    np.dtype(np.int8): "i8",
    np.dtype(np.uint8): "u8",
    np.dtype(np.int32): "i32",
    np.dtype(np.uint32): "u32",
    np.dtype(np.int64): "i64",
}

_form_to_dtype: Final[dict[str, DType]] = {v: k for k, v in _dtype_to_form.items()}


def _form_to_zero_length(form: str) -> Index:
    try:
        dtype = _form_to_dtype[form]
    except KeyError:
        raise AssertionError(f"unrecognized Index form: {form!r}") from None
    return Index(numpy.zeros(0, dtype=dtype))


class Index:
    _expected_dtype: DType | None = None

    def __init__(
        self,
        data,
        *,
        metadata: dict | None = None,
        nplike: NumpyLike | None = None,
    ):
        assert not isinstance(data, Index)
        if nplike is None:
            self._nplike = cast(
                "NumpyLike[ArrayLike]", nplike_of_obj(data, default=Numpy.instance())
            )
        else:
            self._nplike = nplike

        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("Index metadata must be None or a dict")
        self._metadata = metadata
        # We don't care about F, C (it's one dimensional), but we do need
        # the array to be contiguous. This should _not_ return a copy if already
        self._data = self._nplike.ascontiguousarray(
            self._nplike.asarray(data, dtype=self._expected_dtype)
        )

        if len(self._data.shape) != 1:
            raise TypeError("Index data must be one-dimensional")

        if np.issubdtype(self._data.dtype, np.longlong):
            assert np.dtype(np.longlong).itemsize == 8, (
                "longlong is always 64-bit, right?"
            )

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
    def zeros(
        cls, length: ShapeItem, nplike: NumpyLike, dtype: DType | None = None
    ) -> Index:
        if dtype is None:
            dtype = cls._expected_dtype
        return Index(nplike.zeros(length, dtype=dtype), nplike=nplike)

    @classmethod
    def empty(
        cls, length: ShapeItem, nplike: NumpyLike, dtype: DType | None = None
    ) -> Index:
        if dtype is None:
            dtype = cls._expected_dtype
        return Index(nplike.empty(length, dtype=dtype), nplike=nplike)

    @property
    def data(self) -> ArrayLike:
        return self._data

    @property
    def nplike(self) -> NumpyLike:
        return self._nplike

    @property
    def dtype(self) -> DType:
        return self._data.dtype

    @property
    def metadata(self) -> dict:
        if self._metadata is None:
            self._metadata = {}
        return self._metadata

    @property
    def ptr(self):
        if isinstance(self._nplike, Numpy):
            return self._data.ctypes.data
        elif isinstance(self._nplike, Cupy):
            return self._data.data.ptr
        elif isinstance(self._nplike, TypeTracer):
            return 0
        else:
            raise NotImplementedError(
                f"this function hasn't been implemented for the {type(self._nplike).__name__} backend"
            )

    @property
    def length(self) -> ShapeItem:
        return self._data.shape[0]

    def forget_length(self) -> Self:
        tt = TypeTracer.instance()
        if isinstance(self._nplike, type(tt)):
            data = self._data
        else:
            data = self.raw(tt)

        assert hasattr(data, "forget_length")
        return type(self)(data.forget_length(), metadata=self._metadata, nplike=tt)

    def raw(self, nplike: NumpyLike) -> ArrayLike:
        return to_nplike(self.data, nplike, from_nplike=self._nplike)

    def materialize(self) -> Index:
        (out,) = materialize_if_virtual(self._data)
        return Index(out, metadata=self.metadata, nplike=self._nplike)

    @property
    def is_all_materialized(self) -> bool:
        buffer = self._data
        if isinstance(buffer, VirtualArray):
            return buffer.is_materialized
        return True

    @property
    def is_any_materialized(self) -> bool:
        buffer = self._data
        if isinstance(buffer, VirtualArray):
            return buffer.is_materialized
        return True

    def __len__(self) -> int:
        return int(self.length)

    @property
    def __cuda_array_interface__(self):
        return self._data.__cuda_array_interface__  # type: ignore[attr-defined]

    @property
    def __array_interface__(self):
        return self._data.__array_interface__  # type: ignore[attr-defined]

    def __dlpack_device__(self) -> tuple[int, int]:
        return self._data.__dlpack_device__()  # type: ignore[attr-defined]

    def __dlpack__(self, stream: Any = None) -> Any:
        if stream is None:
            return self._data.__dlpack__()  # type: ignore[attr-defined]
        else:
            return self._data.__dlpack__(stream=stream)  # type: ignore[attr-defined]

    def __repr__(self) -> str:
        return self._repr("", "", "")

    def _repr(self, indent: str, pre: str, post: str) -> str:
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
    def form(self) -> str:
        return _dtype_to_form[self._data.dtype]

    def __getitem__(self, where):
        if isinstance(where, slice):
            where = normalize_slice(where, nplike=self.nplike)

        out = self._data[where]

        if hasattr(out, "shape") and len(out.shape) != 0:
            return Index(out, metadata=self.metadata, nplike=self._nplike)
        elif (Jax.is_own_array(out) or Cupy.is_own_array(out)) and len(out.shape) == 0:
            return out.item()
        else:
            return out

    def __setitem__(self, where, what):
        self._data[where] = what

    def to64(self) -> Index:
        return Index(self._nplike.astype(self._data, dtype=np.int64))

    def __copy__(self) -> Self:
        return type(self)(self._data, metadata=self._metadata, nplike=self._nplike)

    def __deepcopy__(self, memo: dict) -> Self:
        return type(self)(
            copy.deepcopy(self._data, memo),
            metadata=copy.deepcopy(self._metadata, memo),
            nplike=self._nplike,
        )

    def _nbytes_part(self) -> ShapeItem:
        return self.data.nbytes

    def to_nplike(self, nplike: NumpyLike) -> Self:
        return type(self)(self.raw(nplike), metadata=self.metadata, nplike=nplike)

    def is_equal_to(
        self, other: Any, index_dtype: bool = True, numpyarray: bool = True
    ) -> bool:
        if index_dtype:
            return (
                not self._nplike.known_data
                or self._nplike.array_equal(self.data, other.data)
            ) and self._data.dtype == other.data.dtype

        else:
            return self._nplike.array_equal(self.data, other.data)

    def _touch_data(self):
        if hasattr(self._data, "touch_data"):
            self._data.touch_data()

    def _touch_shape(self):
        if hasattr(self._data, "touch_shape"):
            self._data.touch_shape()


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
