# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np

import awkward as ak
from awkward._typing import final


class LayoutBuilder:
    def _init(self, parameters):
        self._parameters = parameters

    @property
    def parameters(self):
        return self._parameters

    def __len__(self):
        raise AssertionError("missing implementation")

    def numbatype(self):
        raise AssertionError("missing implementation")

    def snapshot(self):
        raise AssertionError("missing implementation")

    @property
    def form(self):
        raise AssertionError("missing implementation")

    def clear(self):
        raise AssertionError("missing implementation")

    def is_valid(self, error: str):
        raise AssertionError("missing implementation")


@final
class Numpy(LayoutBuilder):
    def __init__(self, dtype, *, parameters=None, initial=1024, resize=8.0):
        self._data = ak.numba.GrowableBuffer(
            dtype=dtype, initial=initial, resize=resize
        )
        self._init(parameters)

    @classmethod
    def _from_buffer(cls, data):
        out = cls.__new__(cls)
        out._data = data
        out._parameters = None
        return out

    def __repr__(self):
        return f"ak.numba.lb.Numpy({self._data.dtype}, parameters={self._parameters})"

    def numbatype(self):
        import numba

        return ak._connect.numba.layoutbuilder.NumpyType(
            numba.from_dtype(self.dtype), numba.types.StringLiteral(self._parameters)
        )

    def __len__(self):
        return len(self._data)

    @property
    def dtype(self):
        return self._data.dtype

    def append(self, x):
        self._data.append(x)

    def extend(self, data):
        self._data.extend(data)

    @property
    def form(self):
        return ak.forms.NumpyForm(
            primitive=ak.types.numpytype.dtype_to_primitive(self._data.dtype),
            parameters=self._parameters,
        )

    def clear(self):
        self._data.clear()

    def is_valid(self, error: str):
        return True

    def snapshot(self) -> ak.contents.Content:
        return ak.contents.NumpyArray(
            self._data.snapshot(), parameters=self._parameters
        )


@final
class Empty(LayoutBuilder):
    def __init__(self):
        self._init(None)

    def __repr__(self):
        return "ak.numba.lb.Empty(parameters=None)"

    def numbatype(self):
        import numba

        return ak._connect.numba.layoutbuilder.EmptyType(
            numba.types.StringLiteral(None)
        )

    def __len__(self):
        return 0

    @property
    def form(self):
        return ak.forms.EmptyForm()

    def clear(self):
        pass

    def is_valid(self, error: str):
        return True

    def snapshot(self) -> ak.contents.Content:
        return ak.contents.EmptyArray()


@final
class ListOffset(LayoutBuilder):
    def __init__(self, dtype, content, *, parameters=None, initial=1024, resize=8.0):
        self._offsets = ak.numba.GrowableBuffer(
            dtype=np.dtype(dtype), initial=initial, resize=resize
        )
        self._offsets.append(0)
        self._content = content
        self._init(parameters)

    def __repr__(self):
        return f"ak.numba.lb.ListOffset({self._offsets.dtype}, {self._content}, parameters={self._parameters})"

    def numbatype(self):
        import numba

        return ak._connect.numba.layoutbuilder.ListOffsetType(
            numba.from_dtype(self.offsets.dtype),
            self.content,
            numba.types.StringLiteral(self._parameters),
        )

    @property
    def offsets(self):
        return self._offsets

    @property
    def content(self):
        return self._content

    @property
    def form(self):
        return ak.forms.ListOffsetForm(
            ak.index._dtype_to_form[self.offsets.dtype],
            self.content.form,
            parameters=self._parameters,
        )

    def begin_list(self):
        return self._content

    def end_list(self):
        self._offsets.append(len(self._content))

    def clear(self):
        self._offsets.clear()
        self._offsets.append(0)
        self._content.clear()

    def __len__(self):
        return self._offsets._length_pos[0] - 1

    def is_valid(self, error: str):
        if len(self._content) != self._offsets.last():
            error = f"ListOffset node{self._id} has content length {len(self._content)} but last offset {self._offsets.last()}"
            return False
        else:
            return self._content.is_valid(error)

    def snapshot(self) -> ak.contents.Content:
        content = self._content.snapshot()

        return ak.contents.listoffsetarray.ListOffsetArray(
            ak.index.Index(self._offsets.snapshot()),
            content,
            parameters=self._parameters,
        )


@final
class Regular(LayoutBuilder):
    def __init__(self, content, size, *, parameters=None):
        self._content = content
        self._size = size
        self._init(parameters)

        if size < 1:
            raise ValueError("unsupported feature: size must be at least 1")

    def __repr__(self):
        return f"ak.numba.lb.Regular({self._content}, {self._size}, parameters={self._parameters})"

    def numbatype(self):
        import numba

        return ak._connect.numba.layoutbuilder.RegularType(
            self.content,
            self.size,
            numba.types.StringLiteral(self._parameters),
        )

    @property
    def content(self):
        return self._content

    @property
    def size(self):
        return self._size

    @property
    def form(self):
        return ak.forms.RegularForm(
            self.content.form,
            self.size,
            parameters=self._parameters,
        )

    def begin_list(self):
        return self.content

    def end_list(self):
        pass

    def clear(self):
        self.content.clear()

    def __len__(self):
        return len(self.content) // self.size

    def is_valid(self, error: str):  # structure_valid
        if len(self.content) != len(self) * self.size:
            error = f"Regular node{self._id} has content length {len(self.content)}, but length {len(self)} and size {self.size}"
            return False
        else:
            return self.content.is_valid(error)

    def snapshot(self) -> ak.contents.Content:
        return ak.contents.RegularArray(
            self._content.snapshot(),
            self._size,
            len(self),
            parameters=self._parameters,
        )


@final
class IndexedOption(LayoutBuilder):
    def __init__(self, dtype, content, *, parameters=None, initial=1024, resize=8.0):
        self._last_valid = -1
        self._index = ak.numba.GrowableBuffer(
            dtype=dtype, initial=initial, resize=resize
        )
        self._content = content
        self._init(parameters)

    def __repr__(self):
        return f"ak.numba.lb.IndexedOption({self._index.dtype}, {self._content}, parameters={self._parameters})"

    def numbatype(self):
        import numba

        return ak._connect.numba.layoutbuilder.IndexedOptionType(
            numba.from_dtype(self.index.dtype),
            self.content,
            numba.types.StringLiteral(self._parameters),
        )

    @property
    def index(self):
        return self._index

    @property
    def content(self):
        return self._content

    @property
    def form(self):
        return ak.forms.IndexedOptionForm(
            ak.index._dtype_to_form[self.index.dtype],
            self.content.form,
            parameters=self._parameters,
        )

    def append_valid(self):
        self._last_valid = len(self._content)
        self._index.append(self._last_valid)
        return self._content

    def extend_valid(self, size):
        start = len(self._content)
        stop = start + size
        self._last_valid = stop - 1
        self._index.extend(list(range(start, stop)))
        return self._content

    def append_invalid(self):
        self._index.append(-1)

    def extend_invalid(self, size):
        self._index.extend([-1] * size)

    def clear(self):
        self._last_valid = -1
        self._index.clear()
        self._content.clear()

    def __len__(self):
        return self._index._length

    def is_valid(self, error: str):
        if len(self._content) != self._last_valid + 1:
            error = f"IndexedOption has content length {len(self._content)} but last valid index is {self._last_valid}"
            return False
        else:
            return self._content.is_valid(error)

    def snapshot(self) -> ak.contents.Content:
        return ak.contents.IndexedOptionArray(
            ak.index.Index64(self._index.snapshot()),
            self._content.snapshot(),
            parameters=self._parameters,
        )


@final
class ByteMasked(LayoutBuilder):
    def __init__(
        self,
        content,
        *,
        valid_when=True,
        parameters=None,
        initial=1024,
        resize=8.0,
    ):
        self._mask = ak.numba.GrowableBuffer(
            dtype=np.dtype(np.bool_), initial=initial, resize=resize
        )
        self._content = content
        self._valid_when = valid_when
        self._init(parameters)

    def __repr__(self):
        return f"ak.numba.lb.ByteMasked({self._content}, valid_when={self._valid_when}, parameters={self._parameters})"

    def numbatype(self):
        import numba

        return ak._connect.numba.layoutbuilder.ByteMaskedType(
            self.content,
            self.valid_when,
            numba.types.StringLiteral(self._parameters),
        )

    @property
    def content(self):
        return self._content

    @property
    def valid_when(self):
        return self._valid_when

    @property
    def form(self):
        return ak.forms.ByteMaskedForm(
            "i8",  # okay to hard-code "i8"
            self.content.form,
            self._valid_when,
            parameters=self._parameters,
        )

    def append_valid(self):
        self._mask.append(self._valid_when)
        return self._content

    def extend_valid(self, size):
        self._mask.extend([self._valid_when] * size)
        return self._content

    def append_invalid(self):
        self._mask.append(not self._valid_when)
        return self._content

    def extend_invalid(self, size):
        self._mask.extend([not self._valid_when] * size)
        return self._content

    def clear(self):
        self._mask.clear()
        self._content.clear()

    def __len__(self):
        return len(self._mask)

    def is_valid(self, error: str):
        if len(self._content) != len(self._mask):
            error = f"ByteMasked has content length {len(self._content)} but mask length {len(self._mask)}"
            return False
        else:
            return self._content.is_valid(error)

    def snapshot(self) -> ak.contents.Content:
        return ak.contents.ByteMaskedArray(
            ak.index.Index8(self._mask.snapshot()),
            self._content.snapshot(),
            valid_when=self._valid_when,
            parameters=self._parameters,
        )


@final
class BitMasked(LayoutBuilder):
    def __init__(
        self,
        dtype,  # mask must be "uint8"
        content,  # FIXME
        valid_when,
        lsb_order,
        *,
        parameters=None,
        initial=1024,
        resize=8.0,
    ):
        self._mask = ak.numba.GrowableBuffer(
            dtype=dtype, initial=initial, resize=resize
        )
        self._content = content
        self._valid_when = valid_when
        self._lsb_order = lsb_order
        self._current_byte_index = np.zeros((2,), dtype=np.uint8)
        self._mask.append(self._current_byte_index[0])
        if self._lsb_order:
            self._cast = np.array(
                [
                    np.uint8(1 << 0),
                    np.uint8(1 << 1),
                    np.uint8(1 << 2),
                    np.uint8(1 << 3),
                    np.uint8(1 << 4),
                    np.uint8(1 << 5),
                    np.uint8(1 << 6),
                    np.uint8(1 << 7),
                ]
            )
        else:
            self._cast = np.array(
                [
                    np.uint8(128 >> 0),
                    np.uint8(128 >> 1),
                    np.uint8(128 >> 2),
                    np.uint8(128 >> 3),
                    np.uint8(128 >> 4),
                    np.uint8(128 >> 5),
                    np.uint8(128 >> 6),
                    np.uint8(128 >> 7),
                ]
            )
        self._init(parameters)

    def __repr__(self):  # as constructor
        return f"ak.numba.lb.BitMasked({self._mask.dtype}, {self._content}, {self._valid_when}, {self._lsb_order}, parameters={self._parameters})"

    def numbatype(self):
        import numba

        return ak._connect.numba.layoutbuilder.BitMaskedType(
            numba.from_dtype(self._mask.dtype),
            self.content,
            self.valid_when,
            self.lsb_order,
            numba.types.StringLiteral(self._parameters),
        )

    @property
    def content(self):
        return self._content

    @property
    def valid_when(self):
        return self._valid_when

    @property
    def lsb_order(self):
        return self._lsb_order

    @property
    def form(self):
        return ak.forms.BitMaskedForm(
            ak.index._dtype_to_form[self._mask.dtype],
            self.content.form,
            self.valid_when,
            self.lsb_order,
            parameters=self._parameters,
        )

    def _append_begin(self):
        """
        Private helper function.
        """
        if self._current_byte_index[1] == 8:
            self._current_byte_index[0] = np.uint8(0)
            self._mask.append(self._current_byte_index[0])
            self._current_byte_index[1] = 0

    def _append_end(self):
        """
        Private helper function.
        """
        self._current_byte_index[1] += 1
        if self._valid_when:
            # 0 indicates null, 1 indicates valid
            self._mask._panels[-1][self._mask._length_pos[1] - 1] = (
                self._current_byte_index[0]
            )
        else:
            # 0 indicates valid, 1 indicates null
            self._mask._panels[-1][
                self._mask._length_pos[1] - 1
            ] = ~self._current_byte_index[0]

    def append_valid(self):
        self._append_begin()
        # current_byte_ and cast_: 0 indicates null, 1 indicates valid
        self._current_byte_index[0] |= self._cast[self._current_byte_index[1]]
        self._append_end()
        return self._content

    def extend_valid(self, size):
        # Just an interface; not actually faster than calling append many times.
        for _ in range(size):
            self.append_valid()
        return self._content

    def append_invalid(self):
        self._append_begin()
        # current_byte_ and cast_ default to null, no change
        self._append_end()
        return self._content

    def extend_invalid(self, size):
        # Just an interface; not actually faster than calling append many times.
        for _ in range(size):
            self.append_invalid()
        return self._content

    def clear(self):
        self._mask.clear()
        self._content.clear()

    def __len__(self):
        return (
            len(self._mask)
            if len(self._mask) == 0
            else (len(self._mask) - 1) * 8 + self._current_byte_index[1]
        )

    def is_valid(self, error: str):
        if len(self._content) != len(self):
            error = f"BitMasked has content length {len(self._content)} but bit mask length {len(self)}"
            return False
        else:
            return self._content.is_valid(error)

    def snapshot(self) -> ak.contents.Content:
        return ak.contents.BitMaskedArray(
            ak.index.Index(self._mask.snapshot()),
            self._content.snapshot(),
            valid_when=self._valid_when,
            length=len(self),
            lsb_order=self._lsb_order,
            parameters=self._parameters,
        )


@final
class Unmasked(LayoutBuilder):
    def __init__(self, content, *, parameters=None):
        self._content = content
        self._init(parameters)

    def __repr__(self):
        return f"ak.numba.lb.Unmasked({self._content}, parameters={self._parameters})"

    def numbatype(self):
        import numba

        return ak._connect.numba.layoutbuilder.UnmaskedType(
            self.content,
            numba.types.StringLiteral(self._parameters),
        )

    @property
    def content(self):
        return self._content

    @property
    def form(self):
        return ak.forms.UnmaskedForm(
            self.content.form,
            parameters=self._parameters,
        )

    def clear(self):
        self._content.clear()

    def __len__(self):
        return len(self._content)

    def is_valid(self, error: str):
        return self._content.is_valid(error)

    def snapshot(self) -> ak.contents.Content:
        return ak.contents.UnmaskedArray(
            self._content.snapshot(),
            parameters=self._parameters,
        )


@final
class Record(LayoutBuilder):
    def __init__(self, contents, fields, *, parameters=None):
        assert len(fields) != 0
        self._contents = tuple(contents)
        self._fields = tuple(fields)
        self._init(parameters)

        if len(self.contents) < 1:
            raise ValueError("unsupported feature: the contents must be nonempty")

    @property
    def contents(self):
        return self._contents

    @property
    def fields(self):
        return self._fields

    def __repr__(self):
        return f"ak.numba.lb.Record({self.contents}, {self.fields}, parameters={self._parameters})"

    @property
    def form(self):
        return ak.forms.RecordForm(
            [content.form for content in self.contents],
            self.fields,
            parameters=self._parameters,
        )

    def numbatype(self):
        import numba

        return ak._connect.numba.layoutbuilder.RecordType(
            tuple(
                ak._connect.numba.layoutbuilder.to_numbatype(it)
                for it in self._contents
            ),
            self.fields,
            numba.types.StringLiteral(self._parameters),
        )

    def content(self, name):
        return self._contents[self._fields.index(name)]

    def clear(self):
        for content in self._contents:
            content.clear()

    def __len__(self):
        return len(self._contents[0])

    def is_valid(self, error: str):
        length = -1
        for i, content in enumerate(self._contents):
            if length == -1:
                length = len(content)
            elif length != len(content):
                error = f"Record has field {self._fields[i]} length {len(content)} that differs from the first length {length}"
                return False
        for content in self._contents:
            if not content.is_valid(error):
                return False
        return True

    def snapshot(self) -> ak.contents.Content:
        contents = []
        for content in self._contents:
            contents.append(content.snapshot())

        return ak.contents.RecordArray(
            contents,
            self._fields,
            parameters=self._parameters,
        )


@final
class Tuple(LayoutBuilder):
    def __init__(self, contents, *, parameters=None):
        assert len(contents) != 0
        self._contents = tuple(contents)
        self._init(parameters)

        if len(self.contents) < 1:
            raise ValueError("unsupported feature: the contents must be nonempty")

    @property
    def contents(self):
        return self._contents

    @property
    def form(self):
        return ak.forms.RecordForm(
            [content.form for content in self.contents],
            fields=None,
            parameters=self._parameters,
        )

    def __repr__(self):
        return f"ak.numba.lb.Tuple({self.contents}, parameters={self._parameters})"

    def numbatype(self):
        import numba

        return ak._connect.numba.layoutbuilder.TupleType(
            self.contents,
            numba.types.StringLiteral(self._parameters),
        )

    def index(self, at):
        return self._contents[at]

    def clear(self):
        for content in self._contents:
            content.clear()

    def __len__(self):
        return len(self._contents[0])

    def is_valid(self, error: str):
        length = -1
        for index, content in enumerate(self._contents):
            if length == -1:
                length = len(content)
            elif length != len(content):
                error = f"Tuple has index {index} length {len(content)} that differs from the first length {length}"
                return False
        for content in self._contents:
            if not content.is_valid(error):
                return False
        return True

    def snapshot(self) -> ak.contents.Content:
        contents = []
        for content in self._contents:
            contents.append(content.snapshot())

        return ak.contents.RecordArray(
            contents,
            None,
            parameters=self._parameters,
        )


@final
class Union(LayoutBuilder):
    def __init__(
        self,
        tags_dtype,
        index_dtype,
        contents,
        *,
        parameters=None,
        initial=1024,
        resize=8.0,
    ):
        self._tags = ak.numba.GrowableBuffer(
            dtype=tags_dtype, initial=initial, resize=resize
        )
        self._index = ak.numba.GrowableBuffer(
            dtype=index_dtype, initial=initial, resize=resize
        )
        self._contents = tuple(contents)
        self._init(parameters)

        if len(self.contents) < 2:
            raise ValueError(
                "unsupported feature: the contents length must be at least 2"
            )

    @property
    def tags(self):
        return self._tags

    @property
    def index(self):
        return self._index

    @property
    def contents(self):
        return self._contents

    @property
    def form(self):
        return ak.forms.UnionForm(
            ak.index._dtype_to_form[self.tags.dtype],
            ak.index._dtype_to_form[self.index.dtype],
            [content.form for content in self.contents],
            parameters=self._parameters,
        )

    def __repr__(self):
        return f"ak.numba.lb.Union({self._tags.dtype}, {self._index.dtype}, {self.contents}, parameters={self._parameters})"

    def numbatype(self):
        import numba

        return ak._connect.numba.layoutbuilder.UnionType(
            numba.from_dtype(self._tags.dtype),
            numba.from_dtype(self._index.dtype),
            self.contents,
            numba.types.StringLiteral(self._parameters),
        )

    def append_content(self, tag):
        which_content = self._contents[tag]
        next_index = len(which_content)
        self._tags.append(tag)
        self._index.append(next_index)
        return which_content

    def clear(self):
        self._tags.clear()
        self._index.clear()
        for content in self._contents:
            content.clear()

    def __len__(self):
        return len(self._tags)

    def is_valid(self, error: str):
        for content in self._contents:
            if not content.is_valid(error):
                return False
        return True

    def snapshot(self) -> ak.contents.Content:
        contents = []
        for content in self._contents:
            contents.append(content.snapshot())

        return ak.contents.UnionArray(
            ak.index.Index8(self._tags.snapshot()),
            ak.index.Index64(self._index.snapshot()),
            contents,
            parameters=self._parameters,
        )
