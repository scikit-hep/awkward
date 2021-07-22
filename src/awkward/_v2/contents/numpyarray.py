# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak
from awkward._v2.contents.content import Content

np = ak.nplike.NumpyMetadata.instance()


class NumpyArray(Content):
    def __init__(self, data, identifier=None, parameters=None):
        self._nplike = ak.nplike.of(data)
        self._data = self._nplike.asarray(data)

        if (
            self._data.dtype not in ak._v2.types.numpytype._dtype_to_primitive
            and not isinstance(self._data.dtype.type, (np.datetime64, np.timdelta64))
        ):
            raise TypeError(
                "{0} 'data' dtype {1} is not supported; must be one of {2}".format(
                    type(self).__name__,
                    repr(self._data.dtype),
                    ", ".join(
                        repr(x) for x in ak._v2.types.numpytype._dtype_to_primitive
                    ),
                )
            )
        if len(self._data.shape) == 0:
            raise TypeError(
                "{0} 'data' must be an array, not {1}".format(
                    type(self).__name__, repr(data)
                )
            )

        self._init(identifier, parameters)

    @property
    def nplike(self):
        return self._nplike

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self._data.shape

    @property
    def strides(self):
        return self._data.strides

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def form(self):
        return ak._v2.forms.NumpyForm(
            ak._v2.types.numpytype._dtype_to_primitive[self._data.dtype],
            self._data.shape[1:],
            has_identifier=self._identifier is not None,
            parameters=self._parameters,
            form_key=None,
        )

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<NumpyArray dtype="]
        out.append(repr(str(self.dtype)))
        if len(self._data.shape) == 1:
            out.append(" len=")
            out.append(repr(str(len(self))))
        else:
            out.append(" shape=")
            out.append(repr(str(self._data.shape)))

        arraystr_lines = self._nplike.array_str(self._data, max_line_width=30).split(
            "\n"
        )
        if len(arraystr_lines) > 1:  # if or this array has an Identifier
            arraystr_lines = self._nplike.array_str(
                self._data, max_line_width=max(80 - len(indent) - 4, 40)
            ).split("\n")
            if len(arraystr_lines) > 5:
                arraystr_lines = arraystr_lines[:2] + [" ..."] + arraystr_lines[-2:]
            out.append(">\n" + indent + "    ")
            out.append(("\n" + indent + "    ").join(arraystr_lines))
            out.append("\n" + indent + "</NumpyArray>")
        else:
            if len(arraystr_lines) > 5:
                arraystr_lines = arraystr_lines[:2] + [" ..."] + arraystr_lines[-2:]
            out.append(">")
            out.append(arraystr_lines[0])
            out.append("</NumpyArray>")

        out.append(post)
        return "".join(out)

    def _getitem_at(self, where):
        try:
            out = self._data[where]
        except IndexError as err:
            raise ak._v2.contents.content.NestedIndexError(self, where, str(err))

        if hasattr(out, "shape") and len(out.shape) != 0:
            return NumpyArray(out, None, None)
        else:
            return out

    def _getitem_range(self, where):
        start, stop, step = where.indices(len(self))
        assert step == 1
        return NumpyArray(
            self._data[where],
            self._range_identifier(start, stop),
            self._parameters,
        )

    def _getitem_field(self, where):
        raise IndexError("field " + repr(where) + " not found")

    def _getitem_fields(self, where):
        raise IndexError("fields " + repr(where) + " not found")

    def _carry(self, carry, allow_lazy):
        assert isinstance(carry, ak._v2.index.Index)

        try:
            nextdata = self._data[carry.data]
        except IndexError as err:
            raise ak._v2.contents.content.NestedIndexError(self, carry.data, str(err))

        return NumpyArray(nextdata, self._carry_identifier(carry), self._parameters)

    def _getitem_next(self, head, tail, advanced):
        if isinstance(head, int):
            raise NotImplementedError

        elif isinstance(head, slice):
            raise NotImplementedError

        elif ak._util.isstr(head):
            raise NotImplementedError

        elif isinstance(head, list):
            raise NotImplementedError

        elif head is np.newaxis:
            raise NotImplementedError

        elif head is Ellipsis:
            raise NotImplementedError

        elif isinstance(head, ak._v2.index.Index64):
            raise NotImplementedError

        elif isinstance(head, ak._v2.contents.ListOffsetArray):
            raise NotImplementedError

        elif isinstance(head, ak._v2.contents.IndexedOptionArray):
            raise NotImplementedError

        else:
            raise AssertionError(repr(head))
