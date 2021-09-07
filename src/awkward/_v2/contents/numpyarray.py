# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import awkward as ak
from awkward._v2._slicing import NestedIndexError
from awkward._v2.contents.content import Content
from awkward._v2.forms.numpyform import NumpyForm

np = ak.nplike.NumpyMetadata.instance()


class NumpyArray(Content):
    def __init__(self, data, identifier=None, parameters=None, nplike=None):
        self._nplike = ak.nplike.of(data) if nplike is None else nplike
        self._data = self._nplike.asarray(data)

        if (
            self._data.dtype not in ak._v2.types.numpytype._dtype_to_primitive
            and not isinstance(self._data.dtype.type, (np.datetime64, np.timedelta64))
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
    def data(self):
        return self._data

    @property
    def shape(self):
        return self._data.shape

    @property
    def inner_shape(self):
        return self._data.shape[1:]

    @property
    def strides(self):
        return self._data.strides

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def nplike(self):
        return self._nplike

    @property
    def nonvirtual_nplike(self):
        return self._nplike

    @property
    def ptr(self):
        return self._data.ctypes.data

    Form = NumpyForm

    @property
    def form(self):
        return self.Form(
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

        extra = self._repr_extra(indent + "    ")
        arraystr_lines = self._nplike.array_str(self._data, max_line_width=30).split(
            "\n"
        )
        if len(extra) != 0 or len(arraystr_lines) > 1:
            arraystr_lines = self._nplike.array_str(
                self._data, max_line_width=max(80 - len(indent) - 4, 40)
            ).split("\n")
            if len(arraystr_lines) > 5:
                arraystr_lines = arraystr_lines[:2] + [" ..."] + arraystr_lines[-2:]
            out.append(">")
            out.extend(extra)
            out.append("\n" + indent + "    ")
            out.append(("\n" + indent + "    ").join(arraystr_lines))
            out.append("\n" + indent + "</NumpyArray>")
        else:
            out.append(">")
            out.append(arraystr_lines[0])
            out.append("</NumpyArray>")

        out.append(post)
        return "".join(out)

    def toRegularArray(self):
        if len(self._data.shape) == 1:
            return self
        else:
            return ak._v2.contents.RegularArray(
                NumpyArray(
                    self._data.reshape((-1,) + self._data.shape[2:]),
                    None,
                    None,
                    nplike=self._nplike,
                ).toRegularArray(),
                self._data.shape[1],
                self._data.shape[0],
                self._identifier,
                self._parameters,
            )

    def maybe_to_nplike(self, nplike):
        return nplike.asarray(self._data)

    def _getitem_nothing(self):
        tmp = self._data[0:0]
        return NumpyArray(
            tmp.reshape((0,) + tmp.shape[2:]),
            self._range_identifier(0, 0),
            None,
            nplike=self._nplike,
        )

    def _getitem_at(self, where):
        try:
            out = self._data[where]
        except IndexError as err:
            raise NestedIndexError(self, where, str(err))

        if hasattr(out, "shape") and len(out.shape) != 0:
            return NumpyArray(out, None, None, nplike=self._nplike)
        else:
            return out

    def _getitem_range(self, where):
        start, stop, step = where.indices(len(self))
        assert step == 1

        try:
            out = self._data[where]
        except IndexError as err:
            raise NestedIndexError(self, where, str(err))

        return NumpyArray(
            out,
            self._range_identifier(start, stop),
            self._parameters,
            nplike=self._nplike,
        )

    def _getitem_field(self, where, only_fields=()):
        raise NestedIndexError(self, where, "not an array of records")

    def _getitem_fields(self, where, only_fields=()):
        if len(where) == 0:
            return self._getitem_range(slice(0, 0))
        raise NestedIndexError(self, where, "not an array of records")

    def _carry(self, carry, allow_lazy, exception):
        assert isinstance(carry, ak._v2.index.Index)
        try:
            nextdata = self._data[carry.data]
        except IndexError as err:
            if issubclass(exception, NestedIndexError):
                raise exception(self, carry.data, str(err))
            else:
                raise exception(str(err))

        return NumpyArray(
            nextdata,
            self._carry_identifier(carry, exception),
            self._parameters,
            nplike=self._nplike,
        )

    def _getitem_next_jagged(self, slicestarts, slicestops, slicecontent, tail):
        if self._data.ndim == 1:
            raise NestedIndexError(
                self,
                ak._v2.contents.ListArray(slicestarts, slicestops, slicecontent),
                "too many jagged slice dimensions for array",
            )
        else:
            next = self.toRegularArray()
            return next._getitem_next_jagged(
                slicestarts, slicestops, slicecontent, tail
            )

    def _getitem_next(self, head, tail, advanced):
        nplike = self._nplike

        if head == ():
            return self

        elif isinstance(head, int):
            where = (slice(None), head) + tail

            try:
                out = self._data[where]
            except IndexError as err:
                raise NestedIndexError(self, (head,) + tail, str(err))

            if hasattr(out, "shape") and len(out.shape) != 0:
                return NumpyArray(out, None, None, nplike=nplike)
            else:
                return out

        elif isinstance(head, slice) or head is np.newaxis or head is Ellipsis:
            where = (slice(None), head) + tail
            try:
                out = self._data[where]
            except IndexError as err:
                raise NestedIndexError(self, (head,) + tail, str(err))
            out2 = NumpyArray(out, None, self._parameters, nplike=nplike)
            return out2
        elif ak._util.isstr(head):
            return self._getitem_next_field(head, tail, advanced)

        elif isinstance(head, list):
            return self._getitem_next_fields(head, tail, advanced)

        elif isinstance(head, ak._v2.index.Index64):
            if advanced is None:
                where = (slice(None), head.data) + tail
            else:
                where = (nplike.asarray(advanced.data), head.data) + tail

            try:
                out = self._data[where]
            except IndexError as err:
                raise NestedIndexError(self, (head,) + tail, str(err))

            return NumpyArray(out, None, self._parameters, nplike=nplike)

        elif isinstance(head, ak._v2.contents.ListOffsetArray):
            where = (slice(None), head) + tail
            try:
                out = self._data[where]
            except IndexError as err:
                raise NestedIndexError(self, (head,) + tail, str(err))
            out2 = NumpyArray(out, None, self._parameters, nplike=nplike)
            return out2

        elif isinstance(head, ak._v2.contents.IndexedOptionArray):
            next = self.toRegularArray()
            return next._getitem_next_missing(head, tail, advanced)

        else:
            raise AssertionError(repr(head))

    def _localindex(self, axis, depth):
        posaxis = self._axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self._localindex_axis0()
        elif len(self.shape) <= 1:
            raise np.AxisError(self, "'axis' out of range for localindex")
        else:
            return self.toRegularArray()._localindex(posaxis, depth)

    def _combinations(self, n, replacement, recordlookup, parameters, axis, depth):
        posaxis = self._axis_wrap_if_negative(axis)
        if posaxis == depth:
            return self._combinations_axis0(n, replacement, recordlookup, parameters)
        elif len(self.shape) <= 1:
            raise np.AxisError("'axis' out of range for combinations")
        else:
            return self.toRegularArray()._combinations(
                n, replacement, recordlookup, parameters, posaxis, depth
            )
