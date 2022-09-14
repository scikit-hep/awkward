# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import copy
from collections.abc import Iterable

import awkward as ak
from awkward._v2.contents.content import Content

np = ak.nplike.NumpyMetadata.instance()


class Record:
    def __init__(self, array, at):
        if not isinstance(array, ak._v2.contents.recordarray.RecordArray):
            raise ak._v2._util.error(
                TypeError(f"Record 'array' must be a RecordArray, not {array!r}")
            )
        if not ak._v2._util.isint(at):
            raise ak._v2._util.error(
                TypeError(f"Record 'at' must be an integer, not {array!r}")
            )
        if at < 0 or at >= array.length:
            raise ak._v2._util.error(
                ValueError(
                    f"Record 'at' must be >= 0 and < len(array) == {array.length}, not {at}"
                )
            )
        else:
            self._array = array
            self._at = at

    @property
    def array(self):
        return self._array

    @property
    def at(self):
        return self._at

    @property
    def fields(self):
        return self._array.fields

    @property
    def is_tuple(self):
        return self._array.is_tuple

    @property
    def as_tuple(self):
        return Record(self._array.as_tuple, self._at)

    @property
    def contents(self):
        out = []
        for field in self._array.fields:
            out.append(self._array[field][self._at])
        return out

    def content(self, index_or_field):
        return self._array.content(index_or_field)[self._at]

    def __repr__(self):
        return self._repr("", "", "")

    def _repr(self, indent, pre, post):
        out = [indent, pre, "<Record at="]
        out.append(repr(str(self._at)))
        out.append(">\n")
        out.append(self._array._repr(indent + "    ", "<array>", "</array>\n"))
        out.append(indent)
        out.append("</Record>")
        out.append(post)
        return "".join(out)

    def validity_error(self, path="layout.array"):
        return self._array.validity_error(path)

    @property
    def parameters(self):
        return self._array.parameters

    def parameter(self, key):
        return self._array.parameter(key)

    def purelist_parameter(self, key):
        return self._array.purelist_parameter(key)

    @property
    def purelist_isregular(self):
        return self._array.purelist_isregular

    @property
    def purelist_depth(self):
        return 0

    @property
    def minmax_depth(self):
        mindepth, maxdepth = self._array.minmax_depth
        return mindepth - 1, maxdepth - 1

    @property
    def branch_depth(self):
        branch, depth = self._array.branch_depth
        return branch, depth - 1

    def axis_wrap_if_negative(self, axis):
        if axis == 0:
            raise ak._v2._util.error(
                np.AxisError("Record type at axis=0 is a scalar, not an array")
            )
        return self._array.axis_wrap_if_negative(axis)

    def __getitem__(self, where):
        with ak._v2._util.SlicingErrorContext(self, where):
            return self._getitem(where)

    def _getitem(self, where):
        if ak._util.isint(where):
            raise ak._v2._util.error(
                IndexError("scalar Record cannot be sliced by an integer")
            )

        elif isinstance(where, slice):
            raise ak._v2._util.error(
                IndexError("scalar Record cannot be sliced by a range slice (`:`)")
            )

        elif ak._util.isstr(where):
            return self._getitem_field(where)

        elif where is np.newaxis:
            raise ak._v2._util.error(
                IndexError("scalar Record cannot be sliced by np.newaxis (`None`)")
            )

        elif where is Ellipsis:
            raise ak._v2._util.error(
                IndexError("scalar Record cannot be sliced by an ellipsis (`...`)")
            )

        elif isinstance(where, tuple) and len(where) == 0:
            return self

        elif isinstance(where, tuple) and len(where) == 1:
            return self._getitem(where[0])

        elif isinstance(where, tuple) and ak._util.isstr(where[0]):
            return self._getitem_field(where[0])._getitem(where[1:])

        elif isinstance(where, ak.highlevel.Array):
            raise ak._v2._util.error(
                IndexError("scalar Record cannot be sliced by an array")
            )

        elif isinstance(where, ak.layout.Content):
            raise ak._v2._util.error(
                IndexError("scalar Record cannot be sliced by an array")
            )

        elif isinstance(where, Content):
            raise ak._v2._util.error(
                IndexError("scalar Record cannot be sliced by an array")
            )

        elif isinstance(where, Iterable) and all(ak._util.isstr(x) for x in where):
            return self._getitem_fields(where)

        elif isinstance(where, Iterable):
            raise ak._v2._util.error(
                IndexError("scalar Record cannot be sliced by an array")
            )

        else:
            raise ak._v2._util.error(
                TypeError(
                    "only field name (str) or names (non-tuple iterable of str) "
                    "are valid indices for slicing a scalar record, not\n\n    "
                    + repr(where)
                )
            )

    def _getitem_field(self, where):
        return self._array._getitem_field(where)._getitem_at(self._at)

    def _getitem_fields(self, where):
        return self._array._getitem_fields(where)._getitem_at(self._at)

    def packed(self):
        if self._array.length == 1:
            return Record(self._array.packed(), self._at)
        else:
            return Record(self._array[self._at : self._at + 1].packed(), 0)

    def to_list(self, behavior=None):
        return self._to_list(behavior, None)

    def _to_list(self, behavior, json_conversions):
        overloaded = (
            ak._v2._util.recordclass(self._array, behavior).__getitem__
            is not ak._v2.highlevel.Record.__getitem__
        )

        if overloaded:
            record = ak._v2._util.wrap(self, behavior=behavior)
            contents = []
            for field in self._array.fields:
                contents.append(record[field])
                if isinstance(
                    contents[-1], (ak._v2.highlevel.Array, ak._v2.highlevel.Record)
                ):
                    contents[-1] = contents[-1]._layout._to_list(
                        behavior, json_conversions
                    )
                elif hasattr(contents[-1], "tolist"):
                    contents[-1] = contents[-1].tolist()

        else:
            contents = [
                content[self._at : self._at + 1]._to_list(behavior, json_conversions)[0]
                for content in self._array.contents
            ]

        if self.is_tuple and json_conversions is None:
            return tuple(contents)
        else:
            return dict(zip(self._array.fields, contents))

    def __copy__(self):
        return Record(self._array, self._at)

    def __deepcopy__(self, memo):
        return Record(copy.deepcopy(self._array, memo), self._at)

    def recursively_apply(
        self,
        action,
        behavior=None,
        depth_context=None,
        lateral_context=None,
        allow_records=True,
        keep_parameters=True,
        numpy_to_regular=True,
        return_array=True,
        function_name=None,
    ):

        out = self._array.recursively_apply(
            action,
            behavior,
            depth_context,
            lateral_context,
            allow_records,
            keep_parameters,
            numpy_to_regular,
            return_array,
            function_name,
        )

        if return_array:
            return Record(out, self._at)
        else:
            return None

    def _jax_flatten(self):
        from awkward._v2._connect.jax import _find_numpyarray_nodes, AuxData

        numpyarray_nodes = _find_numpyarray_nodes(self)
        return (numpyarray_nodes, AuxData(self))

    @classmethod
    def jax_flatten(cls, array):
        assert type(array) is cls
        return array._jax_flatten()

    @classmethod
    def jax_unflatten(cls, aux_data, children):
        from awkward._v2._connect.jax import _replace_numpyarray_nodes

        return ak._v2._util.wrap(
            _replace_numpyarray_nodes(aux_data.layout, list(children))
        )
