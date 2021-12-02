# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import awkward as ak
from awkward._v2.forms.form import Form, _parameters_equal
from awkward._v2.forms.indexedform import IndexedForm


class RecordForm(Form):
    def __init__(
        self,
        contents,
        fields,
        has_identifier=False,
        parameters=None,
        form_key=None,
    ):
        if not isinstance(contents, Iterable):
            raise TypeError(
                "{0} 'contents' must be iterable, not {1}".format(
                    type(self).__name__, repr(contents)
                )
            )
        for content in contents:
            if not isinstance(content, Form):
                raise TypeError(
                    "{0} all 'contents' must be Form subclasses, not {1}".format(
                        type(self).__name__, repr(content)
                    )
                )
        if fields is not None and not isinstance(fields, Iterable):
            raise TypeError(
                "{0} 'fields' must be iterable, not {1}".format(
                    type(self).__name__, repr(contents)
                )
            )

        self._fields = fields
        self._contents = list(contents)
        self._init(has_identifier, parameters, form_key)

    @property
    def fields(self):
        if self._fields is None:
            return [str(i) for i in range(len(self._contents))]
        else:
            return self._fields

    @property
    def is_tuple(self):
        return self._fields is None

    @property
    def contents(self):
        return self._contents

    def __repr__(self):
        args = [repr(self._contents), repr(self._fields)] + self._repr_args()
        return "{0}({1})".format(type(self).__name__, ", ".join(args))

    def index_to_field(self, index):
        if 0 <= index < len(self._contents):
            if self._fields is None:
                return str(index)
            else:
                return self._fields[index]
        else:
            raise IndexError(
                "no index {0} in record with {1} fields".format(
                    index, len(self._contents)
                )
            )

    def field_to_index(self, field):
        if self._fields is None:
            try:
                i = int(field)
            except ValueError:
                pass
            else:
                if 0 <= i < len(self._contents):
                    return i
        else:
            try:
                i = self._fields.index(field)
            except ValueError:
                pass
            else:
                return i
        raise IndexError(
            "no field {0} in record with {1} fields".format(
                repr(field), len(self._contents)
            )
        )

    def has_field(self, field):
        if self._fields is None:
            try:
                i = int(field)
            except ValueError:
                return False
            else:
                return 0 <= i < len(self._contents)
        else:
            return field in self._fields

    def content(self, index_or_field):
        if ak._util.isint(index_or_field):
            index = index_or_field
        elif ak._util.isstr(index_or_field):
            index = self.field_to_index(index_or_field)
        else:
            raise TypeError(
                "index_or_field must be an integer (index) or string (field), not {0}".format(
                    repr(index_or_field)
                )
            )
        return self._contents[index]

    def _tolist_part(self, verbose, toplevel):
        out = {"class": "RecordArray"}

        contents_tolist = [
            content._tolist_part(verbose, toplevel=False) for content in self._contents
        ]
        if self._fields is not None:
            out["contents"] = dict(zip(self._fields, contents_tolist))
        else:
            out["contents"] = contents_tolist

        return self._tolist_extra(out, verbose)

    def _type(self, typestrs):
        return ak._v2.types.recordtype.RecordType(
            [x._type(typestrs) for x in self._contents],
            self._fields,
            self._parameters,
            ak._v2._util.gettypestr(self._parameters, typestrs),
        )

    def __eq__(self, other):
        if isinstance(other, RecordForm):
            if (
                self._has_identifier == other._has_identifier
                and self._form_key == other._form_key
                and self.is_tuple == other.is_tuple
                and len(self._contents) == len(other._contents)
                and _parameters_equal(self._parameters, other._parameters)
            ):
                if self.is_tuple:
                    for i in range(len(self._contents)):
                        if self._contents[i] != other._contents[i]:
                            return False
                    else:
                        return True
                else:
                    if set(self._fields) != set(other._fields):
                        return False
                    else:
                        for field, content in zip(self._fields, self._contents):
                            if content != other.content(field):
                                return False
                        else:
                            return True
            else:
                return False
        else:
            return False

    def generated_compatibility(self, other):
        if other is None:
            return True

        elif isinstance(other, RecordForm):
            if self.is_tuple == other.is_tuple:
                self_fields = set(self._fields)
                other_fields = set(other._fields)
                if self_fields == other_fields:
                    return _parameters_equal(
                        self._parameters, other._parameters
                    ) and all(
                        self.content(x).generated_compatibility(other.content(x))
                        for x in self_fields
                    )
                else:
                    return False
            else:
                return False

        else:
            return False

    def _getitem_range(self):
        return RecordForm(
            self._contents,
            self._fields,
            has_identifier=self._has_identifier,
            parameters=self._parameters,
            form_key=None,
        )

    def _getitem_field(self, where, only_fields=()):
        if len(only_fields) == 0:
            return self.content(where)

        else:
            nexthead, nexttail = ak._v2._slicing.headtail(only_fields)
            if ak._util.isstr(nexthead):
                return self.content(where)._getitem_field(nexthead, nexttail)
            else:
                return self.content(where)._getitem_fields(nexthead, nexttail)

    def _getitem_fields(self, where, only_fields=()):
        indexes = [self.field_to_index(field) for field in where]
        if self._fields is None:
            fields = None
        else:
            fields = [self._fields[i] for i in indexes]

        if len(only_fields) == 0:
            contents = [self.content(i) for i in indexes]
        else:
            nexthead, nexttail = ak._v2._slicing.headtail(only_fields)
            if ak._util.isstr(nexthead):
                contents = [
                    self.content(i)._getitem_field(nexthead, nexttail) for i in indexes
                ]
            else:
                contents = [
                    self.content(i)._getitem_fields(nexthead, nexttail) for i in indexes
                ]

        return RecordForm(
            contents,
            fields,
            has_identifier=self._has_identifier,
            parameters=None,
            form_key=None,
        )

    def _carry(self, allow_lazy):
        if allow_lazy:
            return IndexedForm(
                "i64",
                self,
                has_identifier=self._has_identifier,
                parameters=None,
                form_key=None,
            )
        else:
            return RecordForm(
                self._contents,
                self._fields,
                has_identifier=self._has_identifier,
                parameters=self._parameters,
                form_key=None,
            )

    def purelist_parameter(self, key):
        return self.parameter(key)

    @property
    def purelist_isregular(self):
        return True

    @property
    def purelist_depth(self):
        return 1

    @property
    def minmax_depth(self):
        if len(self._contents) == 0:
            return (1, 1)
        mins, maxs = [], []
        for content in self._contents:
            mindepth, maxdepth = content.minmax_depth
            mins.append(mindepth)
            maxs.append(maxdepth)
        return (min(mins), max(maxs))

    @property
    def branch_depth(self):
        if len(self._contents) == 0:
            return (False, 1)
        anybranch = False
        mindepth = None
        for content in self._contents:
            branch, depth = content.branch_depth
            if mindepth is None:
                mindepth = depth
            if branch or mindepth != depth:
                anybranch = True
            if mindepth > depth:
                mindepth = depth
        return (anybranch, mindepth)

    @property
    def dimension_optiontype(self):
        return False
