# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import awkward as ak
from awkward._v2.forms.form import Form, _parameters_equal, nonvirtual
from awkward._v2.forms.indexedform import IndexedForm


class RecordForm(Form):
    def __init__(
        self,
        contents,
        keys,
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
        if keys is not None and not isinstance(keys, Iterable):
            raise TypeError(
                "{0} 'keys' must be iterable, not {1}".format(
                    type(self).__name__, repr(contents)
                )
            )

        self._keys = keys
        self._contents = list(contents)
        self._init(has_identifier, parameters, form_key)

    @property
    def keys(self):
        if self._keys is None:
            return [str(i) for i in range(len(self._contents))]
        else:
            return self._keys

    @property
    def is_tuple(self):
        return self._keys is None

    @property
    def contents(self):
        return self._contents

    def __repr__(self):
        args = [repr(self._contents), repr(self._keys)] + self._repr_args()
        return "{0}({1})".format(type(self).__name__, ", ".join(args))

    def index_to_key(self, index):
        if 0 <= index < len(self._contents):
            if self._keys is None:
                return str(index)
            else:
                return self._keys[index]
        else:
            raise IndexError(
                "no index {0} in record with {1} fields".format(
                    index, len(self._contents)
                )
            )

    def key_to_index(self, key):
        if self._keys is None:
            try:
                i = int(key)
            except ValueError:
                pass
            else:
                if 0 <= i < len(self._contents):
                    return i
        else:
            try:
                i = self._keys.index(key)
            except ValueError:
                pass
            else:
                return i
        raise IndexError(
            "no field {0} in record with {1} fields".format(
                repr(key), len(self._contents)
            )
        )

    def haskey(self, key):
        if self._keys is None:
            try:
                i = int(key)
            except ValueError:
                return False
            else:
                return 0 <= i < len(self._contents)
        else:
            return key in self._keys

    def content(self, index_or_key):
        if ak._util.isint(index_or_key):
            index = index_or_key
        elif ak._util.isstr(index_or_key):
            index = self.key_to_index(index_or_key)
        else:
            raise TypeError(
                "index_or_key must be an integer (index) or string (key), not {0}".format(
                    repr(index_or_key)
                )
            )
        return self._contents[index]

    def _tolist_part(self, verbose, toplevel):
        out = {"class": "RecordArray"}

        contents_tolist = [
            content._tolist_part(verbose, toplevel=False) for content in self._contents
        ]
        if self._keys is not None:
            out["contents"] = dict(zip(self._keys, contents_tolist))
        else:
            out["contents"] = contents_tolist

        return self._tolist_extra(out, verbose)

    def _type(self, typestrs):
        return ak._v2.types.recordtype.RecordType(
            [x._type(typestrs) for x in self._contents],
            self._keys,
            self._parameters,
            ak._util.gettypestr(self._parameters, typestrs),
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
                    if set(self._keys) != set(other._keys):
                        return False
                    else:
                        for key, content in zip(self._keys, self._contents):
                            if content != other.content(key):
                                return False
                        else:
                            return True
            else:
                return False
        else:
            return False

    def generated_compatibility(self, other):
        other = nonvirtual(other)

        if other is None:
            return True

        elif isinstance(other, RecordForm):
            if self.is_tuple == other.is_tuple:
                self_keys = set(self.keys)
                other_keys = set(other.keys)
                if self_keys == other_keys:
                    return _parameters_equal(
                        self._parameters, other._parameters
                    ) and all(
                        self.content(x).generated_compatibility(other.content(x))
                        for x in self_keys
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
            self._keys,
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
        indexes = [self.key_to_index(key) for key in where]
        if self._keys is None:
            keys = None
        else:
            keys = [self._keys[i] for i in indexes]

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
            keys,
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
                self._keys,
                has_identifier=self._has_identifier,
                parameters=self._parameters,
                form_key=None,
            )

    @property
    def purelist_isregular(self):
        return True

    @property
    def purelist_depth(self):
        return 1

    @property
    def minmax_depth(self):
        if len(self._contents) == 0:
            return (0, 0)
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
