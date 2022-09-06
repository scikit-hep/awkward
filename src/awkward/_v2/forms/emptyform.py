# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import awkward as ak
from awkward._v2.forms.form import Form, _parameters_equal


class EmptyForm(Form):
    is_NumpyType = True
    is_UnknownType = True

    def __init__(self, has_identifier=False, parameters=None, form_key=None):
        self._init(has_identifier, parameters, form_key)

    def __repr__(self):
        args = self._repr_args()
        return "{}({})".format(type(self).__name__, ", ".join(args))

    def _tolist_part(self, verbose, toplevel):
        return self._tolist_extra({"class": "EmptyArray"}, verbose)

    def _type(self, typestrs):
        return ak._v2.types.unknowntype.UnknownType(
            self._parameters,
            ak._v2._util.gettypestr(self._parameters, typestrs),
        )

    def __eq__(self, other):
        return (
            isinstance(other, EmptyForm)
            and self._has_identifier == other._has_identifier
            and self._form_key == other._form_key
        )

    def toNumpyForm(self, dtype):
        return ak._v2.forms.numpyform.from_dtype(dtype, self._parameters)

    def generated_compatibility(self, other):
        if other is None:
            return True

        elif isinstance(other, EmptyForm):
            return _parameters_equal(
                self._parameters, other._parameters, only_array_record=True
            )

        else:
            return False

    def _getitem_range(self):
        return EmptyForm(
            has_identifier=self._has_identifier,
            parameters=self._parameters,
            form_key=None,
        )

    def _getitem_field(self, where, only_fields=()):
        raise ak._v2._util.indexerror(self, where, "not an array of records")

    def _getitem_fields(self, where, only_fields=()):
        raise ak._v2._util.indexerror(self, where, "not an array of records")

    def _carry(self, allow_lazy):
        return EmptyForm(
            has_identifier=self._has_identifier,
            parameters=self._parameters,
            form_key=None,
        )

    def purelist_parameter(self, key):
        if self._parameters is None or key not in self._parameters:
            return None
        else:
            return self._parameters[key]

    @property
    def purelist_isregular(self):
        return True

    @property
    def purelist_depth(self):
        return 1

    @property
    def is_identity_like(self):
        return True

    @property
    def minmax_depth(self):
        return (1, 1)

    @property
    def branch_depth(self):
        return (False, 1)

    @property
    def fields(self):
        return []

    @property
    def is_tuple(self):
        return False

    @property
    def dimension_optiontype(self):
        return False

    def _columns(self, path, output, list_indicator):
        output.append(".".join(path))

    def _select_columns(self, index, specifier, matches, output):
        if any(match and index >= len(item) for item, match in zip(specifier, matches)):
            output.append(None)
        return self

    def _column_types(self):
        return ("empty",)
