# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import re
import keyword
try:
    from collections.abc import Sequence
    from collections.abc import Iterable
except:
    from collections import Sequence
    from collections import Iterable

import numpy

import awkward1._connect._numpy
import awkward1._connect._pandas
import awkward1.layout
import awkward1.operations.convert

_dir_pattern = re.compile(r"^[a-zA-Z_]\w*$")

class Array(awkward1._connect._numpy.NDArrayOperatorsMixin,
            awkward1._connect._pandas.PandasMixin, Sequence):
    def __init__(self, data, behavior=None, checkvalid=False):
        if isinstance(data, awkward1.layout.Content):
            layout = data
        elif isinstance(data, Array):
            layout = data.layout
        elif isinstance(data, numpy.ndarray):
            layout = awkward1.operations.convert.fromnumpy(data,
                                                           highlevel=False)
        elif isinstance(data, str):
            layout = awkward1.operations.convert.fromjson(data,
                                                          highlevel=False)
        elif isinstance(data, dict):
            raise TypeError(
                    "could not convert dict into an awkward1.Array; "
                    "try awkward1.Record")
        else:
            layout = awkward1.operations.convert.fromiter(data,
                                                          highlevel=False,
                                                          allowrecord=False)
        if not isinstance(layout, awkward1.layout.Content):
            raise TypeError("could not convert data into an awkward1.Array")

        if self.__class__ is Array:
            self.__class__ = awkward1._util.arrayclass(layout, behavior)

        self.layout = layout
        self.behavior = behavior
        if checkvalid:
            awkward1.operations.describe.validityerror(self, exception=True)

    @property
    def layout(self):
        return self._layout

    @layout.setter
    def layout(self, layout):
        if isinstance(layout, awkward1.layout.Content):
            self._layout = layout
            self._numbaview = None
        else:
            raise TypeError(
                    "layout must be a subclass of awkward1.layout.Content")

    @property
    def behavior(self):
        return self._behavior

    @behavior.setter
    def behavior(self, behavior):
        if behavior is None or isinstance(behavior, dict):
            self._behavior = behavior
        else:
            raise TypeError("behavior must be None or a dict")

    @property
    def type(self):
        return awkward1.types.ArrayType(
                 self._layout.type(awkward1._util.typestrs(self._behavior)),
                 len(self._layout))

    def __len__(self):
        return len(self._layout)

    def __iter__(self):
        for x in self._layout:
            yield awkward1._util.wrap(x, self._behavior)

    def __getitem__(self, where):
        return awkward1._util.wrap(self._layout[where], self._behavior)

    def __setitem__(self, where, what):
        if not isinstance(where, str):
            raise ValueError(
                    "only fields may be assigned in-place (by field name)")
        self._layout = awkward1.operations.structure.withfield(self._layout,
                                                               what,
                                                               where).layout
        self._numbaview = None

    def __getattr__(self, where):
        if where in dir(type(self)):
            return super(Array, self).__getattribute__(where)
        else:
            if where in self._layout.keys():
                try:
                    return self[where]
                except Exception as err:
                    raise AttributeError(
                            "while trying to get field {0}, an exception "
                            "occurred:\n{1}: {2}".format(repr(where),
                                                         type(err),
                                                         str(err)))
            else:
                raise AttributeError("no field named {0}".format(repr(where)))

    def __dir__(self):
        return sorted(set(dir(super(Array, self))
                          + [x for x in self._layout.keys()
                               if _dir_pattern.match(x) and
                                  not keyword.iskeyword(x)]))

    @property
    def i0(self):
        return self["0"]
    @property
    def i1(self):
        return self["1"]
    @property
    def i2(self):
        return self["2"]
    @property
    def i3(self):
        return self["3"]
    @property
    def i4(self):
        return self["4"]
    @property
    def i5(self):
        return self["5"]
    @property
    def i6(self):
        return self["6"]
    @property
    def i7(self):
        return self["7"]
    @property
    def i8(self):
        return self["8"]
    @property
    def i9(self):
        return self["9"]

    def __str__(self, limit_value=85):
        return awkward1._util.minimally_touching_string(limit_value,
                                                        self._layout,
                                                        self._behavior)

    def __repr__(self, limit_value=40, limit_total=85):
        value = awkward1._util.minimally_touching_string(limit_value,
                                                         self._layout,
                                                         self._behavior)

        limit_type = limit_total - len(value) - len("<Array  type=>")
        type = repr(str(self.type))
        if len(type) > limit_type:
            type = type[:(limit_type - 4)] + "..." + type[-1]

        return "<Array {0} type={1}>".format(value, type)

    def __array__(self, *args, **kwargs):
        if awkward1._util.called_by_module("pandas"):
            try:
                return awkward1._connect._numpy.convert_to_array(self._layout,
                                                                 args,
                                                                 kwargs)
            except:
                out = numpy.empty(len(self._layout), dtype="O")
                for i, x in enumerate(self._layout):
                    out[i] = awkward1._util.wrap(x, self._behavior)
                return out
        else:
            return awkward1._connect._numpy.convert_to_array(self._layout,
                                                             args,
                                                             kwargs)

    def __array_function__(self, func, types, args, kwargs):
        return awkward1._connect._numpy.array_function(func,
                                                       types,
                                                       args,
                                                       kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return awkward1._connect._numpy.array_ufunc(ufunc,
                                                    method,
                                                    inputs,
                                                    kwargs,
                                                    self._behavior)

    @property
    def numbatype(self):
        import numba
        import awkward1._connect._numba
        awkward1._connect._numba.register()
        if self._numbaview is None:
            self._numbaview = \
              awkward1._connect._numba.arrayview.ArrayView.fromarray(self)
        return numba.typeof(self._numbaview)

class Record(awkward1._connect._numpy.NDArrayOperatorsMixin):
    def __init__(self, data, behavior=None, checkvalid=False):
        if isinstance(data, awkward1.layout.Record):
            layout = data
        elif isinstance(data, Record):
            layout = data.layout
        elif isinstance(data, str):
            layout = awkward1.operations.convert.fromjson(data,
                                                          highlevel=False)
        elif isinstance(data, dict):
            layout = awkward1.operations.convert.fromiter([data],
                                                          highlevel=False)[0]
        elif isinstance(data, Iterable):
            raise TypeError("could not convert non-dict into an "
                            "awkward1.Record; try awkward1.Array")
        else:
            layout = None
        if not isinstance(layout, awkward1.layout.Record):
            raise TypeError("could not convert data into an awkward1.Record")

        if self.__class__ is Record:
            self.__class__ = awkward1._util.recordclass(layout, behavior)

        self.layout = layout
        self.behavior = behavior
        if checkvalid:
            awkward1.operations.describe.validityerror(self, exception=True)

    @property
    def layout(self):
        return self._layout

    @layout.setter
    def layout(self, layout):
        if isinstance(layout, awkward1.layout.Record):
            self._layout = layout
            self._numbaview = None
        else:
            raise TypeError(
                    "layout must be a subclass of awkward1.layout.Record")

    @property
    def behavior(self):
        return self._behavior

    @behavior.setter
    def behavior(self, behavior):
        if behavior is None or isinstance(behavior, dict):
            self._behavior = behavior
        else:
            raise TypeError("behavior must be None or a dict")

    @property
    def type(self):
        return self._layout.type(awkward1._util.typestrs(self._behavior))

    def __getitem__(self, where):
        return awkward1._util.wrap(self._layout[where], self._behavior)

    def __setitem__(self, where, what):
        if not isinstance(where, str):
            raise ValueError(
                    "only fields may be assigned in-place (by field name)")
        self._layout = awkward1.operations.structure.withfield(self._layout,
                                                               what,
                                                               where).layout
        self._numbaview = None

    def __getattr__(self, where):
        if where in dir(type(self)):
            return super(Record, self).__getattribute__(where)
        else:
            if where in self._layout.keys():
                try:
                    return self[where]
                except Exception as err:
                    raise AttributeError(
                            "while trying to get field {0}, an exception "
                            "occurred:\n{1}: {2}".format(repr(where),
                                                         type(err),
                                                         str(err)))
            else:
                raise AttributeError("no field named {0}".format(repr(where)))

    def __dir__(self):
        return sorted(set(dir(super(Array, self))
                          + [x for x in self._layout.keys()
                               if _dir_pattern.match(x) and
                               not keyword.iskeyword(x)]))

    @property
    def i0(self):
        return self["0"]
    @property
    def i1(self):
        return self["1"]
    @property
    def i2(self):
        return self["2"]
    @property
    def i3(self):
        return self["3"]
    @property
    def i4(self):
        return self["4"]
    @property
    def i5(self):
        return self["5"]
    @property
    def i6(self):
        return self["6"]
    @property
    def i7(self):
        return self["7"]
    @property
    def i8(self):
        return self["8"]
    @property
    def i9(self):
        return self["9"]

    def __str__(self, limit_value=85):
        return awkward1._util.minimally_touching_string(limit_value + 2,
                                                        self._layout,
                                                        self._behavior)[1:-1]

    def __repr__(self, limit_value=40, limit_total=85):
        value = awkward1._util.minimally_touching_string(limit_value + 2,
                                                         self._layout,
                                                         self._behavior)[1:-1]

        limit_type = limit_total - len(value) - len("<Record  type=>")
        type = repr(str(self.type))
        if len(type) > limit_type:
            type = type[:(limit_type - 4)] + "..." + type[-1]

        return "<Record {0} type={1}>".format(value, type)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return awkward1._connect._numpy.array_ufunc(ufunc,
                                                    method,
                                                    inputs,
                                                    kwargs,
                                                    self._behavior)

    @property
    def numbatype(self):
        import numba
        import awkward1._connect._numba
        awkward1._connect._numba.register()
        if self._numbaview is None:
            self._numbaview = \
              awkward1._connect._numba.arrayview.RecordView.fromrecord(self)
        return numba.typeof(self._numbaview)

class ArrayBuilder(Sequence):
    @classmethod
    def _wrap(cls, layout, behavior=None):
        assert isinstance(layout, awkward1.layout.ArrayBuilder)
        out = cls.__new__(cls)
        out._layout = layout
        out.behavior = behavior
        return out

    def __init__(self, behavior=None):
        self._layout = awkward1.layout.ArrayBuilder()
        self.behavior = behavior

    @property
    def behavior(self):
        return self._behavior

    @behavior.setter
    def behavior(self, behavior):
        if behavior is None or isinstance(behavior, dict):
            self._behavior = behavior
        else:
            raise TypeError("behavior must be None or a dict")

    @property
    def type(self):
        tmp = self._layout.snapshot()
        return awkward1.types.ArrayType(
                 tmp.type(awkward1._util.typestrs(self._behavior)), len(tmp))

    def __len__(self):
        return len(self._layout)

    def __getitem__(self, where):
        return awkward1._util.wrap(self._layout[where], self._behavior)

    def __iter__(self):
        for x in self._layout.snapshot():
            yield awkward1._util.wrap(x, self._behavior)

    def __str__(self, limit_value=85, snapshot=None):
        if snapshot is None:
            snapshot = self.snapshot()
        return snapshot.__str__(limit_value=limit_value)

    def __repr__(self, limit_value=40, limit_total=85):
        snapshot = self.snapshot()
        value = self.__str__(limit_value=limit_value, snapshot=snapshot)

        limit_type = limit_total - len(value) - len("<ArrayBuilder  type=>")
        typestrs = awkward1._util.typestrs(self._behavior)
        type = repr(str(snapshot.type(typestrs)))
        if len(type) > limit_type:
            type = type[:(limit_type - 4)] + "..." + type[-1]

        return "<ArrayBuilder {0} type={1}>".format(value, type)

    def __array__(self, *args, **kwargs):
        return awkward1._connect._numpy.convert_to_array(
                 self._layout.snapshot(), args, kwargs)

    def __array_function__(self, func, types, args, kwargs):
        return awkward1._connect._numpy.array_function(func,
                                                       types,
                                                       args,
                                                       kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return awkward1._connect._numpy.array_ufunc(ufunc,
                                                    method,
                                                    inputs,
                                                    kwargs,
                                                    self._behavior)

    @property
    def numbatype(self):
        import numba
        import awkward1._connect._numba.builder
        awkward1._connect._numba.register()
        return awkward1._connect._numba.builder.ArrayBuilderType(
                 self._behavior)

    def snapshot(self):
        return awkward1._util.wrap(self._layout.snapshot(), self._behavior)

    def null(self):
        self._layout.null()

    def boolean(self, x):
        self._layout.boolean(x)

    def integer(self, x):
        self._layout.integer(x)

    def real(self, x):
        self._layout.real(x)

    def bytestring(self, x):
        self._layout.bytestring(x)

    def string(self, x):
        self._layout.string(x)

    def beginlist(self):
        self._layout.beginlist()

    def endlist(self):
        self._layout.endlist()

    def begintuple(self):
        self._layout.begintuple()

    def index(self, i):
        self._layout.index(i)

    def endtuple(self):
        self._layout.endtuple()

    def beginrecord(self, name=None):
        self._layout.beginrecord(name)

    def field(self, key):
        self._layout.field(key)

    def endrecord(self):
        self._layout.endrecord()

    def append(self, obj, at=None):
        if at is None:
            if isinstance(obj, Record):
                self._layout.append(obj.layout.array, obj.layout.at)
            elif isinstance(obj, Array):
                self._layout.extend(obj.layout)
            else:
                self._layout.fromiter(obj)

        else:
            if isinstance(obj, Array):
                self._layout.append(obj.layout, at)
            else:
                raise TypeError(
                        "'append' method can only be used with 'at' when "
                        "'obj' is an ak.Array")

    def extend(self, obj):
        if isinstance(obj, Array):
            self._layout.extend(obj.layout)
        else:
            raise TypeError("'extend' method requires an ak.Array")
