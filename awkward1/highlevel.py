# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import numpy

import awkward1._numpy
import awkward1._pandas
import awkward1.layout
import awkward1.operations.convert

class Array(awkward1._numpy.NDArrayOperatorsMixin, awkward1._pandas.PandasMixin):
    def __init__(self, data, type=None, classes=None, functions=None):
        if isinstance(data, awkward1.layout.Content):
            layout = data
        elif isinstance(data, Array):
            layout = data.layout
        elif isinstance(data, numpy.ndarray):
            layout = awkward1.operations.convert.fromnumpy(data).layout
        elif isinstance(data, str):
            layout = awkward1.operations.convert.fromjson(data).layout
        else:
            layout = awkward1.operations.convert.fromiter(data).layout
        if not isinstance(layout, awkward1.layout.Content):
            raise TypeError("could not convert data into an awkward1.Array")

        self._classes = classes
        self._functions = functions

        if type is not None:
            if not isinstance(type, awkward1.layout.ArrayType):
                raise TypeError("type must be an awkward1.layout.ArrayType")
            if type.length > len(layout):
                raise TypeError("ArrayType length ({0}) is greater than layout length {1}".format(type.length, len(layout)))
            if type.length < len(layout):
                layout = layout[:type.length]
            cls = awkward1._util.regular_classes(self._classes).get(type.parameters.get("__class__"))
            if cls is not None:
                if not isinstance(cls, __builtins__["type"]) or not issubclass(cls, Array):
                    raise TypeError("type.parameters['__class__'] = {0} must be a subclass of awkward1.Array".format(repr(type.parameters["__class__"])))
                self.__class__ = cls
            layout = layout.astype(type.type)

        self.layout = layout

    @property
    def layout(self):
        return self._layout

    @layout.setter
    def layout(self, layout):
        if not isinstance(layout, awkward1.layout.Content):
            raise TypeError("layout must be a subclass of awkward1.layout.Content")
        self._layout = layout

    @property
    def type(self):
        return awkward1.layout.ArrayType(self._layout.type, len(self._layout))

    def __len__(self):
        return len(self._layout)

    def __iter__(self):
        for x in self._layout:
            yield awkward1._util.wrap(x, self._classes, self._functions)

    def __getitem__(self, where):
        return awkward1._util.wrap(self._layout[where], self._classes, self._functions)

    def __str__(self, limit_value=85):
        return awkward1._util.minimally_touching_string(limit_value, self._layout, self._classes, self._functions)

    def __repr__(self, limit_value=40, limit_total=85):
        value = awkward1._util.minimally_touching_string(limit_value, self._layout, self._classes, self._functions)

        limit_type = limit_total - len(value) - len("<Array  type=>")
        type = repr(str(self.type))
        if len(type) > limit_type:
            type = type[:(limit_type - 4)] + "..." + type[-1]

        return "<Array {0} type={1}>".format(value, type)

    def __array__(self, *args, **kwargs):
        if awkward1._util.called_by_module("pandas"):
            try:
                return awkward1._numpy.convert_to_array(self._layout, args, kwargs)
            except:
                out = numpy.empty(len(self._layout), dtype="O")
                for i, x in enumerate(self._layout):
                    out[i] = awkward1._util.wrap(x, self._classes, self._functions)
                return out
        else:
            return awkward1._numpy.convert_to_array(self._layout, args, kwargs)

    def __array_function__(self, func, types, args, kwargs):
        return awkward1._numpy.array_function(func, types, args, kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return awkward1._numpy.array_ufunc(ufunc, method, inputs, kwargs, self._classes, self._functions)

class Record(awkward1._numpy.NDArrayOperatorsMixin):
    def __init__(self, data, type=None, classes=None, functions=None):
        # FIXME: more checks here
        layout = data
        if not isinstance(layout, awkward1.layout.Record):
            raise TypeError("could not convert data into an awkward1.Record")

        self._classes = classes
        self._functions = functions

        if type is not None:
            if not isinstance(type, awkward1.layout.RecordType):
                raise TypeError("type must be an awkward1.layout.RecordType")
            cls = awkward1._util.regular_classes(self._classes).get(type.parameters.get("__class__"))
            if cls is not None:
                if not isinstance(cls, __builtins__["type"]) or not issubclass(cls, Record):
                    raise TypeError("type.parameters['__class__'] = {0} must be a subclass of awkward1.Record".format(repr(type.parameters["__class__"])))
                self.__class__ = cls
            layout = layout.astype(type)

        self.layout = layout

    @property
    def layout(self):
        return self._layout

    @layout.setter
    def layout(self, layout):
        if not isinstance(layout, awkward1.layout.Record):
            raise TypeError("layout must be a subclass of awkward1.layout.Record")
        self._layout = layout

    @property
    def type(self):
        return self._layout.type

    def __getitem__(self, where):
        return awkward1._util.wrap(self._layout[where], self._classes, self._functions)

    def __str__(self, limit_value=85):
        return awkward1._util.minimally_touching_string(limit_value + 2, self._layout, self._classes, self._functions)[1:-1]

    def __repr__(self, limit_value=40, limit_total=85):
        value = awkward1._util.minimally_touching_string(limit_value + 2, self._layout, self._classes, self._functions)[1:-1]

        limit_type = limit_total - len(value) - len("<Record  type=>")
        type = repr(str(self._layout.type))
        if len(type) > limit_type:
            type = type[:(limit_type - 4)] + "..." + type[-1]

        return "<Record {0} type={1}>".format(value, type)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return awkward1._numpy.array_ufunc(ufunc, method, inputs, kwargs, self._classes, self._functions)

class FillableArray(object):
    def __init__(self, classes=None, functions=None):
        self._fillablearray = awkward1.layout.FillableArray()
        self._classes = classes
        self._functions = functions

    @property
    def type(self):
        tmp = self._fillablearray.snapshot()
        return awkward1.layout.ArrayType(tmp.type, len(tmp))

    def __len__(self):
        return len(self._fillablearray)

    def __getitem__(self, where):
        return awkward1._util.wrap(self._fillablearray[where], self._classes, self._functions)

    def __iter__(self):
        for x in self._fillablearray.snapshot():
            yield awkward1._util.wrap(x, self._classes, self._functions)

    def __str__(self, limit_value=85, snapshot=None):
        if snapshot is None:
            snapshot = self.snapshot()
        return snapshot.__str__(limit_value=limit_value)

    def __repr__(self, limit_value=40, limit_total=85):
        snapshot = self.snapshot()
        value = self.__str__(limit_value=limit_value, snapshot=snapshot)

        limit_type = limit_total - len(value) - len("<FillableArray  type=>")
        type = repr(str(snapshot.type))
        if len(type) > limit_type:
            type = type[:(limit_type - 4)] + "..." + type[-1]

        return "<FillableArray {0} type={1}>".format(value, type)

    def __array__(self, *args, **kwargs):
        return awkward1._numpy.convert_to_array(self._fillablearray.snapshot(), args, kwargs)

    def __array_function__(self, func, types, args, kwargs):
        return awkward1._numpy.array_function(func, types, args, kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return awkward1._numpy.array_ufunc(ufunc, method, inputs, kwargs, self._classes, self._functions)

    def snapshot(self):
        return awkward1._util.wrap(self._fillablearray.snapshot(), self._classes, self._functions)

    def null(self):
        self._fillablearray.null()

    def boolean(self, x):
        self._fillablearray.boolean(x)

    def integer(self, x):
        self._fillablearray.integer(x)

    def real(self, x):
        self._fillablearray.real(x)

    def bytestring(self, x):
        self._fillablearray.bytestring(x)

    def string(self, x):
        self._fillablearray.string(x)

    def beginlist(self):
        self._fillablearray.beginlist()

    def endlist(self):
        self._fillablearray.endlist()

    def begintuple(self):
        self._fillablearray.begintuple()

    def index(self, i):
        self._fillablearray.index(i)

    def endtuple(self):
        self._fillablearray.endtuple()

    def beginrecord(self, name=None):
        self._fillablearray.beginrecord(name)

    def field(self, key):
        self._fillablearray.field(key)

    def endrecord(self):
        self._fillablearray.endrecord()
