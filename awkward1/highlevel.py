# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import numpy

import awkward1.layout
import awkward1.operations.convert

class Array(object):
    def __init__(self, data, namespace=None):
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
        self.layout = layout
        self.namespace = namespace

    @property
    def layout(self):
        return self._layout

    @layout.setter
    def layout(self, layout):
        if not isinstance(layout, awkward1.layout.Content):
            raise TypeError("layout must be a subclass of awkward1.layout.Content")
        self._layout = layout

    @property
    def namespace(self):
        return self._namespace

    @namespace.setter
    def namespace(self, namespace):
        if namespace is None:
            self._namespace = awkward1.namespace
        else:
            self._namespace = namespace

    @property
    def type(self):
        return awkward1.layout.ArrayType(self._layout.type, len(self._layout))

    # @type.setter
    # def type(self, type):
    #     if type is None:
    #         self.__class__ = Array
    #         self._type = None
    #         self._layout.type = None
    #
    #     else:
    #         if not isinstance(type, awkward1.layout.ArrayType):
    #             raise TypeError("type must be an awkward1.layout.ArrayType")
    #
    #         cls = self._namespace.get(type.parameters.get("__class__"))
    #         if cls is None:
    #             cls = Array
    #         elif not isinstance(cls, __builtins__["type"]) or not issubclass(cls, Array):
    #             raise TypeError("type.parameters['__class__'] = {0} must be a subclass of awkward1.Array".format(repr(type.parameters["__class__"])))
    #
    #         self.__class__ = cls
    #         self._type = type
    #         self._layout.type = type.inner()

    def __len__(self):
        return len(self.layout)

    def __iter__(self):
        for x in self.layout:
            yield awkward1._util.wrap(x, self._namespace)

    def __getitem__(self, where):
        return awkward1._util.wrap(self.layout[where], self._namespace)

    def __str__(self, limit_value=85):
        return awkward1._util.minimally_touching_string(limit_value, self._layout, self._namespace)

    def __repr__(self, limit_value=40, limit_total=85):
        value = awkward1._util.minimally_touching_string(limit_value, self._layout, self._namespace)

        limit_type = limit_total - len(value) - len("<Array  type=>")
        type = repr(str(self.type))
        if len(type) > limit_type:
            type = type[:(limit_type - 4)] + "..." + type[-1]

        return "<Array {0} type={1}>".format(value, type)

class Record(object):
    def __init__(self, data, namespace=None):
        # FIXME: more checks here
        layout = data
        if not isinstance(layout, awkward1.layout.Record):
            raise TypeError("could not convert data into an awkward1.Record")
        self.layout = layout
        self.namespace = namespace

    @property
    def layout(self):
        return self._layout

    @layout.setter
    def layout(self, layout):
        if not isinstance(layout, awkward1.layout.Record):
            raise TypeError("layout must be a subclass of awkward1.layout.Record")
        self._layout = layout

    @property
    def namespace(self):
        return self._namespace

    @namespace.setter
    def namespace(self, namespace):
        if namespace is None:
            self._namespace = awkward1.namespace
        else:
            self._namespace = namespace

    @property
    def type(self):
        return self._layout.type

    # @type.setter
    # def type(self, type):
    #     if type is None:
    #         self.__class__ = Record
    #         self._layout.type = None
    #
    #     else:
    #         if not isinstance(type, awkward1.layout.RecordType):
    #             raise TypeError("type must be an awkward1.layout.RecordType")
    #
    #         cls = self._namespace.get(type.parameters.get("__class__"))
    #         if cls is None:
    #             cls = Record
    #         elif not isinstance(cls, __builtins__["type"]) or not issubclass(cls, Record):
    #             raise TypeError("type.parameters['__class__'] = {0} must be a subclass of awkward1.Record".format(repr(type.parameters["__class__"])))
    #
    #         self.__class__ = cls
    #         self._layout.type = type

    def __getitem__(self, where):
        return awkward1._util.wrap(self.layout[where], self._namespace)

    def __str__(self, limit_value=85):
        return awkward1._util.minimally_touching_string(limit_value, self._layout, self._namespace)

    def __repr__(self, limit_value=40, limit_total=85):
        value = awkward1._util.minimally_touching_string(limit_value, self._layout, self._namespace)

        limit_type = limit_total - len(value) - len("<Record  type=>")
        type = repr(str(self.layout.type))
        if len(type) > limit_type:
            type = type[:(limit_type - 4)] + "..." + type[-1]

        return "<Record {0} type={1}>".format(value, type)

class FillableArray(object):
    def __init__(self, namespace=None):
        self._fillablearray = awkward1.layout.FillableArray()
        self.namespace = namespace

    @property
    def namespace(self):
        return self._namespace

    @namespace.setter
    def namespace(self, namespace):
        if namespace is None:
            self._namespace = awkward1.namespace
        else:
            self._namespace = namespace

    @property
    def type(self):
        return awkward1.layout.ArrayType(self._fillablearray.type, len(self._fillablearray))

    def __len__(self):
        return len(self._fillablearray)

    def __getitem__(self, where):
        return awkward1._util.wrap(self._fillablearray[where], self._namespace)

    def __iter__(self):
        for x in self._fillablearray.snapshot():
            yield awkward1._util.wrap(x, self._namespace)

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

    def snapshot(self):
        return awkward1._util.wrap(self._fillablearray.snapshot(), self._namespace)

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

    def endtuple(self):
        self._fillablearray.endtuple()

    def beginrecord(self, name=None):
        self._fillablearray.beginrecord(name)

    def field(self, key):
        self._fillablearray.field(key)

    def endrecord(self):
        self._fillablearray.endrecord()
