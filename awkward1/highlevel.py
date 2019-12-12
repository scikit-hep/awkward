# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import numpy

import awkward1.layout
import awkward1.operations.convert

class Array(object):
    def __init__(self, data, type=None, namespace=None):
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
        if type is not None:
            self.type = type
        else:
            t = self.layout.type.nolength()
            if t.parameters.get("__class__") in self._namespace:
                self.__class__ = self._namespace[t.parameters["__class__"]]

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
        return self._layout.type

    @type.setter
    def type(self, type):
        if not isinstance(type, awkward1.layout.Type):
            raise TypeError("type must be a subclass of awkward1.layout.Type")
        t = type.nolength()
        if t.parameters.get("__class__") in self._namespace:
            self.__class__ = self._namespace[t.parameters["__class__"]]
        self._layout.type = type

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
    def baretype(self):
        return self._layout.baretype

    def __iter__(self):
        for x in self.layout:
            yield awkward1._util.wrap(x, self._namespace)

    def __str__(self, limit_value=85):
        if len(self) == 0:
            return "[]"

        def forward(x, space, brackets=True, wrap=True):
            done = False
            if wrap and isinstance(x, awkward1.layout.Content):
                t = x.type.nolength()
                if t.parameters.get("__class__") in self._namespace:
                    y = self._namespace[t.parameters["__class__"]](x, namespace=self._namespace)
                    if "__repr__" in type(y).__dict__:
                        yield space + repr(y)
                        done = True
            if not done:
                if isinstance(x, awkward1.layout.Content):
                    if brackets:
                        yield space + "["
                    sp = ""
                    for i in range(len(x)):
                        for token in forward(x[i], sp):
                            yield token
                        sp = ", "
                    if brackets:
                        yield "]"
                elif isinstance(x, awkward1.layout.Record):
                    yield space + "{"
                    sp = ""
                    for k in x.keys():
                        key = sp + k + ": "
                        for token in forward(x[k], ""):
                            yield key + token
                            key = ""
                        sp = ", "
                    yield "}"
                elif isinstance(x, (float, numpy.floating)):
                    yield space + "{0:.3g}".format(x)
                else:
                    yield space + repr(x)

        def backward(x, space, brackets=True, wrap=True):
            done = False
            if wrap and isinstance(x, awkward1.layout.Content):
                t = x.type.nolength()
                if t.parameters.get("__class__") in self._namespace:
                    y = self._namespace[t.parameters["__class__"]](x, namespace=self._namespace)
                    if "__repr__" in type(y).__dict__:
                        yield repr(y) + space
                        done = True
            if not done:
                if isinstance(x, awkward1.layout.Content):
                    if brackets:
                        yield "]" + space
                    sp = ""
                    for i in range(len(x) - 1, -1, -1):
                        for token in backward(x[i], sp):
                            yield token
                        sp = ", "
                    if brackets:
                        yield "["
                elif isinstance(x, awkward1.layout.Record):
                    yield "}" + space
                    keys = x.keys()
                    for i in range(len(keys) - 1, -1, -1):
                        last = None
                        for token in backward(x[keys[i]], ""):
                            if last is not None:
                                yield last
                            last = token
                        if last is not None:
                            yield keys[i] + ": " + last
                        if i != 0:
                            yield ", "
                    yield "{"
                elif isinstance(x, (float, numpy.floating)):
                    yield "{0:.3g}".format(x) + space
                else:
                    yield repr(x) + space

        def forever(iterable):
            for token in iterable:
                yield token
            while True:
                yield None

        halfway = len(self.layout) // 2
        left, right = ["["], ["]"]
        leftlen, rightlen = 1, 1
        leftgen = forever(forward(self.layout[:halfway], "", brackets=False, wrap=False))
        rightgen = forever(backward(self.layout[halfway:], "", brackets=False, wrap=False))
        while True:
            l = next(leftgen)
            if l is not None:
                if leftlen + rightlen + len(l) + (2 if l is None and r is None else 6) > limit_value:
                    break
                left.append(l)
                leftlen += len(l)

            r = next(rightgen)
            if r is not None:
                if leftlen + rightlen + len(r) + (2 if l is None and r is None else 6) > limit_value:
                    break
                right.append(r)
                rightlen += len(r)

            if l is None and r is None:
                break

        while len(left) > 1 and (left[-1] == "[" or left[-1] == ", [" or left[-1] == "{" or left[-1] == ", {" or left[-1] == ", "):
            left.pop()
            l = ""
        while len(right) > 1 and (right[-1] == "]" or right[-1] == "], " or right[-1] == "}" or right[-1] == "}, " or right[-1] == ", "):
            right.pop()
            r = ""
        if l is None and r is None:
            if len(left) == 0:
                return "".join(reversed(right)).lstrip(" ")
            else:
                return "".join(left).rstrip(" ") + ", " + "".join(reversed(right)).lstrip(" ")
        else:
            if len(left) == 0 and len(right) == 0:
                return "..."
            elif len(left) == 0:
                return "... " + "".join(reversed(right)).lstrip(" ")
            else:
                return "".join(left).rstrip(" ") + ", ... " + "".join(reversed(right)).lstrip(" ")

    def __repr__(self, limit_value=40, limit_total=85):
        value = self.__str__(limit_value=limit_value)

        limit_type = limit_total - len(value) - len("<Array  type=>")
        type = repr(str(self.layout.type))
        if len(type) > limit_type:
            type = type[:(limit_type - 4)] + "..." + type[-1]

        return "<Array {0} type={1}>".format(value, type)

    def __len__(self):
        return len(self.layout)

    def __getitem__(self, where):
        return awkward1._util.wrap(self.layout[where], self._namespace)

class Record(object):
    def __init__(self, data, type=None):
        # FIXME: more checks here
        layout = data
        if not isinstance(layout, awkward1.layout.Record):
            raise TypeError("could not convert data into an awkward1.Record")
        self.layout = layout
        if type is not None:
            self.type = type

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

    @type.setter
    def type(self, type):
        if not isinstance(type, awkward1.layout.Type):
            raise TypeError("type must be a subclass of awkward1.layout.Type")
        t = type.nolength()
        if t.parameters.get("__class__") in self._namespace:
            self.__class__ = self._namespace[t.parameters["__class__"]]
        self._layout.type = type

    @property
    def baretype(self):
        return self._layout.baretype
