# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import numpy

import awkward1.layout
import awkward1.operations.convert

class Array(object):
    def __init__(self, data, type=None, copy=False):
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

    def __str__(self, limit_value=85):
        def forward(x, space, brackets=True):
            if isinstance(x, awkward1.layout.Content):
                if brackets:
                    yield space + "["
                sp = ""
                for i in range(len(x)):
                    for token in forward(x[i], sp):
                        yield token
                    sp = " "
                if brackets:
                    yield "]"
            elif isinstance(x, awkward1.layout.Record):
                yield space + "{"
                sp = ""
                for k in x.keys():
                    yield sp + k + ": "
                    for token in forward(x[k], ""):
                        yield token
                    sp = ", "
                yield "}"
            elif isinstance(x, (float, numpy.floating)):
                yield space + "{0:.3g}".format(x)
            else:
                yield space + repr(x)

        def backward(x, space, brackets=True):
            if isinstance(x, awkward1.layout.Content):
                if brackets:
                    yield "]" + space
                sp = ""
                for i in range(len(x) - 1, -1, -1):
                    for token in backward(x[i], sp):
                        yield token
                    sp = " "
                if brackets:
                    yield "["
            elif isinstance(x, awkward1.layout.Record):
                yield "}" + space
                sp = ""
                for k in reversed(x.keys()):
                    last = None
                    for token in backward(x[k], ""):
                        if last is not None:
                            yield last
                        last = token
                    if last is not None:
                        yield last + sp
                    yield k + ": "
                    sp = ", "
                yield "{"
            elif isinstance(x, (float, numpy.floating)):
                yield "{0:.3g}".format(x) + space
            else:
                yield repr(x) + space

        def forever(iterator):
            for x in iterator:
                yield x
            while True:
                yield None

        halfway = len(self.layout) // 2
        left, right = ["["], ["]"]
        leftlen, rightlen = 1, 1
        leftgen = forever(forward(self.layout[:halfway], "", brackets=False))
        rightgen = forever(backward(self.layout[halfway:], "", brackets=False))
        while True:
            l = next(leftgen)
            if l is not None:
                if leftlen + rightlen + len(l) + (1 if l is None and r is None else 5) > limit_value:
                    break
                left.append(l)
                leftlen += len(l)

            r = next(rightgen)
            if r is not None:
                if leftlen + rightlen + len(r) + (1 if l is None and r is None else 5) > limit_value:
                    break
                right.append(r)
                rightlen += len(r)

            if l is None and r is None:
                break

        if l is None and r is None:
            return "".join(left).rstrip("[").rstrip("{").rstrip(" ") + " " + "".join(reversed(right)).lstrip("]").lstrip("}").lstrip(" ")
        else:
            return "".join(left).rstrip("[").rstrip("{").rstrip(" ") + " ... " + "".join(reversed(right)).lstrip("]").lstrip("}").lstrip(" ")

    def __repr__(self, limit_value=40, limit_total=85):
        value = self.__str__(limit_value=limit_value)

        limit_type = limit_total - len(value) - len("<Array  type=>")
        type = repr(str(self.layout.type))
        if len(type) > limit_type:
            type = type[:(limit_type - 4)] + "...'"

        return "<Array {0} type={1}>".format(value, type)

    def __len__(self):
        return len(self.layout)

    def __getitem__(self, where):
        layout = self.layout[where]
        if isinstance(layout, awkward1.layout.Content):
            return awkward1.Array(layout)
        elif isinstance(layout, awkward1.layout.Record):
            return awkward1.Record(layout)
        else:
            return layout

class Record(object):
    pass
