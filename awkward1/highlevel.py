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

    def __repr__(self, limit_value=40):
        def forward(array):
            yield "["
            space = ""
            for i in range(len(array)):
                x = array[i]
                if isinstance(x, awkward1.layout.Content):
                    yield space
                    for y in forward(x):
                        yield y
                elif isinstance(x, (float, numpy.floating)):
                    yield space + "{0:3g}".format(x)
                else:
                    yield space + repr(x)
                space = " "
            yield "]"

        def backward(array):
            yield "]"
            space = ""
            for i in range(len(array) - 1, -1, -1):
                x = array[i]
                if isinstance(x, awkward1.layout.Content):
                    yield space
                    for y in backward(x):
                        yield y
                elif isinstance(x, (float, numpy.floating)):
                    yield "{0:3g}".format(x) + space
                else:
                    yield repr(x) + space
                space = " "
            yield "["

        def forever(iterator):
            for x in iterator:
                yield x
            while True:
                yield None

        halfway = len(self.layout) // 2
        left, right = [], []
        leftlen, rightlen = 0, 0
        leftgen = forever(forward(self.layout[:halfway]))
        rightgen = forever(backward(self.layout[halfway:]))
        while True:
            l = next(leftgen)
            if l is not None:
                if leftlen + rightlen + len(l) > limit_value:
                    break
                left.append(l)
                leftlen += len(l)

            r = next(rightgen)
            if r is not None:
                if leftlen + rightlen + len(r) > limit_value:
                    break
                right.append(r)
                rightlen += len(r)

            if l is None and r is None:
                break

        if l is None and r is None:
            value = "".join(left).rstrip("[").rstrip("{").rstrip(" ") + " " + "".join(reversed(right)).lstrip("]").lstrip("}").lstrip(" ")
        else:
            value = "".join(left).rstrip("[").rstrip("{").rstrip(" ") + " ... " + "".join(reversed(right)).lstrip("]").lstrip("}").lstrip(" ")

        return "<Array {0} type={1}>".format(value, repr(str(self.layout.type)))

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
