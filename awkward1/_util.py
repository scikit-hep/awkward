# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import inspect
import numbers
import re

import numpy

import awkward1.layout

unknowntypes = (awkward1.layout.EmptyArray,)

indexedtypes = (awkward1.layout.IndexedArray32, awkward1.layout.IndexedArrayU32, awkward1.layout.IndexedArray64)

uniontypes = (awkward1.layout.UnionArray8_32, awkward1.layout.UnionArray8_U32, awkward1.layout.UnionArray8_64)

optiontypes = (awkward1.layout.IndexedOptionArray32, awkward1.layout.IndexedOptionArray64)

listtypes = (awkward1.layout.RegularArray, awkward1.layout.ListArray32, awkward1.layout.ListArrayU32, awkward1.layout.ListArray64, awkward1.layout.ListOffsetArray32, awkward1.layout.ListOffsetArrayU32, awkward1.layout.ListOffsetArray64)

recordtypes = (awkward1.layout.RecordArray,)

def regular_classes(classes):
    import awkward1
    if classes is None:
        return awkward1.classes
    else:
        return classes

def regular_functions(functions):
    import awkward1
    if functions is None:
        return awkward1.functions
    else:
        return functions

def wrap(content, classes, functions):
    import awkward1.layout

    if isinstance(content, awkward1.layout.Content):
        cls = regular_classes(classes).get(content.parameters.get("__class__"))
        if cls is None or (isinstance(cls, type) and not issubclass(cls, awkward1.Array)):
            cls = awkward1.Array
        return cls(content, classes=classes, functions=functions)

    elif isinstance(content, awkward1.layout.Record):
        cls = regular_classes(classes).get(content.parameters.get("__class__"))
        if cls is None or (isinstance(cls, type) and not issubclass(cls, awkward1.Record)):
            cls = awkward1.Record
        return cls(content, classes=classes, functions=functions)

    else:
        return content

def called_by_module(modulename):
    frame = inspect.currentframe()
    while frame is not None:
        name = getattr(inspect.getmodule(frame), '__name__', None)
        if name is not None and (name == modulename or name.startswith(modulename + ".")):
            return True
        frame = frame.f_back
    return True

def key2index(keys, key):
    if keys is None:
        attempt = None
    else:
        try:
            attempt = keys.index(key)
        except ValueError:
            attempt = None

    if attempt is None:
        m = key2index._pattern.match(key)
        if m is not None:
            attempt = m.group(0)

    if attempt is None:
        raise ValueError("key {0} not found in record".format(repr(key)))
    else:
        return attempt

key2index._pattern = re.compile(r"^[1-9][0-9]*$")

def minimally_touching_string(limit_length, layout, classes, functions):
    import awkward1.layout

    if isinstance(layout, awkward1.layout.Record):
        layout = layout.array[layout.at : layout.at + 1]

    if len(layout) == 0:
        return "[]"

    def forward(x, space, brackets=True, wrap=True):
        done = False
        if wrap and isinstance(x, awkward1.layout.Content):
            cls = regular_classes(classes).get(x.parameters.get("__class__"))
            if cls is not None and isinstance(cls, type) and issubclass(cls, awkward1.Array):
                y = cls(x, classes=classes, functions=functions)
                if "__repr__" in type(y).__dict__:
                    yield space + repr(y)
                    done = True
        if wrap and isinstance(x, awkward1.layout.Record):
            cls = regular_classes(classes).get(x.parameters.get("__class__"))
            if cls is not None and isinstance(cls, type) and issubclass(cls, awkward1.Record):
                y = cls(x, classes=classes, functions=functions)
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
            elif isinstance(x, awkward1.layout.Record) and x.istuple:
                yield space + "("
                sp = ""
                for i in range(x.numfields):
                    key = sp
                    for token in forward(x[str(i)], ""):
                        yield key + token
                        key = ""
                    sp = ", "
                yield ")"
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
            cls = regular_classes(classes).get(x.parameters.get("__class__"))
            if cls is not None and isinstance(cls, type) and issubclass(cls, awkward1.Array):
                y = cls(x, classes=classes, functions=functions)
                if "__repr__" in type(y).__dict__:
                    yield repr(y) + space
                    done = True
        if wrap and isinstance(x, awkward1.layout.Record):
            cls = regular_classes(classes).get(x.parameters.get("__class__"))
            if cls is not None and isinstance(cls, type) and issubclass(cls, awkward1.Record):
                y = cls(x, classes=classes, functions=functions)
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
            elif isinstance(x, awkward1.layout.Record) and x.istuple:
                yield ")" + space
                for i in range(x.numfields - 1, -1, -1):
                    last = None
                    for token in backward(x[str(i)], ""):
                        if last is not None:
                            yield last
                        last = token
                    if last is not None:
                        yield last
                    if i != 0:
                        yield ", "
                yield "("
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

    halfway = len(layout) // 2
    left, right = ["["], ["]"]
    leftlen, rightlen = 1, 1
    leftgen = forever(forward(layout[:halfway], "", brackets=False, wrap=False))
    rightgen = forever(backward(layout[halfway:], "", brackets=False, wrap=False))
    while True:
        l = next(leftgen)
        if l is not None:
            if leftlen + rightlen + len(l) + (2 if l is None and r is None else 6) > limit_length:
                break
            left.append(l)
            leftlen += len(l)

        r = next(rightgen)
        if r is not None:
            if leftlen + rightlen + len(r) + (2 if l is None and r is None else 6) > limit_length:
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
        if left == ["["]:
            return "[" + "".join(reversed(right)).lstrip(" ")
        else:
            return "".join(left).rstrip(" ") + ", " + "".join(reversed(right)).lstrip(" ")
    else:
        if left == ["["] and right == ["]"]:
            return "[...]"
        elif left == ["["]:
            return "[... " + "".join(reversed(right)).lstrip(" ")
        else:
            return "".join(left).rstrip(" ") + ", ... " + "".join(reversed(right)).lstrip(" ")
