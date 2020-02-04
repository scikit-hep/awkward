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

def combine_classes(arrays):
    classes = None
    for x in arrays:
        if isinstance(x, (awkward1.highlevel.Array, awkward1.highlevel.Record, awkward1.highlevel.FillableArray)) and x._classes is not None:
            if classes is None:
                classes = dict(x._classes)
            else:
                classes.update(x._classes)
    return classes

def combine_functions(arrays):
    functions = None
    for x in arrays:
        if isinstance(x, (awkward1.highlevel.Array, awkward1.highlevel.Record, awkward1.highlevel.FillableArray)) and x._functions is not None:
            if functions is None:
                functions = dict(x._functions)
            else:
                functions.update(x._functions)
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

def extra(args, kwargs, defaults):
    out = []
    for i in range(len(defaults)):
        name, default = defaults[i]
        if i < len(args):
            out.append(args[i])
        elif name in kwargs:
            out.append(kwargs[name])
        else:
            out.append(default)
    return out

def called_by_module(modulename):
    frame = inspect.currentframe()
    while frame is not None:
        name = getattr(inspect.getmodule(frame), "__name__", None)
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

def broadcast_and_apply(inputs, getfunction):
    def checklength(inputs):
        length = len(inputs[0])
        for x in inputs[1:]:
            if len(x) != length:
                raise ValueError("cannot broadcast {0} of length {1} with {2} of length {3}".format(type(inputs[0]).__name__, length, type(x).__name__, len(x)))

    def apply(inputs):
        # handle implicit right-broadcasting (i.e. NumPy-like)
        if any(isinstance(x, listtypes) for x in inputs):
            maxdepth = max(x.purelist_depth for x in inputs if isinstance(x, awkward1.layout.Content))
            if maxdepth > 0 and all(x.purelist_isregular for x in inputs if isinstance(x, awkward1.layout.Content)):
                nextinputs = []
                for x in inputs:
                    if isinstance(x, awkward1.layout.Content):
                        while x.purelist_depth < maxdepth:
                            x = awkward1.layout.RegularArray(x, 1)
                    nextinputs.append(x)
                if any(x is not y for x, y in zip(inputs, nextinputs)):
                    return apply(nextinputs)

        # now all lengths must agree
        checklength([x for x in inputs if isinstance(x, awkward1.layout.Content)])

        function = getfunction(inputs)

        # the rest of this is one switch statement
        if function is not None:
            return function()

        elif any(isinstance(x, unknowntypes) for x in inputs):
            return apply([x if not isinstance(x, unknowntypes) else awkward1.layout.NumpyArray(numpy.array([], dtype=numpy.int64)) for x in inputs])

        elif any(isinstance(x, awkward1.layout.NumpyArray) and x.ndim > 1 for x in inputs):
            return apply([x if not (isinstance(x, awkward1.layout.NumpyArray) and x.ndim > 1) else x.regularize_shape() for x in inputs])

        elif any(isinstance(x, indexedtypes) for x in inputs):
            return apply([x if not isinstance(x, indexedtypes) else x.project() for x in inputs])

        elif any(isinstance(x, uniontypes) for x in inputs):
            tagslist = []
            length = None
            for x in inputs:
                if isinstance(x, uniontypes):
                    tagslist.append(numpy.asarray(x.tags))
                    if length is None:
                        length = len(tagslist[-1])
                    elif length != len(tagslist[-1]):
                        raise ValueError("cannot broadcast UnionArray of length {0} with UnionArray of length {1}".format(length, len(tagslist[-1])))

            combos = numpy.stack(tagslist, axis=-1)
            combos = combos.view([(str(i), combos.dtype) for i in range(len(tagslist))]).reshape(length)

            tags = numpy.empty(length, dtype=numpy.int8)
            index = numpy.empty(length, dtype=numpy.int64)
            contents = []
            for tag, combo in enumerate(numpy.unique(combos)):
                mask = (combos == combo)
                tags[mask] = tag
                index[mask] = numpy.arange(numpy.count_nonzero(mask))
                nextinputs = []
                for i, x in enumerate(inputs):
                    if isinstance(x, uniontypes):
                        nextinputs.append(x[mask].project(combo[str(i)]))
                    elif isinstance(x, awkward1.layout.Content):
                        nextinputs.append(x[mask])
                    else:
                        nextinputs.append(x)
                contents.append(apply(nextinputs))

            tags = awkward1.layout.Index8(tags)
            index = awkward1.layout.Index64(index)
            return awkward1.layout.UnionArray8_64(tags, index, contents)

        elif any(isinstance(x, optiontypes) for x in inputs):
            mask = None
            for x in inputs:
                if isinstance(x, (awkward1.layout.IndexedOptionArray32, awkward1.layout.IndexedOptionArray64)):
                    m = numpy.asarray(x.index) < 0
                    if mask is None:
                        mask = m
                    else:
                        numpy.bitwise_or(mask, m, out=mask)

            nextmask = awkward1.layout.Index8(mask.view(numpy.int8))
            index = numpy.full(len(mask), -1, dtype=numpy.int64)
            index[~mask] = numpy.arange(len(mask) - numpy.count_nonzero(mask), dtype=numpy.int64)
            index = awkward1.layout.Index64(index)
            if any(not isinstance(x, optiontypes) for x in inputs):
                nextindex = numpy.arange(len(mask), dtype=numpy.int64)
                nextindex[mask] = -1
                nextindex = awkward1.layout.Index64(nextindex)

            nextinputs = []
            for x in inputs:
                if isinstance(x, optiontypes):
                    nextinputs.append(x.project(nextmask))
                else:
                    nextinputs.append(awkward1.layout.IndexedOptionArray64(nextindex, x).project(nextmask))

            return awkward1.layout.IndexedOptionArray64(index, apply(nextinputs))

        elif any(isinstance(x, listtypes) for x in inputs):
            if all(isinstance(x, awkward1.layout.RegularArray) or not isinstance(x, listtypes) for x in inputs):
                maxsize = max([x.size for x in inputs if isinstance(x, awkward1.layout.RegularArray)])
                for x in inputs:
                    if isinstance(x, awkward1.layout.RegularArray):
                        if maxsize > 1 and x.size == 1:
                            tmpindex = awkward1.layout.Index64(numpy.repeat(numpy.arange(len(x), dtype=numpy.int64), maxsize))
                nextinputs = []
                for x in inputs:
                    if isinstance(x, awkward1.layout.RegularArray):
                        if maxsize > 1 and x.size == 1:
                            nextinputs.append(awkward1.layout.IndexedArray64(tmpindex, x.content).project())
                        elif x.size == maxsize:
                            nextinputs.append(x.content)
                        else:
                            raise ValueError("cannot broadcast RegularArray of size {0} with RegularArray of size {1}".format(x.size, maxsize))
                    else:
                        nextinputs.append(x)
                return awkward1.layout.RegularArray(apply(nextinputs), maxsize)

            else:
                for x in inputs:
                    if isinstance(x, listtypes) and not isinstance(x, awkward1.layout.RegularArray):
                        first = x
                        break
                offsets = first.compact_offsets64()
                nextinputs = []
                for x in inputs:
                    if isinstance(x, listtypes):
                        nextinputs.append(x.broadcast_tooffsets64(offsets).content)
                    # handle implicit left-broadcasting (unlike NumPy)
                    elif isinstance(x, awkward1.layout.Content):
                        nextinputs.append(awkward1.layout.RegularArray(x, 1).broadcast_tooffsets64(offsets).content)
                    else:
                        nextinputs.append(x)
                return awkward1.layout.ListOffsetArray64(offsets, apply(nextinputs))

        elif any(isinstance(x, recordtypes) for x in inputs):
            keys = None
            length = None
            istuple = True
            for x in inputs:
                if isinstance(x, recordtypes):
                    if keys is None:
                        keys = x.keys()
                    elif set(keys) != set(x.keys()):
                        raise ValueError("cannot broadcast records because keys don't match:\n    {0}\n    {1}".format(", ".join(sorted(keys)), ", ".join(sorted(x.keys()))))
                    if length is None:
                        length = len(x)
                    elif length != len(x):
                        raise ValueError("cannot broadcast RecordArray of length {0} with RecordArray of length {1}".format(length, len(x)))
                    if not x.istuple:
                        istuple = False

            if len(keys) == 0:
                return awkward1.layout.RecordArray(length, istuple)
            else:
                contents = []
                for key in keys:
                    contents.append(apply([x if not isinstance(x, recordtypes) else x[key] for x in inputs]))
                return awkward1.layout.RecordArray(contents, keys)

        else:
            raise ValueError("cannot broadcast: {0}".format(", ".join(type(x) for x in inputs)))

    isscalar = []

    def pack(inputs):
        maxlen = -1
        for x in inputs:
            if isinstance(x, awkward1.layout.Content):
                maxlen = max(maxlen, len(x))
        if maxlen < 0:
            maxlen = 1
        nextinputs = []
        for x in inputs:
            if isinstance(x, awkward1.layout.Record):
                index = numpy.full(maxlen, x.at, dtype=numpy.int64)
                nextinputs.append(awkward1.layout.RegularArray(x.array[index], maxlen))
                isscalar.append(True)
            elif isinstance(x, awkward1.layout.Content):
                nextinputs.append(awkward1.layout.RegularArray(x, len(x)))
                isscalar.append(False)
            else:
                nextinputs.append(x)
                isscalar.append(True)
        return nextinputs

    def unpack(x):
        if all(isscalar):
            if len(x) == 0:
                return x.getitem_nothing().getitem_nothing()
            else:
                return x[0][0]
        else:
            if len(x) == 0:
                return x.getitem_nothing()
            else:
                return x[0]

    return unpack(apply(pack(inputs)))

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
