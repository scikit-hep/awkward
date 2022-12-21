# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import math
import numbers
import re

import awkward as ak

numpy = ak._nplikes.Numpy.instance()


def half(integer):
    return int(math.ceil(integer / 2))


def alternate(length):
    halfindex = half(length)
    forward = iter(range(halfindex))
    backward = iter(range(length - 1, halfindex - 1, -1))
    going_forward, going_backward = True, True
    while going_forward or going_backward:
        if going_forward:
            try:
                yield True, next(forward)
            except StopIteration:
                going_forward = False
        if going_backward:
            try:
                yield False, next(backward)
            except StopIteration:
                going_backward = False


is_identifier = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*$")


# avoid recursion in which ak.Array.__getitem__ calls prettyprint
# to form an error string: private reimplementation of ak.Array.__getitem__


def get_at(data, index):
    out = data._layout._getitem_at(index)
    if isinstance(out, ak.contents.NumpyArray):
        array_param = out.parameter("__array__")
        if array_param == "byte":
            return ak._util.tobytes(out._raw(numpy))
        elif array_param == "char":
            return ak._util.tobytes(out._raw(numpy)).decode(errors="surrogateescape")
    if isinstance(out, (ak.contents.Content, ak.record.Record)):
        return ak._util.wrap(out, data._behavior)
    else:
        return out


def get_field(data, field):
    out = data._layout._getitem_field(field)
    if isinstance(out, ak.contents.NumpyArray):
        array_param = out.parameter("__array__")
        if array_param == "byte":
            return ak._util.tobytes(out._raw(numpy))
        elif array_param == "char":
            return ak._util.tobytes(out._raw(numpy)).decode(errors="surrogateescape")
    if isinstance(out, (ak.contents.Content, ak.record.Record)):
        return ak._util.wrap(out, data._behavior)
    else:
        return out


def custom_str(current):
    if (
        issubclass(type(current), ak.highlevel.Record)
        and not type(current).__str__ is ak.highlevel.Record.__str__
    ) or (
        issubclass(type(current), ak.highlevel.Array)
        and not type(current).__str__ is ak.highlevel.Array.__str__
    ):
        return str(current)

    elif (
        issubclass(type(current), ak.highlevel.Record)
        and not type(current).__repr__ is ak.highlevel.Record.__repr__
    ) or (
        issubclass(type(current), ak.highlevel.Array)
        and not type(current).__repr__ is ak.highlevel.Array.__repr__
    ):
        return repr(current)

    else:
        return None


def valuestr_horiz(data, limit_cols):
    original_limit_cols = limit_cols

    if isinstance(data, ak.highlevel.Array):
        front, back = ["["], ["]"]
        limit_cols -= 2

        if len(data) == 0:
            return 2, front + back

        elif len(data) == 1:
            cols_taken, strs = valuestr_horiz(get_at(data, 0), limit_cols)
            return 2 + cols_taken, front + strs + back

        else:
            limit_cols -= 5  # anticipate the ", ..."
            which = 0
            for forward, index in alternate(len(data)):
                current = get_at(data, index)
                if forward:
                    for_comma = 0 if which == 0 else 2
                    cols_taken, strs = valuestr_horiz(current, limit_cols - for_comma)

                    custom = custom_str(current)
                    if custom is not None:
                        strs = custom

                    if limit_cols - (for_comma + cols_taken) >= 0:
                        if which != 0:
                            front.append(", ")
                            limit_cols -= 2
                        front.extend(strs)
                        limit_cols -= cols_taken
                    else:
                        break
                else:
                    cols_taken, strs = valuestr_horiz(current, limit_cols - 2)

                    custom = custom_str(current)
                    if custom is not None:
                        strs = custom

                    if limit_cols - (2 + cols_taken) >= 0:
                        back[:0] = strs
                        back.insert(0, ", ")
                        limit_cols -= 2 + cols_taken
                    else:
                        break

                which += 1

            if which == 0:
                front.append("...")
                limit_cols -= 3
            elif which != len(data):
                front.append(", ...")
                limit_cols -= 5

            limit_cols += 5  # credit the ", ..."
            return original_limit_cols - limit_cols, front + back

    elif isinstance(data, ak.highlevel.Record):
        is_tuple = data.layout.is_tuple

        front = ["("] if is_tuple else ["{"]
        limit_cols -= 2  # both the opening and closing brackets
        limit_cols -= 5  # anticipate the ", ..."

        which = 0
        fields = data.fields
        for key in fields:
            for_comma = 0 if which == 0 else 2
            if is_tuple:
                key_str = ""
            else:
                if is_identifier.match(key) is None:
                    key_str = repr(key) + ": "
                    if key_str.startswith("u"):
                        key_str = key_str[1:]
                else:
                    key_str = key + ": "

            if limit_cols - (for_comma + len(key_str) + 3) >= 0:
                if which != 0:
                    front.append(", ")
                    limit_cols -= 2
                front.append(key_str)
                limit_cols -= len(key_str)
                which += 1

                target = limit_cols if len(fields) == 1 else half(limit_cols)
                cols_taken, strs = valuestr_horiz(get_field(data, key), target)
                if limit_cols - cols_taken >= 0:
                    front.extend(strs)
                    limit_cols -= cols_taken
                else:
                    front.append("...")
                    limit_cols -= 3
                    break

            else:
                break

            which += 1

        if len(fields) != 0:
            if which == 0:
                front.append("...")
                limit_cols -= 3
            elif which != 2 * len(fields):
                front.append(", ...")
                limit_cols -= 5

        limit_cols += 5  # credit the ", ..."
        front.append(")" if is_tuple else "}")
        return original_limit_cols - limit_cols, front

    else:
        if isinstance(data, (str, bytes)):
            out = repr(data)
        elif isinstance(data, numbers.Integral):
            out = str(data)
        elif isinstance(data, numbers.Real):
            out = f"{data:.3g}"
        elif isinstance(data, numbers.Complex):
            out = f"{data.real:.2g}+{data.imag:.2g}j"
        else:
            out = str(data)

        return len(out), [out]


def valuestr(data, limit_rows, limit_cols):
    if (
        isinstance(data, (ak.highlevel.Array, ak.highlevel.Record))
        and not data.layout.backend.nplike.known_data
    ):
        data.layout._touch_data(recursive=True)

    if limit_rows <= 1:
        _, strs = valuestr_horiz(data, limit_cols)
        return "".join(strs)

    elif isinstance(data, ak.highlevel.Array):
        front, back = [], []
        which = 0
        for forward, index in alternate(len(data)):
            _, strs = valuestr_horiz(get_at(data, index), limit_cols - 2)
            if forward:
                front.append("".join(strs))
            else:
                back.insert(0, "".join(strs))

            which += 1
            if which >= limit_rows:
                break

        if len(data) != 0 and which != len(data):
            back[0] = "..."

        out = front + back
        for i, val in enumerate(out):
            if i > 0:
                val = out[i] = " " + val
            else:
                val = out[i] = "[" + val
            if i < len(out) - 1:
                out[i] = val + ","
            else:
                out[i] = val + "]"

        return "\n".join(out)

    elif isinstance(data, ak.highlevel.Record):
        is_tuple = data.layout.is_tuple

        front = []

        which = 0
        fields = data.fields
        for key in fields:
            if is_tuple:
                key_str = ""
            else:
                if is_identifier.match(key) is None:
                    key_str = repr(key) + ": "
                    if key_str.startswith("u"):
                        key_str = key_str[1:]
                else:
                    key_str = key + ": "
            _, strs = valuestr_horiz(
                get_field(data, key), limit_cols - 2 - len(key_str)
            )
            front.append(key_str + "".join(strs))

            which += 1
            if which >= limit_rows:
                break

        if len(fields) != 0 and which != len(fields):
            front[-1] = "..."

        out = front
        for i, val in enumerate(out):
            if i > 0:
                val = out[i] = " " + val
            elif data.is_tuple:
                val = out[i] = "(" + val
            else:
                val = out[i] = "{" + val
            if i < len(out) - 1:
                out[i] = val + ","
            elif data.is_tuple:
                out[i] = val + ")"
            else:
                out[i] = val + "}"
        return "\n".join(out)

    else:
        raise ak._errors.wrap_error(AssertionError(type(data)))
