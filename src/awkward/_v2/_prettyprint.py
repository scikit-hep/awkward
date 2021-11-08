# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import math
import re
import numbers

import awkward as ak


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


def valuestr_horiz(data, limit_cols):
    original_limit_cols = limit_cols

    if isinstance(data, ak._v2.highlevel.Array):
        front, back = ["["], ["]"]
        limit_cols -= 2

        if len(data) == 0:
            return 2, front + back

        elif len(data) == 1:
            cols_taken, strs = valuestr_horiz(data[0], limit_cols)
            return 2 + cols_taken, front + strs + back

        else:
            limit_cols -= 5  # anticipate the ", ..."
            which = 0
            for forward, index in alternate(len(data)):
                if forward:
                    for_comma = 0 if which == 0 else 2
                    cols_taken, strs = valuestr_horiz(
                        data[index], limit_cols - for_comma
                    )
                    if limit_cols - (for_comma + cols_taken) >= 0:
                        if which != 0:
                            front.append(", ")
                            limit_cols -= 2
                        front.extend(strs)
                        limit_cols -= cols_taken
                    else:
                        break
                else:
                    cols_taken, strs = valuestr_horiz(data[index], limit_cols - 2)
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

    elif isinstance(data, ak._v2.highlevel.Record):
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
                cols_taken, strs = valuestr_horiz(data[key], target)
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
            out = "{0:.3g}".format(data)
        elif isinstance(data, numbers.Complex):
            out = "{0:.2g}+{1:.2g}j".format(data.real, data.imag)
        else:
            out = str(data)

        return len(out), [out]


def valuestr(data, limit_rows, limit_cols):
    if limit_rows <= 1:
        _, strs = valuestr_horiz(data, limit_cols)
        return "".join(strs)

    elif isinstance(data, ak._v2.highlevel.Array):
        front, back = [], []
        which = 0
        for forward, index in alternate(len(data)):
            _, strs = valuestr_horiz(data[index], limit_cols - 2)
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
        for i in range(len(out)):
            if i > 0:
                out[i] = " " + out[i]
            else:
                out[i] = "[" + out[i]
            if i < len(out) - 1:
                out[i] = out[i] + ","
            else:
                out[i] = out[i] + "]"
        return "\n".join(out)

    elif isinstance(data, ak._v2.highlevel.Record):
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

            _, strs = valuestr_horiz(data[key], limit_cols - 2 - len(key_str))
            front.append(key_str + "".join(strs))

            which += 1
            if which >= limit_rows:
                break

        if len(fields) != 0 and which != len(fields):
            front[-1] = "..."

        out = front
        for i in range(len(out)):
            if i > 0:
                out[i] = " " + out[i]
            elif data.is_tuple:
                out[i] = "(" + out[i]
            else:
                out[i] = "{" + out[i]
            if i < len(out) - 1:
                out[i] = out[i] + ","
            elif data.is_tuple:
                out[i] = out[i] + ")"
            else:
                out[i] = out[i] + "}"
        return "\n".join(out)

    else:
        raise AssertionError(type(data))
