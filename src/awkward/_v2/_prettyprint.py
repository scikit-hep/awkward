# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import math
import re

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


def horizontal(data, num_cols):
    original_num_cols = num_cols

    if isinstance(data, ak._v2.contents.Content):
        front, back = ["["], ["]"]
        num_cols -= 2

        if len(data) == 0:
            return 2, front + back

        elif len(data) == 1:
            cols_taken, strs = horizontal(data[0], num_cols)
            return 2 + cols_taken, front + strs + back

        else:
            num_cols -= 5  # anticipate the ", ..."
            which = 0
            for forward, index in alternate(len(data)):
                if forward:
                    for_comma = 0 if which == 0 else 2
                    cols_taken, strs = horizontal(data[index], num_cols - for_comma)
                    if num_cols - (for_comma + cols_taken) >= 0:
                        if which != 0:
                            front.append(", ")
                            num_cols -= 2
                        front.extend(strs)
                        num_cols -= cols_taken
                    else:
                        break
                else:
                    cols_taken, strs = horizontal(data[index], num_cols - 2)
                    if num_cols - (2 + cols_taken) >= 0:
                        back[:0] = strs
                        back.insert(0, ", ")
                        num_cols -= 2 + cols_taken
                    else:
                        break

                which += 1

            if which == 0:
                front.append("...")
                num_cols -= 3
            elif which != len(data):
                front.append(", ...")
                num_cols -= 5

            num_cols += 5  # credit the ", ..."
            return original_num_cols - num_cols, front + back

    elif isinstance(data, ak._v2.record.Record):
        is_tuple = data.is_tuple

        front = ["("] if is_tuple else ["{"]
        num_cols -= 2  # both the opening and closing brackets
        num_cols -= 5  # anticipate the ", ..."

        which = 0
        keys = data.keys
        for key in keys:
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

            if num_cols - (for_comma + len(key_str) + 3) >= 0:
                if which != 0:
                    front.append(", ")
                    num_cols -= 2
                front.append(key_str)
                num_cols -= len(key_str)
                which += 1

                target = num_cols if len(keys) == 1 else half(num_cols)
                cols_taken, strs = horizontal(data[key], target)
                if num_cols - cols_taken >= 0:
                    front.extend(strs)
                    num_cols -= cols_taken
                else:
                    front.append("...")
                    num_cols -= 3
                    break

            else:
                break

            which += 1

        if len(keys) != 0:
            if which == 0:
                front.append("...")
                num_cols -= 3
            elif which != 2 * len(keys):
                front.append(", ...")
                num_cols -= 5

        num_cols += 5  # credit the ", ..."
        front.append(")" if is_tuple else "}")
        return original_num_cols - num_cols, front

    else:
        out = str(data)
        return len(out), [out]


def pretty(data, num_rows, num_cols):
    if num_rows <= 1:
        _, strs = horizontal(data, num_cols)
        return "".join(strs)

    elif isinstance(data, ak._v2.contents.Content):
        front, back = [], []
        which = 0
        for forward, index in alternate(len(data)):
            _, strs = horizontal(data[index], num_cols - 2)
            if forward:
                front.append("".join(strs))
            else:
                back.insert(0, "".join(strs))

            which += 1
            if which >= num_rows:
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

    elif isinstance(data, ak._v2.record.Record):
        front = []
        which = 0
        keys = data.keys
        for key in keys:
            raise NotImplementedError("deal with key_str")

            _, strs = horizontal(data[key], num_cols - 2)
            front.append("".join(strs))

            which += 1
            if which >= num_rows:
                break

        if len(keys) != 0 and which != len(data):
            back[0] = "..."

        out = front + back
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
