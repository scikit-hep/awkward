import math


def calc_threads(length):
    if length > 1024:
        return (1024, 1, 1)
    else:
        return (length, 1, 1)


def calc_blocks(length):
    if length > 1024:
        return (math.ceil(length / 1024), 1, 1)
    else:
        return (1, 1, 1)


def success():
    pass
