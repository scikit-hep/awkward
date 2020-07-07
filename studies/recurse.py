def foo1(toptr, fromptr, ndim, shape, strides):
    if ndim == 1:
        for i in range(shape[0]):
            toptr[i] = fromptr[i * strides[0]]
    else:
        for i in range(shape[0]):
            temptoptr = toptr[i * shape[1] :]
            tempfromptr = fromptr[i * strides[0] :]
            err = foo1(temptoptr, fromptr, ndim - 1, shape[1:], strides[1:])
            toptr[i * shape[i] :] = temptoptr
            fromptr[i * strides[0] :] = tempfromptr
            if err != 1:
                return err
    return 1


def foo2(toptr, fromptr, ndim, shape, strides):
    count = 1
    while count < len(shape):
        for i in range(shape[0]):
            if ndim == 1:
                for i in range(shape[0]):
                    toptr[i] = fromptr[i * strides[0]]
                return 1
            else:
                toptr = toptr[i * shape[1] :]
                fromptr = fromptr[i * strides[0] :]
                ndim = ndim - count
                strides = strides[count:]
                shape = shape[count:]
            count += 1
    return 1


def test1():
    toptr = [0] * 10
    fromptr = [1, 2, 3] * 10
    ndim = 2
    shape = [2, 5, 4, 3]
    strides = [3, 4, 1, 2]
    # print(len(toptr))
    foo1(toptr, fromptr, ndim, shape, strides)
    print(toptr)
    # print(len(toptr))


def bar1(tocarry, toindex, fromindex, j, stop, n, replacement):
    while fromindex[j] < stop:
        if replacement:
            for k in range(j + 1, n):
                fromindex[k] = fromindex[j]
        else:
            for k in range(j + 1, n):
                fromindex[k] = fromindex[j] + k - j
        if (j + 1) == n:
            for k in range(n):
                tocarry[k][toindex[k]] = fromindex[k]
                toindex[k] += 1
        else:
            bar1(tocarry, toindex, fromindex, j + 1, stop, n, replacement)
        fromindex[j] += 1

def bar2(tocarry, toindex, fromindex, j, stop, n, replacement):
    for i in range(j, n - 1):
        if replacement:
            for k in range(j + 1, n):
                fromindex[k] = fromindex[j]
        else:
            for k in range(j + 1, n):
                fromindex[k] = fromindex[j] + k - j
        if         

def test2():
    toindex = [0]*10
    tocarry = [[0]*10]*10
    fromindex = [5, 3, 2, 6, 4, 8, 1, 9, 4, 4]
    bar1(tocarry, toindex, fromindex, 0, 7, 9, True)
    print(tocarry)

if __name__ == "__main__":
    test2()
