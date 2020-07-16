def foo(toptr, fromptr, ndim, shape, strides):
    if ndim == 1:
        for i in range(shape[0]):
            toptr[i] = fromptr[i*strides[0]]
    else:
        for i in range(shape[0]):
            err = foo(toptr[i*shape[1]:], fromptr[i*strides[0]:], ndim - 1, shape[1:], strides[1:])
            if err != 1:
                return err
    return 1
