// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     from awkward._connect.cuda._reducers import reduce_with_cupy_at
//     (toptr, fromptr, parents, lenparents,
//     outlength, _invocation_index, _err_code) = args
//     identity = fromptr.dtype.type(1)
//     reduce_with_cupy_at(
//         cupy.multiply,
//         toptr[:outlength],
//         fromptr[:lenparents],
//         parents[:lenparents],
//         identity
//     )
// END PYTHON
