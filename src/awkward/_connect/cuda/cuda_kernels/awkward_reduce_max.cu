// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     """
//     Max reduction for sorted, present parents on device using
//     dtype promotion for atomic performance.
//     """
//     from awkward._connect.cuda._reducers import reduce_with_cupy_at
//     (toptr, fromptr, parents, lenparents,
//      outlength, identity, _invocation_index, _err_code) = args
//     # We use cupy.maximum for the reduction.
//     # The identity provided in args should be -inf (or the min value for the dtype).
//     reduce_with_cupy_at(
//         cupy.maximum,
//         toptr[:outlength],
//         fromptr[:lenparents],
//         parents[:lenparents],
//         identity
//    )
// END PYTHON
