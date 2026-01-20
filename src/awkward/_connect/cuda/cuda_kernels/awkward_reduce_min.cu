// BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     """
//     Min reduction for sorted, present parents on device using
//     dtype promotion for atomic performance.
//     """
//     from awkward._connect.cuda._reducers import reduce_with_cupy_at
//     (toptr, fromptr, parents, lenparents,
//      outlength, identity, _invocation_index, _err_code) = args
//     # We use the helper to handle promotion (e.g., int8 -> int32)
//     # and the atomic operation (cupy.minimum.at).
//     # We slice the toptr to outlength and other arrays to lenparents.
//     reduce_with_cupy_at(
//         cupy.minimum,
//         toptr[:outlength],
//         fromptr[:lenparents],
//         parents[:lenparents],
//         identity
//     )
// END PYTHON
