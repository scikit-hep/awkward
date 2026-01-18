// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

// BEGIN PYTHON
// def f(grid, block, args):
//     """
//     Min reduction for sorted, present parents on device:
//     (toptr, fromptr, parents, lenparents, outlength, identity, invocation_index, err_code)
//     """
//     (toptr, fromptr, parents, lenparents,
//      outlength, identity, invocation_index, err_code) = args
//     toptr[:outlength] = identity
//     cupy.minimum.at(
//        toptr[:outlength],
//        parents[:lenparents],
//        fromptr[:lenparents]
//    )
// END PYTHON
