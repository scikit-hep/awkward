Release history
---------------

Unreleased (`main branch <https://github.com/scikit-hep/awkward-1.0>`__ on GitHub)
====================================================================================

* PR `#1544 <https://github.com/scikit-hep/awkward-1.0/pull/1544>`__: fix ak2 convert class name msg.
* PR `#1543 <https://github.com/scikit-hep/awkward-1.0/pull/1543>`__: [pre-commit.ci] pre-commit autoupdate.
* PR `#1542 <https://github.com/scikit-hep/awkward-1.0/pull/1542>`__: migrate ArrayBuilder to new GrowableBuffer.
* PR `#1527 <https://github.com/scikit-hep/awkward-1.0/pull/1527>`__: [pre-commit.ci] pre-commit autoupdate.

Release `1.9.0rc8 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.9.0rc8>`__
========================================================================================

 (`pip <https://pypi.org/project/awkward/1.9.0rc8/>`__)

* PR `#1541 <https://github.com/scikit-hep/awkward-1.0/pull/1541>`__: docs: add aryan26roy as a contributor for code.
* PR `#1540 <https://github.com/scikit-hep/awkward-1.0/pull/1540>`__: docs: add ManasviGoyal as a contributor for code.
* PR `#1538 <https://github.com/scikit-hep/awkward-1.0/pull/1538>`__: Solving the endian bug on MacOS.
* PR `#1539 <https://github.com/scikit-hep/awkward-1.0/pull/1539>`__: Fix: ak._v2.is_none check for axis value.
* PR `#1532 <https://github.com/scikit-hep/awkward-1.0/pull/1532>`__: Fix: Error when using ak.copy in v2.
* PR `#1537 <https://github.com/scikit-hep/awkward-1.0/pull/1537>`__: Fixed RecordArray.__repr__ (last vestige of 'override' misunderstanding).
* PR `#1531 <https://github.com/scikit-hep/awkward-1.0/pull/1531>`__: Fix: Initialize values behind the mask in ak.to_numpy.
* PR `#1535 <https://github.com/scikit-hep/awkward-1.0/pull/1535>`__: Growable Buffer header.
* PR `#1536 <https://github.com/scikit-hep/awkward-1.0/pull/1536>`__: Build(deps): bump pypa/cibuildwheel from 2.7.0 to 2.8.0.
* PR `#1533 <https://github.com/scikit-hep/awkward-1.0/pull/1533>`__: Fix: numba pre-commit issues.
* PR `#1524 <https://github.com/scikit-hep/awkward-1.0/pull/1524>`__: restructure cpp headers.
* PR `#1484 <https://github.com/scikit-hep/awkward-1.0/pull/1484>`__: LayoutBuilder migration to v2.
* PR `#1523 <https://github.com/scikit-hep/awkward-1.0/pull/1523>`__: add C++ headers-only distribution configuration.
* PR `#1521 <https://github.com/scikit-hep/awkward-1.0/pull/1521>`__: Build(deps): bump actions/download-artifact from 2 to 3.
* PR `#1520 <https://github.com/scikit-hep/awkward-1.0/pull/1520>`__: Build(deps): bump actions/checkout from 2 to 3.
* PR `#1519 <https://github.com/scikit-hep/awkward-1.0/pull/1519>`__: Build(deps): bump actions/upload-artifact from 2 to 3.
* PR `#1518 <https://github.com/scikit-hep/awkward-1.0/pull/1518>`__: chore: remove unneeded lines.

Release `1.9.0rc7 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.9.0rc7>`__
========================================================================================

 (`pip <https://pypi.org/project/awkward/1.9.0rc7/>`__)

* PR `#1514 <https://github.com/scikit-hep/awkward-1.0/pull/1514>`__: Update type-parser for v2.
* PR `#1516 <https://github.com/scikit-hep/awkward-1.0/pull/1516>`__: Package: fix MANIFEST to include cpp-headers.
* PR `#1515 <https://github.com/scikit-hep/awkward-1.0/pull/1515>`__: Swap 'merged dtype same as NumPy' test of v1 for test of v2.

Release `1.9.0rc6 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.9.0rc6>`__
========================================================================================

 (`pip <https://pypi.org/project/awkward/1.9.0rc6/>`__)

* PR `#1513 <https://github.com/scikit-hep/awkward-1.0/pull/1513>`__: Add typeparser to v2.
* PR `#1508 <https://github.com/scikit-hep/awkward-1.0/pull/1508>`__: from rdataframe for awkward arrays.
* PR `#1510 <https://github.com/scikit-hep/awkward-1.0/pull/1510>`__: Build(deps): bump pypa/cibuildwheel from 2.6.1 to 2.7.0.

Release `1.9.0rc5 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.9.0rc5>`__
========================================================================================

 (`pip <https://pypi.org/project/awkward/1.9.0rc5/>`__)

* PR `#1505 <https://github.com/scikit-hep/awkward-1.0/pull/1505>`__: Feat: add missing ``_like`` methods to ``TypeTracer``
* PR `#1503 <https://github.com/scikit-hep/awkward-1.0/pull/1503>`__: [pre-commit.ci] pre-commit autoupdate.
* PR `#1485 <https://github.com/scikit-hep/awkward-1.0/pull/1485>`__: Add the beginning of an example of templated LayoutBuilder.
* PR `#1474 <https://github.com/scikit-hep/awkward-1.0/pull/1474>`__: rdataframe to awkward.
* PR `#1502 <https://github.com/scikit-hep/awkward-1.0/pull/1502>`__: Fix for issue 1406.
* PR `#1499 <https://github.com/scikit-hep/awkward-1.0/pull/1499>`__: Fix slicing for UnmaskedArrays (which come from Arrow).
* PR `#1498 <https://github.com/scikit-hep/awkward-1.0/pull/1498>`__: Fixed typo in unmaskedarray.py.
* PR `#1487 <https://github.com/scikit-hep/awkward-1.0/pull/1487>`__: Adding repr overriden behavior.
* PR `#1491 <https://github.com/scikit-hep/awkward-1.0/pull/1491>`__: Adding a Forth Based Avro Reader.
* PR `#1497 <https://github.com/scikit-hep/awkward-1.0/pull/1497>`__: Fixed slicing shape for array of booleans.
* PR `#1496 <https://github.com/scikit-hep/awkward-1.0/pull/1496>`__: _to_numpy method should return a numpy array.
* PR `#1490 <https://github.com/scikit-hep/awkward-1.0/pull/1490>`__: Refactoring to include index_nplike and reducers.
* PR `#1493 <https://github.com/scikit-hep/awkward-1.0/pull/1493>`__: Build(deps): bump pypa/cibuildwheel from 2.5.0 to 2.6.1.
* PR `#1492 <https://github.com/scikit-hep/awkward-1.0/pull/1492>`__: Fix categorical equality handling (bad copy-paste from v1).
* PR `#1486 <https://github.com/scikit-hep/awkward-1.0/pull/1486>`__: Fix selecting columns from Parquet.
* PR `#1478 <https://github.com/scikit-hep/awkward-1.0/pull/1478>`__: ``to_rdataframe`` extensive tests and bug fixes. (**also:** `#1477 <https://github.com/scikit-hep/awkward-1.0/issues/1477>`__)
* PR `#1475 <https://github.com/scikit-hep/awkward-1.0/pull/1475>`__: update awkward-1.0 to awkward.
* PR `#1447 <https://github.com/scikit-hep/awkward-1.0/pull/1447>`__: This PR attempts to add autodifferentiation support for Awkward Arrays using JAX pytrees.
* PR `#1470 <https://github.com/scikit-hep/awkward-1.0/pull/1470>`__: Rename fillna -> fill_none.
* PR `#1469 <https://github.com/scikit-hep/awkward-1.0/pull/1469>`__: _getitem_* functions must consistently set the slicer in handle_error.
* PR `#1468 <https://github.com/scikit-hep/awkward-1.0/pull/1468>`__: Rename low-level methods to match high-level function names.
* PR `#1467 <https://github.com/scikit-hep/awkward-1.0/pull/1467>`__: Flatten directory structure under src/awkward/_v2/operations.
* PR `#1465 <https://github.com/scikit-hep/awkward-1.0/pull/1465>`__: [pre-commit.ci] pre-commit autoupdate.
* PR `#1462 <https://github.com/scikit-hep/awkward-1.0/pull/1462>`__: Build(deps): bump docker/setup-qemu-action from 1.2.0 to 2.0.0.
* PR `#1464 <https://github.com/scikit-hep/awkward-1.0/pull/1464>`__: Ignore a NumPy 1.22 warning in Numba and fix the flake8-print T001 --> T201 change.

Release `1.9.0rc4 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.9.0rc4>`__
========================================================================================

 (`pip <https://pypi.org/project/awkward/1.9.0rc4/>`__)

*(no pull requests)*

Release `1.9.0rc3 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.9.0rc3>`__
========================================================================================

 (`pip <https://pypi.org/project/awkward/1.9.0rc3/>`__)

* PR `#1449 <https://github.com/scikit-hep/awkward-1.0/pull/1449>`__: Revamping the to_json/from_json interface.
* PR `#1458 <https://github.com/scikit-hep/awkward-1.0/pull/1458>`__: Streamline recursively_apply for small slices of big arrays.
* PR `#1457 <https://github.com/scikit-hep/awkward-1.0/pull/1457>`__: Fix RDataFrame GetColumnNames order in test.
* PR `#1446 <https://github.com/scikit-hep/awkward-1.0/pull/1446>`__: awkward rdataframe source tests.
* PR `#1456 <https://github.com/scikit-hep/awkward-1.0/pull/1456>`__: Fixes ``to_layout`` with ``allow_records=False`` and allows single-record writing to Arrow and Parquet. (**also:** `#1453 <https://github.com/scikit-hep/awkward-1.0/issues/1453>`__)
* PR `#1455 <https://github.com/scikit-hep/awkward-1.0/pull/1455>`__: Build(deps): bump pypa/cibuildwheel from 2.4.0 to 2.5.0.
* PR `#1445 <https://github.com/scikit-hep/awkward-1.0/pull/1445>`__: Pass on skipped v2 tests.
* PR `#1429 <https://github.com/scikit-hep/awkward-1.0/pull/1429>`__: Fix: is_unique() for IndexedArray.
* PR `#1444 <https://github.com/scikit-hep/awkward-1.0/pull/1444>`__: Enable ak.singletons.
* PR `#1443 <https://github.com/scikit-hep/awkward-1.0/pull/1443>`__: Enable ak.firsts.
* PR `#1374 <https://github.com/scikit-hep/awkward-1.0/pull/1374>`__: awkward to rdataframe.
* PR `#1440 <https://github.com/scikit-hep/awkward-1.0/pull/1440>`__: Implementing ak._v2.to_parquet.
* PR `#1437 <https://github.com/scikit-hep/awkward-1.0/pull/1437>`__: Enable mixins behavior.
* PR `#1435 <https://github.com/scikit-hep/awkward-1.0/pull/1435>`__: Remove duplicated import of to/from-parquet.
* PR `#1434 <https://github.com/scikit-hep/awkward-1.0/pull/1434>`__: Enable categorical behavior - testing.

Release `1.9.0rc2 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.9.0rc2>`__
========================================================================================

 (`pip <https://pypi.org/project/awkward/1.9.0rc2/>`__)

* PR `#1433 <https://github.com/scikit-hep/awkward-1.0/pull/1433>`__: Add Array and Record.__delitem__. And fix show(type=True).
* PR `#1432 <https://github.com/scikit-hep/awkward-1.0/pull/1432>`__: Register both v1 and v2 Arrays in Numba entry_points.
* PR `#1431 <https://github.com/scikit-hep/awkward-1.0/pull/1431>`__: Fixed miscellaneous optiontype-in-Parquet bugs. (**also:** `#1397 <https://github.com/scikit-hep/awkward-1.0/issues/1397>`__)
* PR `#1397 <https://github.com/scikit-hep/awkward-1.0/pull/1397>`__: split up functions.
* PR `#1430 <https://github.com/scikit-hep/awkward-1.0/pull/1430>`__: Pretty-printing types.
* PR `#1428 <https://github.com/scikit-hep/awkward-1.0/pull/1428>`__: Implements ak.nan_to_none and all of the ak.nan* functions to override NumPy's.
* PR `#1421 <https://github.com/scikit-hep/awkward-1.0/pull/1421>`__: Enabled string/categorical behavior.
* PR `#1427 <https://github.com/scikit-hep/awkward-1.0/pull/1427>`__: Enable broadcasting of string equality.
* PR `#1426 <https://github.com/scikit-hep/awkward-1.0/pull/1426>`__: ListOffsetArray._reduce_next is not implemented for 32-bit (it could be, but this PR just fixes the error).
* PR `#1425 <https://github.com/scikit-hep/awkward-1.0/pull/1425>`__: Fix ak._v2.to_arrow for sliced ListOffsetArray.

Release `1.9.0rc1 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.9.0rc1>`__
========================================================================================

 (`pip <https://pypi.org/project/awkward/1.9.0rc1/>`__)

* PR `#1422 <https://github.com/scikit-hep/awkward-1.0/pull/1422>`__: Update AwkwardForth documentation and move it from the wiki to the standard docs.
* PR `#1410 <https://github.com/scikit-hep/awkward-1.0/pull/1410>`__: Removed bytemask() in favour of mask_as_bool()
* PR `#1415 <https://github.com/scikit-hep/awkward-1.0/pull/1415>`__: Passing behaviour in ak._v2 functions.
* PR `#1419 <https://github.com/scikit-hep/awkward-1.0/pull/1419>`__: Fix iteration over NumpyArray type.
* PR `#1418 <https://github.com/scikit-hep/awkward-1.0/pull/1418>`__: Fix performance issue in v2 tolist.
* PR `#1416 <https://github.com/scikit-hep/awkward-1.0/pull/1416>`__: docs: add Ahmad-AlSubaie as a contributor for code.
* PR `#1413 <https://github.com/scikit-hep/awkward-1.0/pull/1413>`__: replace llvmlite.ir instead of llvmlite.llvmpy.core.
* PR `#1412 <https://github.com/scikit-hep/awkward-1.0/pull/1412>`__: fix: pypy 3.9.
* PR `#1399 <https://github.com/scikit-hep/awkward-1.0/pull/1399>`__: This PR adds JAX as a new nplike.
* PR `#1409 <https://github.com/scikit-hep/awkward-1.0/pull/1409>`__: [pre-commit.ci] pre-commit autoupdate.
* PR `#1407 <https://github.com/scikit-hep/awkward-1.0/pull/1407>`__: Windows builds stopped working; be looser about directory name.
* PR `#1404 <https://github.com/scikit-hep/awkward-1.0/pull/1404>`__: Fix: ``from_numpy`` references ``ListArray64`` (**also:** `#1403 <https://github.com/scikit-hep/awkward-1.0/issues/1403>`__)
* PR `#1401 <https://github.com/scikit-hep/awkward-1.0/pull/1401>`__: Implement ``recursively_apply`` for ``Record``
* PR `#1398 <https://github.com/scikit-hep/awkward-1.0/pull/1398>`__: ROOT doesn't recognize for-each iterators without operator==
* PR `#1390 <https://github.com/scikit-hep/awkward-1.0/pull/1390>`__: This PR adds all the remaining kernels in the studies directory.
* PR `#1395 <https://github.com/scikit-hep/awkward-1.0/pull/1395>`__: [pre-commit.ci] pre-commit autoupdate.
* PR `#1394 <https://github.com/scikit-hep/awkward-1.0/pull/1394>`__: Build(deps): bump pypa/cibuildwheel from 2.3.1 to 2.4.0.
* PR `#1393 <https://github.com/scikit-hep/awkward-1.0/pull/1393>`__: ErrorContexts should only contain strings.
* PR `#1392 <https://github.com/scikit-hep/awkward-1.0/pull/1392>`__: High-level ak._v2.Array clean-ups.
* PR `#1387 <https://github.com/scikit-hep/awkward-1.0/pull/1387>`__: No zero-length shortcuts for ak.argsort (v1 & v2). (**also:** `psf/black#2964 <https://github.com/psf/black/issues/2964>`__)
* PR `#1385 <https://github.com/scikit-hep/awkward-1.0/pull/1385>`__: fix: bump black to 22.3.0 due to click 8.1 release\n\nSee https://github.com/psf/black/issues/2964 for details.
* PR `#1381 <https://github.com/scikit-hep/awkward-1.0/pull/1381>`__: This PR adds the generated kernels and simplifies the template specialization generation process.
* PR `#1384 <https://github.com/scikit-hep/awkward-1.0/pull/1384>`__: Fix _prettyprint after 'for i in range' changed to 'for i, val in enumerate'.
* PR `#1383 <https://github.com/scikit-hep/awkward-1.0/pull/1383>`__: Protect test 1300 from ROOT without C++17 (or, at least, without std::optional).
* PR `#1380 <https://github.com/scikit-hep/awkward-1.0/pull/1380>`__: Reducers with axis=None and typetracers.
* PR `#1378 <https://github.com/scikit-hep/awkward-1.0/pull/1378>`__: Fixes nonlocal reducers in which the first list is empty.
* PR `#1355 <https://github.com/scikit-hep/awkward-1.0/pull/1355>`__: This PR sets up the architecture to call CuPy Raw Kernels from Awkward.
* PR `#1376 <https://github.com/scikit-hep/awkward-1.0/pull/1376>`__: Allow NumPy arrays in CppStatements; fix row_groups in single-file from_parquet.
* PR `#1373 <https://github.com/scikit-hep/awkward-1.0/pull/1373>`__: Feat: add ``depth_limit`` to ``ak.broadcast_arrays``
* PR `#1365 <https://github.com/scikit-hep/awkward-1.0/pull/1365>`__: Refactor: cleanup reducer.
* PR `#1360 <https://github.com/scikit-hep/awkward-1.0/pull/1360>`__: C++ refactoring: ak.unflatten.
* PR `#1372 <https://github.com/scikit-hep/awkward-1.0/pull/1372>`__: Allow NumPy arrays in CppStatements; fix row_groups in single-file from_parquet.
* PR `#1369 <https://github.com/scikit-hep/awkward-1.0/pull/1369>`__: C++ refactoring: ak.to_pandas.
* PR `#1367 <https://github.com/scikit-hep/awkward-1.0/pull/1367>`__: C++ refactoring: ak.copy.
* PR `#1338 <https://github.com/scikit-hep/awkward-1.0/pull/1338>`__: First version of ak._v2.from_parquet.
* PR `#1368 <https://github.com/scikit-hep/awkward-1.0/pull/1368>`__: C++ refactoring: ak.broadcast_arrays.
* PR `#1359 <https://github.com/scikit-hep/awkward-1.0/pull/1359>`__: Pure Cling demo and improvements to C++ JIT infrastructure. (**also:** `#1295 <https://github.com/scikit-hep/awkward-1.0/issues/1295>`__)
* PR `#1370 <https://github.com/scikit-hep/awkward-1.0/pull/1370>`__: [pre-commit.ci] pre-commit autoupdate.
* PR `#1364 <https://github.com/scikit-hep/awkward-1.0/pull/1364>`__: Fixes `#1363 <https://github.com/scikit-hep/awkward-1.0/issues/1363>`__ by ensuring that arguments documented as 'iterable of X' aren't used in 'len(X)'.
* PR `#1354 <https://github.com/scikit-hep/awkward-1.0/pull/1354>`__: C++ refactoring: ak.unzip.
* PR `#1356 <https://github.com/scikit-hep/awkward-1.0/pull/1356>`__: Fix: fix docstring line.
* PR `#1351 <https://github.com/scikit-hep/awkward-1.0/pull/1351>`__: Feat: add ``is_tuple`` describe operation.
* PR `#1347 <https://github.com/scikit-hep/awkward-1.0/pull/1347>`__: C++ refactoring: ak.run_lengths.
* PR `#1352 <https://github.com/scikit-hep/awkward-1.0/pull/1352>`__: C++ refactoring: ak.nan_to_num.
* PR `#1346 <https://github.com/scikit-hep/awkward-1.0/pull/1346>`__: Fix PR `#788 <https://github.com/scikit-hep/awkward-1.0/issues/788>`__: avoid materializing VirtualArrays in ak.with_name.
* PR `#1327 <https://github.com/scikit-hep/awkward-1.0/pull/1327>`__: Straighten out error handling via a thread-local (but otherwise global) context.
* PR `#1340 <https://github.com/scikit-hep/awkward-1.0/pull/1340>`__: ak.flatten and ak.ravel should test for nplike.ndarray, not np.ndarray.
* PR `#1329 <https://github.com/scikit-hep/awkward-1.0/pull/1329>`__: Fixed ak.num with axis=0 in typetracer.

Release `1.8.0 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.8.0>`__
==================================================================================

 (`pip <https://pypi.org/project/awkward/1.8.0/>`__)

*(no pull requests)*

Release `1.8.0rc7 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.8.0rc7>`__
========================================================================================

 (`pip <https://pypi.org/project/awkward/1.8.0rc7/>`__)

* PR `#1326 <https://github.com/scikit-hep/awkward-1.0/pull/1326>`__: Docs: fix typo in documentation.
* PR `#1314 <https://github.com/scikit-hep/awkward-1.0/pull/1314>`__: chore: remove extra files from the wheels.
* PR `#1313 <https://github.com/scikit-hep/awkward-1.0/pull/1313>`__: ci: avoid PyPI cuda wheel upload.
* PR `#1316 <https://github.com/scikit-hep/awkward-1.0/pull/1316>`__: chore: bump pybind11 to 2.9.1.
* PR `#1312 <https://github.com/scikit-hep/awkward-1.0/pull/1312>`__: Keep as much length knowledge as possible in typetracers.
* PR `#1322 <https://github.com/scikit-hep/awkward-1.0/pull/1322>`__: chore: wheel not required for setuptools PEP 517 (all-repos)
* PR `#1317 <https://github.com/scikit-hep/awkward-1.0/pull/1317>`__: C++ refactoring: ak.cartesian, ak.argcartesian.
* PR `#1301 <https://github.com/scikit-hep/awkward-1.0/pull/1301>`__: C++ refactoring: ak.strings_astype.
* PR `#1308 <https://github.com/scikit-hep/awkward-1.0/pull/1308>`__: Feat: add ``optiontype_outside_record`` argument to ``ak.zip``
* PR `#1307 <https://github.com/scikit-hep/awkward-1.0/pull/1307>`__: C++ refactoring: ak.argcombinations, ak.combinations.
* PR `#1309 <https://github.com/scikit-hep/awkward-1.0/pull/1309>`__: C++ refactoring: ak.sort.

Release `1.8.0rc6 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.8.0rc6>`__
========================================================================================

 (`pip <https://pypi.org/project/awkward/1.8.0rc6/>`__)

* PR `#1310 <https://github.com/scikit-hep/awkward-1.0/pull/1310>`__: Fix lost 'behavior' in 'ak.unzip'.
* PR `#1300 <https://github.com/scikit-hep/awkward-1.0/pull/1300>`__: Implement Awkward --> C++ with Cling.
* PR `#1306 <https://github.com/scikit-hep/awkward-1.0/pull/1306>`__: Version of awkward_cuda should be tied with awkward.
* PR `#1304 <https://github.com/scikit-hep/awkward-1.0/pull/1304>`__: C++ refactoring: ak.argsort.
* PR `#1303 <https://github.com/scikit-hep/awkward-1.0/pull/1303>`__: Fix: do not increment field index for nested lists.

Release `1.8.0rc5 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.8.0rc5>`__
========================================================================================

 (`pip <https://pypi.org/project/awkward/1.8.0rc5/>`__)

*(no pull requests)*

Release `1.8.0rc4 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.8.0rc4>`__
========================================================================================

 (`pip <https://pypi.org/project/awkward/1.8.0rc4/>`__)

* PR `#1299 <https://github.com/scikit-hep/awkward-1.0/pull/1299>`__: Remove unnecessary line blank from the tops of almost all files .
* PR `#1298 <https://github.com/scikit-hep/awkward-1.0/pull/1298>`__: Allow ak.nan_to_num arguments to be arrays.
* PR `#1296 <https://github.com/scikit-hep/awkward-1.0/pull/1296>`__: ak.fields.
* PR `#1297 <https://github.com/scikit-hep/awkward-1.0/pull/1297>`__: ak.without_parameters.
* PR `#1293 <https://github.com/scikit-hep/awkward-1.0/pull/1293>`__: C++ refactoring: ak.full_like, ak.zeros_like, ak.ones_like.
* PR `#1275 <https://github.com/scikit-hep/awkward-1.0/pull/1275>`__: style: pylint 1.
* PR `#1274 <https://github.com/scikit-hep/awkward-1.0/pull/1274>`__: Fixing `#1266 <https://github.com/scikit-hep/awkward-1.0/issues/1266>`__ (in v1 and v2), possibly by reordering nextparents.
* PR `#1294 <https://github.com/scikit-hep/awkward-1.0/pull/1294>`__: C++ refactoring: ak._v2.from_arrow_schema function.
* PR `#1289 <https://github.com/scikit-hep/awkward-1.0/pull/1289>`__: C++ refactoring: ak.with_parameter.
* PR `#1292 <https://github.com/scikit-hep/awkward-1.0/pull/1292>`__: C++ refactoring: ak.with_field.
* PR `#1290 <https://github.com/scikit-hep/awkward-1.0/pull/1290>`__: typo.
* PR `#1276 <https://github.com/scikit-hep/awkward-1.0/pull/1276>`__: This PR adds support to call kernels in CUDA from v2 Awkward Arrays.
* PR `#1249 <https://github.com/scikit-hep/awkward-1.0/pull/1249>`__: Fix: support nested option types in ``ak.is_none`` (**also:** `#1193 <https://github.com/scikit-hep/awkward-1.0/issues/1193>`__, `#1193 <https://github.com/scikit-hep/awkward-1.0/issues/1193>`__)
* PR `#1279 <https://github.com/scikit-hep/awkward-1.0/pull/1279>`__: Fix: simplify output in {Byte,Bit}MaskedArray.
* PR `#1277 <https://github.com/scikit-hep/awkward-1.0/pull/1277>`__: [pre-commit.ci] pre-commit autoupdate.
* PR `#1240 <https://github.com/scikit-hep/awkward-1.0/pull/1240>`__: Getting Numba to work for v2 arrays.
* PR `#1207 <https://github.com/scikit-hep/awkward-1.0/pull/1207>`__: json de-/serialisation from/to string or file.
* PR `#1270 <https://github.com/scikit-hep/awkward-1.0/pull/1270>`__: Add GHA to build CUDA Wheels and update the cuda build script.
* PR `#1262 <https://github.com/scikit-hep/awkward-1.0/pull/1262>`__: chore: initial nox and pylint support.
* PR `#1267 <https://github.com/scikit-hep/awkward-1.0/pull/1267>`__: style: update to first non-pre-release black!
* PR `#1265 <https://github.com/scikit-hep/awkward-1.0/pull/1265>`__: Bump pypa/gh-action-pypi-publish from 1.4.2 to 1.5.0.
* PR `#1257 <https://github.com/scikit-hep/awkward-1.0/pull/1257>`__: Add a .zenodo.json file to specify a set of authors.
* PR `#1264 <https://github.com/scikit-hep/awkward-1.0/pull/1264>`__: Bump pypa/cibuildwheel from 1.12.0 to 2.3.1.
* PR `#1263 <https://github.com/scikit-hep/awkward-1.0/pull/1263>`__: chore: add dependabot for actions.
* PR `#1259 <https://github.com/scikit-hep/awkward-1.0/pull/1259>`__: Fix: fix ``ByteMaskedArray.simplify_optiontype()``
* PR `#1258 <https://github.com/scikit-hep/awkward-1.0/pull/1258>`__: Remove distutils reference in test (now an error).
* PR `#1255 <https://github.com/scikit-hep/awkward-1.0/pull/1255>`__: chore: update pytest config, 6.0+
* PR `#1242 <https://github.com/scikit-hep/awkward-1.0/pull/1242>`__: C++ refactoring: ak.parameters.
* PR `#1243 <https://github.com/scikit-hep/awkward-1.0/pull/1243>`__: style: add shellcheck.
* PR `#1254 <https://github.com/scikit-hep/awkward-1.0/pull/1254>`__: fix: building twice was broken.
* PR `#1248 <https://github.com/scikit-hep/awkward-1.0/pull/1248>`__: Fix: support mixed array types in ``NumpyLike.to_rectilinear``
* PR `#1246 <https://github.com/scikit-hep/awkward-1.0/pull/1246>`__: style: further cleanup for Python 3.6+
* PR `#1244 <https://github.com/scikit-hep/awkward-1.0/pull/1244>`__: style: pyupgrade to 3.6.
* PR `#1245 <https://github.com/scikit-hep/awkward-1.0/pull/1245>`__: layout.completely_flatten should not concatenate (performance issue).
* PR `#1234 <https://github.com/scikit-hep/awkward-1.0/pull/1234>`__: C++ refactoring: ak.type and ak.values_astype.
* PR `#1214 <https://github.com/scikit-hep/awkward-1.0/pull/1214>`__: Fix: drop parameters for flattened RecordArray.

Release `1.8.0rc3 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.8.0rc3>`__
========================================================================================

 (`pip <https://pypi.org/project/awkward/1.8.0rc3/>`__)

* PR `#1239 <https://github.com/scikit-hep/awkward-1.0/pull/1239>`__: Revert "Build wheels for ppc64le (`#1224 <https://github.com/scikit-hep/awkward-1.0/issues/1224>`__)"

Release `1.8.0rc2 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.8.0rc2>`__
========================================================================================

* PR `#1233 <https://github.com/scikit-hep/awkward-1.0/pull/1233>`__: C++ refactoring: ak.with_name.
* PR `#1231 <https://github.com/scikit-hep/awkward-1.0/pull/1231>`__: Updated the generate-cuda script. Works for py >= 3.8.
* PR `#1224 <https://github.com/scikit-hep/awkward-1.0/pull/1224>`__: Build wheels for ppc64le.
* PR `#1237 <https://github.com/scikit-hep/awkward-1.0/pull/1237>`__: Remove Windows 32-bit from the Python 3.10 build.
* PR `#1229 <https://github.com/scikit-hep/awkward-1.0/pull/1229>`__: C++ refactoring: ak.pad_none.
* PR `#1232 <https://github.com/scikit-hep/awkward-1.0/pull/1232>`__: macos segfault bugfix.
* PR `#1225 <https://github.com/scikit-hep/awkward-1.0/pull/1225>`__: C++ refactoring: ak.zip.
* PR `#1228 <https://github.com/scikit-hep/awkward-1.0/pull/1228>`__: Redo PR `#1227 <https://github.com/scikit-hep/awkward-1.0/issues/1227>`__: fixing 'emptyArray' typo.
* PR `#1226 <https://github.com/scikit-hep/awkward-1.0/pull/1226>`__: C++ refactoring: ak.num.
* PR `#1217 <https://github.com/scikit-hep/awkward-1.0/pull/1217>`__: C++ refactoring: ak.flatten.
* PR `#1220 <https://github.com/scikit-hep/awkward-1.0/pull/1220>`__: C++ refactoring: ak.where.
* PR `#1223 <https://github.com/scikit-hep/awkward-1.0/pull/1223>`__: Restore pybind11 2.9.0.
* PR `#1218 <https://github.com/scikit-hep/awkward-1.0/pull/1218>`__: Make highlevel __repr__ safe for typetracers.
* PR `#1219 <https://github.com/scikit-hep/awkward-1.0/pull/1219>`__: C++ refactoring: ak.mask.
* PR `#1221 <https://github.com/scikit-hep/awkward-1.0/pull/1221>`__: C++ refactoring: ak.local_index.
* PR `#1222 <https://github.com/scikit-hep/awkward-1.0/pull/1222>`__: C++ refactoring: ak.ravel.
* PR `#1211 <https://github.com/scikit-hep/awkward-1.0/pull/1211>`__: Removed v1_to_v2 from all v2 tests. (**also:** `#962 <https://github.com/scikit-hep/awkward-1.0/issues/962>`__)
* PR `#1215 <https://github.com/scikit-hep/awkward-1.0/pull/1215>`__: Fixed handling of list-nested boolean slices.
* PR `#1212 <https://github.com/scikit-hep/awkward-1.0/pull/1212>`__: Drop Win32 Py3.10 test and musllinux in deployment.
* PR `#1213 <https://github.com/scikit-hep/awkward-1.0/pull/1213>`__: [pre-commit.ci] pre-commit autoupdate.

Release `1.8.0rc1 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.8.0rc1>`__
========================================================================================

* PR `#1188 <https://github.com/scikit-hep/awkward-1.0/pull/1188>`__: ci: try Numba RC on 3.10.
* PR `#1199 <https://github.com/scikit-hep/awkward-1.0/pull/1199>`__: chore: bump to pybind11 2.9.0.
* PR `#1210 <https://github.com/scikit-hep/awkward-1.0/pull/1210>`__: docs: add BioGeek as a contributor for doc.
* PR `#1208 <https://github.com/scikit-hep/awkward-1.0/pull/1208>`__: ak._v2 namespace is now filled with the right symbols.
* PR `#1206 <https://github.com/scikit-hep/awkward-1.0/pull/1206>`__: Highlevel non-reducers and improved testing/fixes for reducers.
* PR `#1204 <https://github.com/scikit-hep/awkward-1.0/pull/1204>`__: ak._v2.operations.convert.to_numpy is done.
* PR `#1203 <https://github.com/scikit-hep/awkward-1.0/pull/1203>`__: Don't let ak.to_list act on v2 arrays (finishing `#1201 <https://github.com/scikit-hep/awkward-1.0/issues/1201>`__).
* PR `#1202 <https://github.com/scikit-hep/awkward-1.0/pull/1202>`__: Better error message for Content::axis_wrap_if_negative.
* PR `#1201 <https://github.com/scikit-hep/awkward-1.0/pull/1201>`__: Implemented v2 ak.to_list and switched all v2 tests to use it.
* PR `#1198 <https://github.com/scikit-hep/awkward-1.0/pull/1198>`__: Allow non-array iterables in __array_function__.
* PR `#1197 <https://github.com/scikit-hep/awkward-1.0/pull/1197>`__: Fix ak.singletons for non-optional data.
* PR `#1196 <https://github.com/scikit-hep/awkward-1.0/pull/1196>`__: Remove distutils dependence.
* PR `#1195 <https://github.com/scikit-hep/awkward-1.0/pull/1195>`__: Fix: _pack_layout should also pack projected index arrays.
* PR `#1194 <https://github.com/scikit-hep/awkward-1.0/pull/1194>`__: [pre-commit.ci] pre-commit autoupdate.
* PR `#948 <https://github.com/scikit-hep/awkward-1.0/pull/948>`__: pictures for a tutorial.
* PR `#1155 <https://github.com/scikit-hep/awkward-1.0/pull/1155>`__: ArrayBuilder: replace shared with unique.
* PR `#1011 <https://github.com/scikit-hep/awkward-1.0/pull/1011>`__: chore: bump pybind11 to 2.8.0.
* PR `#1186 <https://github.com/scikit-hep/awkward-1.0/pull/1186>`__: feat: bump cibuildwheel, add Python 3.10.
* PR `#1187 <https://github.com/scikit-hep/awkward-1.0/pull/1187>`__: Remove duplicated text.
* PR `#1184 <https://github.com/scikit-hep/awkward-1.0/pull/1184>`__: Drop all length information from TypeTracer, get all tests working again.
* PR `#1183 <https://github.com/scikit-hep/awkward-1.0/pull/1183>`__: Bugs found by the Dask project: broaden type-tracers' applicability.
* PR `#1172 <https://github.com/scikit-hep/awkward-1.0/pull/1172>`__: First bug found by @martindurant.
* PR `#1182 <https://github.com/scikit-hep/awkward-1.0/pull/1182>`__: Remove Python 2.7 and 3.5 support.
* PR `#1181 <https://github.com/scikit-hep/awkward-1.0/pull/1181>`__: Fixed zeros in RegularArray shape.
* PR `#1175 <https://github.com/scikit-hep/awkward-1.0/pull/1175>`__: NumpyArray::numbers_to_type must use flattened_length, not length.
* PR `#1180 <https://github.com/scikit-hep/awkward-1.0/pull/1180>`__: ak.to_numpy with RegularArray of size zero and non-zero length.
* PR `#1179 <https://github.com/scikit-hep/awkward-1.0/pull/1179>`__: Raise ValueError for incompatible union types in ak.unzip.
* PR `#1178 <https://github.com/scikit-hep/awkward-1.0/pull/1178>`__: Fix leading zeros in ak.unflatten.
* PR `#1174 <https://github.com/scikit-hep/awkward-1.0/pull/1174>`__: [pre-commit.ci] pre-commit autoupdate.

Release `1.7.0 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.7.0>`__
==================================================================================

 (`pip <https://pypi.org/project/awkward/1.7.0/>`__)

* PR `#1170 <https://github.com/scikit-hep/awkward-1.0/pull/1170>`__: Parquet files with zero record batches.
* PR `#1169 <https://github.com/scikit-hep/awkward-1.0/pull/1169>`__: remove debug printout.
* PR `#1161 <https://github.com/scikit-hep/awkward-1.0/pull/1161>`__: C++ refactoring: ak.concatenate.
* PR `#1164 <https://github.com/scikit-hep/awkward-1.0/pull/1164>`__: C++ refactoring: to and from json.
* PR `#1168 <https://github.com/scikit-hep/awkward-1.0/pull/1168>`__: avoid division by zero.
* PR `#1166 <https://github.com/scikit-hep/awkward-1.0/pull/1166>`__: Preserve order in v1 RecordForm.contents.
* PR `#1165 <https://github.com/scikit-hep/awkward-1.0/pull/1165>`__: Second try at specialized JSON: RapidJSON + custom assembly.
* PR `#1162 <https://github.com/scikit-hep/awkward-1.0/pull/1162>`__: ak.from_json_schema as a demonstration of generating AwkwardForth from a type-schema.
* PR `#1163 <https://github.com/scikit-hep/awkward-1.0/pull/1163>`__: [pre-commit.ci] pre-commit autoupdate.
* PR `#1148 <https://github.com/scikit-hep/awkward-1.0/pull/1148>`__: C++ refactoring: flatten()
* PR `#1160 <https://github.com/scikit-hep/awkward-1.0/pull/1160>`__: Better decompiled print-outs for 'case' and 'enum'.
* PR `#1159 <https://github.com/scikit-hep/awkward-1.0/pull/1159>`__: Add JSON commands to AwkwardForth.
* PR `#1147 <https://github.com/scikit-hep/awkward-1.0/pull/1147>`__: C++ refactoring: fillna() operation.
* PR `#1145 <https://github.com/scikit-hep/awkward-1.0/pull/1145>`__: C++ refactoring: numbers_to_type()
* PR `#1150 <https://github.com/scikit-hep/awkward-1.0/pull/1150>`__: C++ refactoring: to_numpy()
* PR `#1156 <https://github.com/scikit-hep/awkward-1.0/pull/1156>`__: Replace leaf Nones with +-inf for argmin/argmax axis=None.
* PR `#1153 <https://github.com/scikit-hep/awkward-1.0/pull/1153>`__: 'behaviorof' should take Array, not layouts, and setting 'behavior' should set the '__class__'.
* PR `#1154 <https://github.com/scikit-hep/awkward-1.0/pull/1154>`__: Arrow Tables should preserve parameters.
* PR `#1149 <https://github.com/scikit-hep/awkward-1.0/pull/1149>`__: C++ refactoring: handle datetime and timedelta.
* PR `#1142 <https://github.com/scikit-hep/awkward-1.0/pull/1142>`__: C++ refactoring: prepared high-level ArrayBuilder and reducer functions, though still untested.
* PR `#1146 <https://github.com/scikit-hep/awkward-1.0/pull/1146>`__: primitive_to_dtype and dtype_to_primitive as functions.
* PR `#1138 <https://github.com/scikit-hep/awkward-1.0/pull/1138>`__: C++ refactoring: nbytes.
* PR `#1143 <https://github.com/scikit-hep/awkward-1.0/pull/1143>`__: C++ refactoring: NumPy ufuncs for v2.
* PR `#1137 <https://github.com/scikit-hep/awkward-1.0/pull/1137>`__: C++ refactoring: num()
* PR `#1134 <https://github.com/scikit-hep/awkward-1.0/pull/1134>`__: C++ refactoring: to_buffers and from_buffers.
* PR `#1140 <https://github.com/scikit-hep/awkward-1.0/pull/1140>`__: remove workaround, use merge.
* PR `#1141 <https://github.com/scikit-hep/awkward-1.0/pull/1141>`__: fix unionarray sort and enable tests.
* PR `#1135 <https://github.com/scikit-hep/awkward-1.0/pull/1135>`__: C++ Refactoring: Implement rpad and rpad_and_clip.
* PR `#1111 <https://github.com/scikit-hep/awkward-1.0/pull/1111>`__: C++ refactoring: unique and is_unique.
* PR `#1128 <https://github.com/scikit-hep/awkward-1.0/pull/1128>`__: [pre-commit.ci] pre-commit autoupdate.
* PR `#1132 <https://github.com/scikit-hep/awkward-1.0/pull/1132>`__: C++ refactoring: utility methods for high-level functions in v2.
* PR `#1130 <https://github.com/scikit-hep/awkward-1.0/pull/1130>`__: Removed 'simplify_uniontype' from content -- already in unionarray.
* PR `#1131 <https://github.com/scikit-hep/awkward-1.0/pull/1131>`__: High-level to/from_arrow functions for v2.
* PR `#1129 <https://github.com/scikit-hep/awkward-1.0/pull/1129>`__: Update Azure Pipeline's Windows VM image and adapt to Arrow and Numba updates.
* PR `#1125 <https://github.com/scikit-hep/awkward-1.0/pull/1125>`__: C++ refactoring: to_arrow and from_arrow in v2.
* PR `#1124 <https://github.com/scikit-hep/awkward-1.0/pull/1124>`__: Make the commented-out code in v2 a better guide.
* PR `#1123 <https://github.com/scikit-hep/awkward-1.0/pull/1123>`__: Renamed record 'key' -> 'field' to be consistent with high-level.
* PR `#1122 <https://github.com/scikit-hep/awkward-1.0/pull/1122>`__: Working on the high-level ak.Array for v2. (**also:** `#838 <https://github.com/scikit-hep/awkward-1.0/issues/838>`__)
* PR `#1121 <https://github.com/scikit-hep/awkward-1.0/pull/1121>`__: Stubs for high-level interface in the src/awkward/_v2 directory.
* PR `#1120 <https://github.com/scikit-hep/awkward-1.0/pull/1120>`__: Enable codecov.
* PR `#1116 <https://github.com/scikit-hep/awkward-1.0/pull/1116>`__: C++ refactoring: project - bit/byte/unmaskedarray.
* PR `#1119 <https://github.com/scikit-hep/awkward-1.0/pull/1119>`__: Remove v2 VirtualArray (to try using Dask only).
* PR `#1118 <https://github.com/scikit-hep/awkward-1.0/pull/1118>`__: Prepare the 1.7.0 deprecation (ak.fill_none default axis).
* PR `#1117 <https://github.com/scikit-hep/awkward-1.0/pull/1117>`__: Move v2 tests into their own directory.
* PR `#1082 <https://github.com/scikit-hep/awkward-1.0/pull/1082>`__: C++ refactoring: Merge and Simplify Types.

Release `1.5.1 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.5.1>`__
==================================================================================

 (`pip <https://pypi.org/project/awkward/1.5.1/>`__)

* PR `#1114 <https://github.com/scikit-hep/awkward-1.0/pull/1114>`__: Fixes copyjson casting bug.
* PR `#1110 <https://github.com/scikit-hep/awkward-1.0/pull/1110>`__: Implemented the type tracer for Awkward-Dask. (**also:** `#959 <https://github.com/scikit-hep/awkward-1.0/issues/959>`__, `#1031 <https://github.com/scikit-hep/awkward-1.0/issues/1031>`__)
* PR `#1112 <https://github.com/scikit-hep/awkward-1.0/pull/1112>`__: [pre-commit.ci] pre-commit autoupdate.
* PR `#1092 <https://github.com/scikit-hep/awkward-1.0/pull/1092>`__: C++ refactoring: argsort.
* PR `#1099 <https://github.com/scikit-hep/awkward-1.0/pull/1099>`__: C++ refactoring: reducers.
* PR `#1109 <https://github.com/scikit-hep/awkward-1.0/pull/1109>`__: "The Good Parts" of `#1095 <https://github.com/scikit-hep/awkward-1.0/issues/1095>`__, which I'm closing.
* PR `#1085 <https://github.com/scikit-hep/awkward-1.0/pull/1085>`__: Fix: add utility to check whether a string is a filepath. (**also:** `#1084 <https://github.com/scikit-hep/awkward-1.0/issues/1084>`__)
* PR `#1108 <https://github.com/scikit-hep/awkward-1.0/pull/1108>`__: Fixed `#1071 <https://github.com/scikit-hep/awkward-1.0/issues/1071>`__: mask_identity=False should not return option type.
* PR `#1102 <https://github.com/scikit-hep/awkward-1.0/pull/1102>`__: [pre-commit.ci] pre-commit autoupdate.
* PR `#1101 <https://github.com/scikit-hep/awkward-1.0/pull/1101>`__: C++ Refactoring: Implement validityerror for all array types.
* PR `#1094 <https://github.com/scikit-hep/awkward-1.0/pull/1094>`__: Little fixes from meeting with @ianna.
* PR `#1072 <https://github.com/scikit-hep/awkward-1.0/pull/1072>`__: C++ refactoring: sort.
* PR `#1091 <https://github.com/scikit-hep/awkward-1.0/pull/1091>`__: Respect CMAKE_ARGS if set by the environment.

Release `1.5.0 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.5.0>`__
==================================================================================

 (`pip <https://pypi.org/project/awkward/1.5.0/>`__)

*(no pull requests)*

Release `1.5.0rc2 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.5.0rc2>`__
========================================================================================

* PR `#1089 <https://github.com/scikit-hep/awkward-1.0/pull/1089>`__: Revert 'NumPy' spelling to 'Numpy' in code only.
* PR `#1087 <https://github.com/scikit-hep/awkward-1.0/pull/1087>`__: docs: add bmwiedemann as a contributor for code.
* PR `#1088 <https://github.com/scikit-hep/awkward-1.0/pull/1088>`__: docs: add SantamRC as a contributor for test.

Release `1.5.0rc1 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.5.0rc1>`__
========================================================================================

* PR `#1086 <https://github.com/scikit-hep/awkward-1.0/pull/1086>`__: docs: add matthewfeickert as a contributor for maintenance.
* PR `#1081 <https://github.com/scikit-hep/awkward-1.0/pull/1081>`__: Data for Remaining Kernel Functions.
* PR `#1003 <https://github.com/scikit-hep/awkward-1.0/pull/1003>`__: chore: fix spelling and check in pre-commit.
* PR `#1070 <https://github.com/scikit-hep/awkward-1.0/pull/1070>`__: Try fixing the search box by upgrading sphinx-rtd-theme.
* PR `#1079 <https://github.com/scikit-hep/awkward-1.0/pull/1079>`__: Implementing VirtualArray in Awkward v2.
* PR `#1073 <https://github.com/scikit-hep/awkward-1.0/pull/1073>`__: C++ refactoring: handling ListOffsetArray and IndexedOptionArray in _getitem_next.
* PR `#1074 <https://github.com/scikit-hep/awkward-1.0/pull/1074>`__: C++ refactoring: Implementing combinations.
* PR `#1065 <https://github.com/scikit-hep/awkward-1.0/pull/1065>`__: Unit Tests.
* PR `#1078 <https://github.com/scikit-hep/awkward-1.0/pull/1078>`__: [pre-commit.ci] pre-commit autoupdate.
* PR `#1063 <https://github.com/scikit-hep/awkward-1.0/pull/1063>`__: LayoutBuilder template using either ForthMachine32 or ForthMachine64 and a tutorial.
* PR `#1036 <https://github.com/scikit-hep/awkward-1.0/pull/1036>`__: C++ refactoring: testing_starting _getitem_next.
* PR `#1059 <https://github.com/scikit-hep/awkward-1.0/pull/1059>`__: C++ refactoring: Implementing _localindex.
* PR `#1067 <https://github.com/scikit-hep/awkward-1.0/pull/1067>`__: Fixes `#1066 <https://github.com/scikit-hep/awkward-1.0/issues/1066>`__, ak.to_numpy can return masked-structured arrays.
* PR `#1045 <https://github.com/scikit-hep/awkward-1.0/pull/1045>`__: LayoutBuilder refactoring.
* PR `#1062 <https://github.com/scikit-hep/awkward-1.0/pull/1062>`__: Optimizing common take operations.
* PR `#1061 <https://github.com/scikit-hep/awkward-1.0/pull/1061>`__: Explicit ak.Record.__iter__ (iterates over fields, like dict) and better ak.from_iter handling of nested ak.Record and ak.Array.
* PR `#1050 <https://github.com/scikit-hep/awkward-1.0/pull/1050>`__: Allow to override build date with SOURCE_DATE_EPOCH.
* PR `#1058 <https://github.com/scikit-hep/awkward-1.0/pull/1058>`__: Fix deprecation warning stack level.
* PR `#1056 <https://github.com/scikit-hep/awkward-1.0/pull/1056>`__: Fix ak.fill_none fill value's handling of NumPy dimension.
* PR `#1054 <https://github.com/scikit-hep/awkward-1.0/pull/1054>`__: Remove debugging printout (2).
* PR `#1030 <https://github.com/scikit-hep/awkward-1.0/pull/1030>`__: Feat: add ``name`` parameter to ``mixin_class``
* PR `#1051 <https://github.com/scikit-hep/awkward-1.0/pull/1051>`__: Remove debugging printout.
* PR `#977 <https://github.com/scikit-hep/awkward-1.0/pull/977>`__: ArrayBuilder refactoring.
* PR `#1031 <https://github.com/scikit-hep/awkward-1.0/pull/1031>`__: C++ refactoring: starting _getitem_next.
* PR `#1035 <https://github.com/scikit-hep/awkward-1.0/pull/1035>`__: docs: add ioanaif as a contributor for code, test.
* PR `#1029 <https://github.com/scikit-hep/awkward-1.0/pull/1029>`__: Fixed `#1026 <https://github.com/scikit-hep/awkward-1.0/issues/1026>`__; jagged slicing of multidim NumpyArray.
* PR `#1028 <https://github.com/scikit-hep/awkward-1.0/pull/1028>`__: Reverting `#694 <https://github.com/scikit-hep/awkward-1.0/issues/694>`__: SliceVarNewAxis.
* PR `#1025 <https://github.com/scikit-hep/awkward-1.0/pull/1025>`__: Rename _getitem_array as _carry and have it take Index.
* PR `#1024 <https://github.com/scikit-hep/awkward-1.0/pull/1024>`__: Chore: correct spelling of "operation"
* PR `#1023 <https://github.com/scikit-hep/awkward-1.0/pull/1023>`__: Docs: make link to layout.
* PR `#1021 <https://github.com/scikit-hep/awkward-1.0/pull/1021>`__: Feat: pack ``Record``s in ``ak.packed``
* PR `#1019 <https://github.com/scikit-hep/awkward-1.0/pull/1019>`__: Fix: set ``numpy_to_regular=True`` in ``broadcast_arrays`` (**also:** `#1017 <https://github.com/scikit-hep/awkward-1.0/issues/1017>`__)
* PR `#959 <https://github.com/scikit-hep/awkward-1.0/pull/959>`__: C++ refactoring: _getitem_array implementation.
* PR `#1018 <https://github.com/scikit-hep/awkward-1.0/pull/1018>`__: [pre-commit.ci] pre-commit autoupdate.
* PR `#1016 <https://github.com/scikit-hep/awkward-1.0/pull/1016>`__: Documentation: clarify left & right broadcasting.
* PR `#1013 <https://github.com/scikit-hep/awkward-1.0/pull/1013>`__: Bugfix: do not use ``regular_to_jagged`` in ``ak.zip`` (**also:** `#1012 <https://github.com/scikit-hep/awkward-1.0/issues/1012>`__)
* PR `#1008 <https://github.com/scikit-hep/awkward-1.0/pull/1008>`__: Bugfix: fix ``ak.packed`` for ``RegularArray``s with ``.size=0`` (**also:** `#1006 <https://github.com/scikit-hep/awkward-1.0/issues/1006>`__)
* PR `#1004 <https://github.com/scikit-hep/awkward-1.0/pull/1004>`__: docs: touch up contributing.
* PR `#1009 <https://github.com/scikit-hep/awkward-1.0/pull/1009>`__: Bugfix: support empty buffers in ``from_buffers`` (**also:** `#1007 <https://github.com/scikit-hep/awkward-1.0/issues/1007>`__)
* PR `#1005 <https://github.com/scikit-hep/awkward-1.0/pull/1005>`__: Fixes `#595 <https://github.com/scikit-hep/awkward-1.0/issues/595>`__ and `#630 <https://github.com/scikit-hep/awkward-1.0/issues/630>`__; adds a default for NEP-18.
* PR `#985 <https://github.com/scikit-hep/awkward-1.0/pull/985>`__: Feature: add ``np.ravel``
* PR `#1001 <https://github.com/scikit-hep/awkward-1.0/pull/1001>`__: Fixes `#998 <https://github.com/scikit-hep/awkward-1.0/issues/998>`__ and `#1000 <https://github.com/scikit-hep/awkward-1.0/issues/1000>`__; argmax for ListOffsetArray with nonzero start. Also optimizes toListOffsetArray64(true).
* PR `#997 <https://github.com/scikit-hep/awkward-1.0/pull/997>`__: Fixes `#982 <https://github.com/scikit-hep/awkward-1.0/issues/982>`__ by accounting for an additional kind of 'gap' in nonlocal reducers.
* PR `#993 <https://github.com/scikit-hep/awkward-1.0/pull/993>`__: Fix high-level ak.Array.__dir__ to include methods and properties of overridden classes.
* PR `#994 <https://github.com/scikit-hep/awkward-1.0/pull/994>`__: Fixes two bugs in `#992 <https://github.com/scikit-hep/awkward-1.0/issues/992>`__: double-masking of reducers and unmasking of ak.ptp.
* PR `#995 <https://github.com/scikit-hep/awkward-1.0/pull/995>`__: Fixes `#546 <https://github.com/scikit-hep/awkward-1.0/issues/546>`__, ak.fill_none losing the replacement value's dtype.
* PR `#917 <https://github.com/scikit-hep/awkward-1.0/pull/917>`__: Feature: add axis parameter to ``ak.fill_none`` (**also:** `#920 <https://github.com/scikit-hep/awkward-1.0/issues/920>`__)
* PR `#991 <https://github.com/scikit-hep/awkward-1.0/pull/991>`__: Fixed bug `#770 <https://github.com/scikit-hep/awkward-1.0/issues/770>`__, `#930 <https://github.com/scikit-hep/awkward-1.0/issues/930>`__: not a policy issue; UnionForm::purelist_parameter was incorrectly comparing its contents' direct parameters, rather than their purelist_parameters.
* PR `#987 <https://github.com/scikit-hep/awkward-1.0/pull/987>`__: Feature: add GitHub Issue Forms.
* PR `#980 <https://github.com/scikit-hep/awkward-1.0/pull/980>`__: Bugfix: support n-dim ``NumpyArray``s in ``ak.where``
* PR `#988 <https://github.com/scikit-hep/awkward-1.0/pull/988>`__: fix: Unrestrict jaxlib upper bound and exclude jaxlib v0.1.68. (**also:** `#963 <https://github.com/scikit-hep/awkward-1.0/issues/963>`__)

Release `1.4.0 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.4.0>`__
==================================================================================

 (`pip <https://pypi.org/project/awkward/1.4.0/>`__)

*(no pull requests)*

Release `1.4.0rc6 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.4.0rc6>`__
========================================================================================

* PR `#976 <https://github.com/scikit-hep/awkward-1.0/pull/976>`__: Bugfix: support multidimensional NumPy mask arrays in ``ak.mask`` (**also:** `#975 <https://github.com/scikit-hep/awkward-1.0/issues/975>`__, `#975 <https://github.com/scikit-hep/awkward-1.0/issues/975>`__)

Release `1.4.0rc5 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.4.0rc5>`__
========================================================================================

* PR `#946 <https://github.com/scikit-hep/awkward-1.0/pull/946>`__: sorting an indexed option array in ``axis0`` bug fix; argsort to account ``None``s.
* PR `#974 <https://github.com/scikit-hep/awkward-1.0/pull/974>`__: Bugfix: fix `#973 <https://github.com/scikit-hep/awkward-1.0/issues/973>`__.

Release `1.4.0rc4 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.4.0rc4>`__
========================================================================================

* PR `#972 <https://github.com/scikit-hep/awkward-1.0/pull/972>`__: Feature: add layout transformer & simplify unpacked.
* PR `#970 <https://github.com/scikit-hep/awkward-1.0/pull/970>`__: Fix `#968 <https://github.com/scikit-hep/awkward-1.0/issues/968>`__, missing 'import awkward._io'.

Release `1.4.0rc3 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.4.0rc3>`__
========================================================================================

*(no pull requests)*

Release `1.4.0rc2 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.4.0rc2>`__
========================================================================================

* PR `#966 <https://github.com/scikit-hep/awkward-1.0/pull/966>`__: Make dev/generate-kernel-signatures.py part of the build process and add Python ctypes signatures as well.
* PR `#961 <https://github.com/scikit-hep/awkward-1.0/pull/961>`__: ci: move to cibw on GHA.

Release `1.4.0rc1 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.4.0rc1>`__
========================================================================================

* PR `#963 <https://github.com/scikit-hep/awkward-1.0/pull/963>`__: Test jaxlib<0.1.68 for segfault.
* PR `#962 <https://github.com/scikit-hep/awkward-1.0/pull/962>`__: C++ refactoring: convert and compare v1 and v2 arrays; renamed v2 recordarray -> keys.
* PR `#958 <https://github.com/scikit-hep/awkward-1.0/pull/958>`__: C++ refactoring: new Forms must accept old Form JSON.
* PR `#957 <https://github.com/scikit-hep/awkward-1.0/pull/957>`__: C++ refactoring: Type and Form classes - touchups.
* PR `#954 <https://github.com/scikit-hep/awkward-1.0/pull/954>`__: fix(setup): sync with cmake_example.
* PR `#953 <https://github.com/scikit-hep/awkward-1.0/pull/953>`__: tests: fix loading from any directory.
* PR `#914 <https://github.com/scikit-hep/awkward-1.0/pull/914>`__: C++ refactoring: Type and Form classes.
* PR `#955 <https://github.com/scikit-hep/awkward-1.0/pull/955>`__: fix: manylinux1 couldn't take None/newaxis.
* PR `#951 <https://github.com/scikit-hep/awkward-1.0/pull/951>`__: Writing documentation on 2021-06-23.
* PR `#952 <https://github.com/scikit-hep/awkward-1.0/pull/952>`__: Refactor: use ``maybe_wrap`` in source.
* PR `#950 <https://github.com/scikit-hep/awkward-1.0/pull/950>`__: Appropriate FileNotFoundError for ak.from_json.
* PR `#947 <https://github.com/scikit-hep/awkward-1.0/pull/947>`__: How much documentation can I get done today?
* PR `#935 <https://github.com/scikit-hep/awkward-1.0/pull/935>`__: Separate ``from_parquet`` into different routines.
* PR `#943 <https://github.com/scikit-hep/awkward-1.0/pull/943>`__: Tutorial documentation on 2021-06-18. (**also:** `#704 <https://github.com/scikit-hep/awkward-1.0/issues/704>`__)
* PR `#942 <https://github.com/scikit-hep/awkward-1.0/pull/942>`__: Bugfix: fix ``with_cache`` implementation.
* PR `#937 <https://github.com/scikit-hep/awkward-1.0/pull/937>`__: Bugfix: truncate ``ListOffsetArray`` contents.
* PR `#931 <https://github.com/scikit-hep/awkward-1.0/pull/931>`__: Writing tutorial documentation 2021-06-16.
* PR `#924 <https://github.com/scikit-hep/awkward-1.0/pull/924>`__: rename TypedArrayBuilder to LayoutBuilder.
* PR `#896 <https://github.com/scikit-hep/awkward-1.0/pull/896>`__: C++ refactoring: Content classes.
* PR `#929 <https://github.com/scikit-hep/awkward-1.0/pull/929>`__: Refactor: remove ``ak.deprecations_as_errors``
* PR `#874 <https://github.com/scikit-hep/awkward-1.0/pull/874>`__: Should strings from __getitem__ be Python str? (and bytes?)
* PR `#922 <https://github.com/scikit-hep/awkward-1.0/pull/922>`__: Bugfix: use ``ak.packed`` in ``ak.unflatten`` (**also:** `#910 <https://github.com/scikit-hep/awkward-1.0/issues/910>`__, `#910 <https://github.com/scikit-hep/awkward-1.0/issues/910>`__)
* PR `#928 <https://github.com/scikit-hep/awkward-1.0/pull/928>`__: Bugfix: return correct ``nbytes`` value for multidimensional NumPy arrays. (**also:** `#927 <https://github.com/scikit-hep/awkward-1.0/issues/927>`__, `#927 <https://github.com/scikit-hep/awkward-1.0/issues/927>`__)
* PR `#923 <https://github.com/scikit-hep/awkward-1.0/pull/923>`__: Bugfix: check for ``file``-like objects in ``from_parquet``
* PR `#925 <https://github.com/scikit-hep/awkward-1.0/pull/925>`__: Added '[todo]' to unwritten documentation, added a few nodes, and added description of ak.packed to how-to-convert-buffers.md.
* PR `#916 <https://github.com/scikit-hep/awkward-1.0/pull/916>`__: ``ak.values_astype`` support ``dtype`` specifier ``np.datetime64`` to convert ``?int`` or ``?float`` typed unix timestamps to ``datetime64``
* PR `#912 <https://github.com/scikit-hep/awkward-1.0/pull/912>`__: Feature: add ``ak.packed``
* PR `#919 <https://github.com/scikit-hep/awkward-1.0/pull/919>`__: [pre-commit.ci] pre-commit autoupdate.
* PR `#918 <https://github.com/scikit-hep/awkward-1.0/pull/918>`__: Feature: add ``pass_apply`` argument to util.
* PR `#915 <https://github.com/scikit-hep/awkward-1.0/pull/915>`__: revert to datetime64 and timedelta64.
* PR `#835 <https://github.com/scikit-hep/awkward-1.0/pull/835>`__: support 'datetime64' and 'timedelta64' types.
* PR `#907 <https://github.com/scikit-hep/awkward-1.0/pull/907>`__: WIP: initial support for reading ``pyarrow.lib.FixedSizeListType``
* PR `#904 <https://github.com/scikit-hep/awkward-1.0/pull/904>`__: ArrayView expects contiguous NumpyArrays, so make sure they're contiguous.
* PR `#901 <https://github.com/scikit-hep/awkward-1.0/pull/901>`__: Distinguish cache keys for non-leaf nodes.
* PR `#897 <https://github.com/scikit-hep/awkward-1.0/pull/897>`__: More precise Content documentation.
* PR `#895 <https://github.com/scikit-hep/awkward-1.0/pull/895>`__: Fixes `#894 <https://github.com/scikit-hep/awkward-1.0/issues/894>`__.
* PR `#884 <https://github.com/scikit-hep/awkward-1.0/pull/884>`__: C++ refactoring: Index and Identities (Identifier)
* PR `#890 <https://github.com/scikit-hep/awkward-1.0/pull/890>`__: Feature: add ``ak.ptp``
* PR `#891 <https://github.com/scikit-hep/awkward-1.0/pull/891>`__: Documentation: fix typo in reducers.

Release `1.3.0 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.3.0>`__
==================================================================================

 (`pip <https://pypi.org/project/awkward/1.3.0/>`__)

* PR `#868 <https://github.com/scikit-hep/awkward-1.0/pull/868>`__: Matrix multiplication of a non-array vector. (**also:** `#881 <https://github.com/scikit-hep/awkward-1.0/issues/881>`__)
* PR `#885 <https://github.com/scikit-hep/awkward-1.0/pull/885>`__: [pre-commit.ci] pre-commit autoupdate.
* PR `#880 <https://github.com/scikit-hep/awkward-1.0/pull/880>`__: Fix `#879 <https://github.com/scikit-hep/awkward-1.0/issues/879>`__.
* PR `#878 <https://github.com/scikit-hep/awkward-1.0/pull/878>`__: Fix some issues with null-typed Arrow/Parquet columns.
* PR `#877 <https://github.com/scikit-hep/awkward-1.0/pull/877>`__: Buffer pointers should come from ``data()``, not ``ptr().get()``. (**also:** `#876 <https://github.com/scikit-hep/awkward-1.0/issues/876>`__)
* PR `#871 <https://github.com/scikit-hep/awkward-1.0/pull/871>`__: Fixes for Parquet, Numba, Dask test.
* PR `#870 <https://github.com/scikit-hep/awkward-1.0/pull/870>`__: Simplify UnionArray::getitem_field(s) and ak.flatten axis=None.
* PR `#869 <https://github.com/scikit-hep/awkward-1.0/pull/869>`__: [pre-commit.ci] pre-commit autoupdate.
* PR `#867 <https://github.com/scikit-hep/awkward-1.0/pull/867>`__: Fix `#865 <https://github.com/scikit-hep/awkward-1.0/issues/865>`__, only build forms for columns that are requested in lazy mode (ak.from_parquet)
* PR `#864 <https://github.com/scikit-hep/awkward-1.0/pull/864>`__: Bugfix: possible fix for `#863 <https://github.com/scikit-hep/awkward-1.0/issues/863>`__.
* PR `#860 <https://github.com/scikit-hep/awkward-1.0/pull/860>`__: corrected the class names in documentation.
* PR `#862 <https://github.com/scikit-hep/awkward-1.0/pull/862>`__: [pre-commit.ci] pre-commit autoupdate.

Release `1.2.3 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.2.3>`__
==================================================================================

 (`pip <https://pypi.org/project/awkward/1.2.3/>`__)

*(no pull requests)*

Release `1.3.0rc4 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.3.0rc4>`__
========================================================================================

* PR `#858 <https://github.com/scikit-hep/awkward-1.0/pull/858>`__: arrays of complex types concatenate.
* PR `#856 <https://github.com/scikit-hep/awkward-1.0/pull/856>`__: [pre-commit.ci] pre-commit autoupdate.
* PR `#833 <https://github.com/scikit-hep/awkward-1.0/pull/833>`__: chore: cmake cleanup.
* PR `#834 <https://github.com/scikit-hep/awkward-1.0/pull/834>`__: style: setup.cfg formatting.
* PR `#855 <https://github.com/scikit-hep/awkward-1.0/pull/855>`__: This PR adds documentation for differentiation using JAX.
* PR `#850 <https://github.com/scikit-hep/awkward-1.0/pull/850>`__: masked array sort and argsort bug fix.
* PR `#847 <https://github.com/scikit-hep/awkward-1.0/pull/847>`__: Bugfix: fix `#846 <https://github.com/scikit-hep/awkward-1.0/issues/846>`__ - matrix multiplication with numpy array.
* PR `#851 <https://github.com/scikit-hep/awkward-1.0/pull/851>`__: [pre-commit.ci] pre-commit autoupdate.
* PR `#848 <https://github.com/scikit-hep/awkward-1.0/pull/848>`__: docs: add agoose77 as a contributor.
* PR `#844 <https://github.com/scikit-hep/awkward-1.0/pull/844>`__: Removed "ignore" from some Flake8 complaints.
* PR `#793 <https://github.com/scikit-hep/awkward-1.0/pull/793>`__: This PR integrates JAX element wise differentiation into the main codebase.
* PR `#842 <https://github.com/scikit-hep/awkward-1.0/pull/842>`__: [pre-commit.ci] pre-commit autoupdate.
* PR `#839 <https://github.com/scikit-hep/awkward-1.0/pull/839>`__: Allow scalars in ak.zip and make an ak.Record if they're all scalars.
* PR `#837 <https://github.com/scikit-hep/awkward-1.0/pull/837>`__: Handle ndim != 1 bool arrays in to_arrow (and hence, Parquet).
* PR `#831 <https://github.com/scikit-hep/awkward-1.0/pull/831>`__: chore: cleanup setup.py.
* PR `#814 <https://github.com/scikit-hep/awkward-1.0/pull/814>`__: Added dtype argument to ones/zeros/full_like functions.
* PR `#829 <https://github.com/scikit-hep/awkward-1.0/pull/829>`__: Handle Arrow's DataType(null).
* PR `#827 <https://github.com/scikit-hep/awkward-1.0/pull/827>`__: docs: add HenryDayHall as a contributor.
* PR `#826 <https://github.com/scikit-hep/awkward-1.0/pull/826>`__: fixed string inequality comparison.
* PR `#769 <https://github.com/scikit-hep/awkward-1.0/pull/769>`__: Typed Array Builder from Form.
* PR `#825 <https://github.com/scikit-hep/awkward-1.0/pull/825>`__: Fix ak.Record's promote to behavior.

Release `1.2.2 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.2.2>`__
==================================================================================

 (`pip <https://pypi.org/project/awkward/1.2.2/>`__)

* PR `#820 <https://github.com/scikit-hep/awkward-1.0/pull/820>`__: Fixes issue `#819 <https://github.com/scikit-hep/awkward-1.0/issues/819>`__: unflattening at axis>0 with a scalar.

Release `1.3.0rc3 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.3.0rc3>`__
========================================================================================

* PR `#816 <https://github.com/scikit-hep/awkward-1.0/pull/816>`__: Broadcast union types to all possibilities, even ones with no instances in the array.

Release `1.3.0rc2 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.3.0rc2>`__
========================================================================================

* PR `#812 <https://github.com/scikit-hep/awkward-1.0/pull/812>`__: argsort bugfix for empty arrays.

Release `1.3.0rc1 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.3.0rc1>`__
========================================================================================

* PR `#808 <https://github.com/scikit-hep/awkward-1.0/pull/808>`__: Issue `#805 <https://github.com/scikit-hep/awkward-1.0/issues/805>`__: fix an empty case when broadcasting UnionArrays.

Release `1.2.1 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.2.1>`__
==================================================================================

 (`pip <https://pypi.org/project/awkward/1.2.1/>`__)

* PR `#800 <https://github.com/scikit-hep/awkward-1.0/pull/800>`__: [pre-commit.ci] pre-commit autoupdate.
* PR `#803 <https://github.com/scikit-hep/awkward-1.0/pull/803>`__: argsort to return an index-type in the case that the input list array is empty.

Release `1.2.0 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.2.0>`__
==================================================================================

 (`pip <https://pypi.org/project/awkward/1.2.0/>`__)

*(no pull requests)*

Release `1.2.0rc6 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.2.0rc6>`__
========================================================================================

* PR `#799 <https://github.com/scikit-hep/awkward-1.0/pull/799>`__: Forbid 'pyarrow.lib.Tensor' in 'ak.to_parquet'.

Release `1.2.0rc5 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.2.0rc5>`__
========================================================================================

* PR `#796 <https://github.com/scikit-hep/awkward-1.0/pull/796>`__: Add mechanism to cast objects before __array_ufunc__.
* PR `#795 <https://github.com/scikit-hep/awkward-1.0/pull/795>`__: Fixed `#794 <https://github.com/scikit-hep/awkward-1.0/issues/794>`__, ak.cartesian on an empty array.

Release `1.2.0rc4 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.2.0rc4>`__
========================================================================================

* PR `#790 <https://github.com/scikit-hep/awkward-1.0/pull/790>`__: Implement 'np.nan_to_num' and 'np.isclose' for Vector.
* PR `#789 <https://github.com/scikit-hep/awkward-1.0/pull/789>`__: When RecordArrays are lazily carried as IndexedArrays, the IndexedArrays shouldn't copy the RecordArray parameters. Also, 'with_name' may be able to simplify some UnionArrays after homogenizing names.

Release `1.2.0rc3 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.2.0rc3>`__
========================================================================================

* PR `#787 <https://github.com/scikit-hep/awkward-1.0/pull/787>`__: Don't prevent other Numba extensions from using operators.
* PR `#779 <https://github.com/scikit-hep/awkward-1.0/pull/779>`__: Adapt getitem of DifferentiableArray for JAX.
* PR `#785 <https://github.com/scikit-hep/awkward-1.0/pull/785>`__: [pre-commit.ci] pre-commit autoupdate.
* PR `#786 <https://github.com/scikit-hep/awkward-1.0/pull/786>`__: ci: cleanup and remove 3.9 restrictions.
* PR `#784 <https://github.com/scikit-hep/awkward-1.0/pull/784>`__: Disambiguate offsets-index cache keys when lazily reading Parquet.
* PR `#781 <https://github.com/scikit-hep/awkward-1.0/pull/781>`__: AwkwardForth: add an s" core word to define strings.
* PR `#773 <https://github.com/scikit-hep/awkward-1.0/pull/773>`__: Add type parser to main codebase.
* PR `#776 <https://github.com/scikit-hep/awkward-1.0/pull/776>`__: Fix bug raised on StackOverflow.
* PR `#767 <https://github.com/scikit-hep/awkward-1.0/pull/767>`__: Conceptual test of TypedArrayBuilder through AwkwardForth.
* PR `#772 <https://github.com/scikit-hep/awkward-1.0/pull/772>`__: Fixes `#771 <https://github.com/scikit-hep/awkward-1.0/issues/771>`__; constructing Array with different length columns should raise error.
* PR `#668 <https://github.com/scikit-hep/awkward-1.0/pull/668>`__: Study writing a type parser for TypedArrayBuilder.
* PR `#766 <https://github.com/scikit-hep/awkward-1.0/pull/766>`__: Prevent combinations of characters (from a bug on Mattermost).
* PR `#690 <https://github.com/scikit-hep/awkward-1.0/pull/690>`__: array builder time profiler study.
* PR `#765 <https://github.com/scikit-hep/awkward-1.0/pull/765>`__: Consider this implementation of a DifferentiableArray for JAX.
* PR `#764 <https://github.com/scikit-hep/awkward-1.0/pull/764>`__: Fixes `#763 <https://github.com/scikit-hep/awkward-1.0/issues/763>`__ by assigning a better type to EmptyArray::argsort_next.
* PR `#762 <https://github.com/scikit-hep/awkward-1.0/pull/762>`__: More documentation, starting with "how to build".

Release `1.2.0rc2 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.2.0rc2>`__
========================================================================================

* PR `#760 <https://github.com/scikit-hep/awkward-1.0/pull/760>`__: Simplify 'ak.to_arrow' list handling.
* PR `#757 <https://github.com/scikit-hep/awkward-1.0/pull/757>`__: Fixes `#756 <https://github.com/scikit-hep/awkward-1.0/issues/756>`__; ak.num on PartitionedArrays.
* PR `#755 <https://github.com/scikit-hep/awkward-1.0/pull/755>`__: Print ak.layout.RecordArray's 'length' unequivocally to aid debugging.
* PR `#754 <https://github.com/scikit-hep/awkward-1.0/pull/754>`__: Minor tweak so that ak.unflatten works for CuPy arrays. This doesn't count as support yet.
* PR `#752 <https://github.com/scikit-hep/awkward-1.0/pull/752>`__: Fixes `#722 <https://github.com/scikit-hep/awkward-1.0/issues/722>`__; better error message for bad concatenation.
* PR `#751 <https://github.com/scikit-hep/awkward-1.0/pull/751>`__: Fixes `#740 <https://github.com/scikit-hep/awkward-1.0/issues/740>`__; NumPy scalars should be iterated over as numbers.
* PR `#750 <https://github.com/scikit-hep/awkward-1.0/pull/750>`__: Added 'ak.Array.to_list', 'ak.Array.to_numpy', and 'ak.Record.to_list', and simplified the documentation to point to the functions they call.
* PR `#748 <https://github.com/scikit-hep/awkward-1.0/pull/748>`__: I had somehow forgotten to handle 'row_groups' in 'ak.from_parquet'. Fixed now.
* PR `#743 <https://github.com/scikit-hep/awkward-1.0/pull/743>`__: ak.unflatten should include trailing zero-length counts in the array.

Release `1.2.0rc1 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.2.0rc1>`__
========================================================================================

* PR `#738 <https://github.com/scikit-hep/awkward-1.0/pull/738>`__: docs: add drahnreb as a contributor.
* PR `#736 <https://github.com/scikit-hep/awkward-1.0/pull/736>`__: Implement argsort for strings.
* PR `#737 <https://github.com/scikit-hep/awkward-1.0/pull/737>`__: Define the depth of an array of strings to be 1.
* PR `#735 <https://github.com/scikit-hep/awkward-1.0/pull/735>`__: Allow ak.run_lengths to recognize strings as distinguishable values.
* PR `#734 <https://github.com/scikit-hep/awkward-1.0/pull/734>`__: Implemented 'ak.strings_astype' to convert strings into numbers.
* PR `#733 <https://github.com/scikit-hep/awkward-1.0/pull/733>`__: Implemented 'ak.run_lengths' to enable group-by operations.
* PR `#732 <https://github.com/scikit-hep/awkward-1.0/pull/732>`__: Allow 'ak.unflatten' to be used on PartitionedArrays.
* PR `#731 <https://github.com/scikit-hep/awkward-1.0/pull/731>`__: Add an 'axis' parameter to 'ak.unflatten'.
* PR `#727 <https://github.com/scikit-hep/awkward-1.0/pull/727>`__: fix some issues of categorical arrays. (**also:** `#674 <https://github.com/scikit-hep/awkward-1.0/issues/674>`__)

Release `1.1.2 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.1.2>`__
==================================================================================

 (`pip <https://pypi.org/project/awkward/1.1.2/>`__)

* PR `#729 <https://github.com/scikit-hep/awkward-1.0/pull/729>`__: Only 'simplify'ed option-type and union-type arrays are now considered valid.
* PR `#726 <https://github.com/scikit-hep/awkward-1.0/pull/726>`__: Fixed `#724 <https://github.com/scikit-hep/awkward-1.0/issues/724>`__, a segfault in ak.flatten.
* PR `#725 <https://github.com/scikit-hep/awkward-1.0/pull/725>`__: Ensure that a jagged slice fits the array's length.
* PR `#720 <https://github.com/scikit-hep/awkward-1.0/pull/720>`__: fix: add missing files to the manifest, include a check.

Release `1.1.1 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.1.1>`__
==================================================================================

 (`pip <https://pypi.org/project/awkward/1.1.1/>`__)

* PR `#719 <https://github.com/scikit-hep/awkward-1.0/pull/719>`__: Prevent nullptr in PyArrayGenerator::caches.
* PR `#717 <https://github.com/scikit-hep/awkward-1.0/pull/717>`__: Every function with 'highlevel=True' gets 'behavior=None', which overrides behaviors from the input arrays. Also dropped 'ak.is_unique' because it isn't a well-designed high-level function.

Release `1.1.0 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.1.0>`__
==================================================================================

 (`pip <https://pypi.org/project/awkward/1.1.0/>`__)

*(no pull requests)*

Release `1.1.0rc5 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.1.0rc5>`__
========================================================================================

* PR `#714 <https://github.com/scikit-hep/awkward-1.0/pull/714>`__: getitem_field should simplify_optiontype (for option-type arrays).
* PR `#709 <https://github.com/scikit-hep/awkward-1.0/pull/709>`__: ARROW-10930 has been fixed, and we depend on it in the new Parquet-handling code, so the minimum Arrow is now 3.0.

Release `1.1.0rc4 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.1.0rc4>`__
========================================================================================

* PR `#707 <https://github.com/scikit-hep/awkward-1.0/pull/707>`__: Python 2 can use ellipsis now.
* PR `#706 <https://github.com/scikit-hep/awkward-1.0/pull/706>`__: Read and write Parquet datasets (sets of files).

Release `1.1.0rc3 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.1.0rc3>`__
========================================================================================

* PR `#699 <https://github.com/scikit-hep/awkward-1.0/pull/699>`__: Fixing more bugs revealed by prepping for SciPy.
* PR `#703 <https://github.com/scikit-hep/awkward-1.0/pull/703>`__: Fixed `#702 <https://github.com/scikit-hep/awkward-1.0/issues/702>`__, 'ak.to_arrow' with PartitionedArrays.

Release `1.1.0rc2 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.1.0rc2>`__
========================================================================================

* PR `#698 <https://github.com/scikit-hep/awkward-1.0/pull/698>`__: Remove ak.*_arrayset functions in preparation for 1.1.0.
* PR `#693 <https://github.com/scikit-hep/awkward-1.0/pull/693>`__: Fixes for SciPy 2021 prep. (**also:** `#694 <https://github.com/scikit-hep/awkward-1.0/issues/694>`__)
* PR `#697 <https://github.com/scikit-hep/awkward-1.0/pull/697>`__: add a check in a tuple builder for an out of bounds index.
* PR `#691 <https://github.com/scikit-hep/awkward-1.0/pull/691>`__: Fixes `#689 <https://github.com/scikit-hep/awkward-1.0/issues/689>`__, the dimension of arrays returned by empty slices.

Release `1.1.0rc1 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.1.0rc1>`__
========================================================================================

* PR `#688 <https://github.com/scikit-hep/awkward-1.0/pull/688>`__: Give lazy Parquet files information about their Forms.
* PR `#687 <https://github.com/scikit-hep/awkward-1.0/pull/687>`__: check content length before arg/sorting.
* PR `#683 <https://github.com/scikit-hep/awkward-1.0/pull/683>`__: refactor: pulling static info into setup.cfg.
* PR `#685 <https://github.com/scikit-hep/awkward-1.0/pull/685>`__: chore: flake8.
* PR `#676 <https://github.com/scikit-hep/awkward-1.0/pull/676>`__: chore: update pybind11 2.6.2.
* PR `#672 <https://github.com/scikit-hep/awkward-1.0/pull/672>`__: Put RNTuple measurements on the performance plot.
* PR `#677 <https://github.com/scikit-hep/awkward-1.0/pull/677>`__: docs: include GitHub button in GitHub dropdown.
* PR `#682 <https://github.com/scikit-hep/awkward-1.0/pull/682>`__: tests: fix B015.
* PR `#684 <https://github.com/scikit-hep/awkward-1.0/pull/684>`__: UnknownType documentation had a copy-paste error; fixed now.
* PR `#681 <https://github.com/scikit-hep/awkward-1.0/pull/681>`__: fix: flake8 F811.
* PR `#680 <https://github.com/scikit-hep/awkward-1.0/pull/680>`__: Remove right-broadcasting from most uses of 'broadcast_and_apply'. It's almost never what people want, and we're only obliged to maintain it in functions that generalize NumPy (like ufuncs and 'ak.where').
* PR `#652 <https://github.com/scikit-hep/awkward-1.0/pull/652>`__: complex numbers support.
* PR `#675 <https://github.com/scikit-hep/awkward-1.0/pull/675>`__: style: pre-commit.
* PR `#669 <https://github.com/scikit-hep/awkward-1.0/pull/669>`__: Avoid specifying target CUDA architecture.
* PR `#673 <https://github.com/scikit-hep/awkward-1.0/pull/673>`__: Fixes `#671 <https://github.com/scikit-hep/awkward-1.0/issues/671>`__ by allowing buffers in ak.from_buffers to be larger than strictly necessary (following the rules that define lengths of array nodes; https://awkward-array.readthedocs.io/en/latest/ak.layout.Content.html).
* PR `#670 <https://github.com/scikit-hep/awkward-1.0/pull/670>`__: fix warnings and run static analyser.
* PR `#667 <https://github.com/scikit-hep/awkward-1.0/pull/667>`__: Update CONTRIBUTING.md documentation.
* PR `#666 <https://github.com/scikit-hep/awkward-1.0/pull/666>`__: Remove unused variable from dev/generate-cuda.py.
* PR `#665 <https://github.com/scikit-hep/awkward-1.0/pull/665>`__: Fix generated file cleanup script.
* PR `#664 <https://github.com/scikit-hep/awkward-1.0/pull/664>`__: Actually remember to release the GIL before doing some multithreading tests.
* PR `#661 <https://github.com/scikit-hep/awkward-1.0/pull/661>`__: AwkwardForth additions to make Avro and Parquet parsing possible.
* PR `#662 <https://github.com/scikit-hep/awkward-1.0/pull/662>`__: Cleanups after PR `#648 <https://github.com/scikit-hep/awkward-1.0/issues/648>`__.
* PR `#645 <https://github.com/scikit-hep/awkward-1.0/pull/645>`__: This PR adds the from_cuda_array_interface, to form CUDA arrays in a more general fashion.
* PR `#657 <https://github.com/scikit-hep/awkward-1.0/pull/657>`__: With the new lazy slices, it's now possible to get into 'VirtualArray::getitem_next_jagged'.
* PR `#653 <https://github.com/scikit-hep/awkward-1.0/pull/653>`__: Bugfixes in ForthMachine (discovered by writing documentation).
* PR `#656 <https://github.com/scikit-hep/awkward-1.0/pull/656>`__: `scikit-hep/uproot4#244 <https://github.com/scikit-hep/uproot4/issues/244>`__ revealed surprising semantics of ak.zip with regular arrays. Regular array case has been changed to act the same as jagged arrays.
* PR `#648 <https://github.com/scikit-hep/awkward-1.0/pull/648>`__: Add a ForthMachine to the codebase, copying from 'studies'.
* PR `#650 <https://github.com/scikit-hep/awkward-1.0/pull/650>`__: Fixes `#649 <https://github.com/scikit-hep/awkward-1.0/issues/649>`__, wording in documentation.
* PR `#647 <https://github.com/scikit-hep/awkward-1.0/pull/647>`__: Fix JupyterBook formatting and add the executable notebooks to CI tests.
* PR `#646 <https://github.com/scikit-hep/awkward-1.0/pull/646>`__: Black and flake8.
* PR `#639 <https://github.com/scikit-hep/awkward-1.0/pull/639>`__: add quick sort and argsort without recursion.
* PR `#638 <https://github.com/scikit-hep/awkward-1.0/pull/638>`__: Prototype Forth virtual machine in C++ (close to what will be added to Awkward Array).
* PR `#644 <https://github.com/scikit-hep/awkward-1.0/pull/644>`__: Querying array depth should never materialize if anything has a Form.
* PR `#643 <https://github.com/scikit-hep/awkward-1.0/pull/643>`__: Prevent trivial carrying, which can also prevent materialization of some VirtualArrays.

Release `1.0.2 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.0.2>`__
==================================================================================

 (`pip <https://pypi.org/project/awkward/1.0.2/>`__)

* PR `#642 <https://github.com/scikit-hep/awkward-1.0/pull/642>`__: Materialize virtual arrays in to_arrow.

Release `1.0.2rc5 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.0.2rc5>`__
========================================================================================

* PR `#620 <https://github.com/scikit-hep/awkward-1.0/pull/620>`__: Prototype Forth VM for filling Awkward Arrays from Uproot.
* PR `#636 <https://github.com/scikit-hep/awkward-1.0/pull/636>`__: Accept NumPy integers in slices.

Release `1.0.2rc4 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.0.2rc4>`__
========================================================================================

* PR `#635 <https://github.com/scikit-hep/awkward-1.0/pull/635>`__: Fixes `#634 <https://github.com/scikit-hep/awkward-1.0/issues/634>`__.
* PR `#631 <https://github.com/scikit-hep/awkward-1.0/pull/631>`__: Fixes `#629 <https://github.com/scikit-hep/awkward-1.0/issues/629>`__.
* PR `#627 <https://github.com/scikit-hep/awkward-1.0/pull/627>`__: A dict of arrays requires 'behaviorof(*arrays.values())'.
* PR `#626 <https://github.com/scikit-hep/awkward-1.0/pull/626>`__: Homogenize and streamline pass-through of Array behavior.
* PR `#625 <https://github.com/scikit-hep/awkward-1.0/pull/625>`__: Fixes `#624 <https://github.com/scikit-hep/awkward-1.0/issues/624>`__: unhandled 'offsets' in pyarrow arrays.
* PR `#619 <https://github.com/scikit-hep/awkward-1.0/pull/619>`__: Implemented column selection within multiple records.
* PR `#617 <https://github.com/scikit-hep/awkward-1.0/pull/617>`__: Created a roadmap.

Release `1.0.2rc3 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.0.2rc3>`__
========================================================================================

* PR `#571 <https://github.com/scikit-hep/awkward-1.0/pull/571>`__: validity checking based on known parameters.
* PR `#613 <https://github.com/scikit-hep/awkward-1.0/pull/613>`__: ak.concatenate should minimally touch lazy arrays. (**also:** `#603 <https://github.com/scikit-hep/awkward-1.0/issues/603>`__)
* PR `#612 <https://github.com/scikit-hep/awkward-1.0/pull/612>`__: Added 'axis_wrap_if_negative' to PartitionedArray.
* PR `#611 <https://github.com/scikit-hep/awkward-1.0/pull/611>`__: Add setuptools to requirements for pkg_resources.
* PR `#610 <https://github.com/scikit-hep/awkward-1.0/pull/610>`__: Revise concatenate with axis != 0.

Release `1.0.2rc2 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.0.2rc2>`__
========================================================================================

*(no pull requests)*

Release `1.0.2rc1 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.0.2rc1>`__
========================================================================================

* PR `#606 <https://github.com/scikit-hep/awkward-1.0/pull/606>`__: More complicated example revealed bugs in Arrow conversion.
* PR `#605 <https://github.com/scikit-hep/awkward-1.0/pull/605>`__: fix: avoid hardcoded threads and macOS target.
* PR `#604 <https://github.com/scikit-hep/awkward-1.0/pull/604>`__: Make tests work in 32-bit.
* PR `#602 <https://github.com/scikit-hep/awkward-1.0/pull/602>`__: Use pyarrow.field to preserve nullability in Arrow conversion.
* PR `#599 <https://github.com/scikit-hep/awkward-1.0/pull/599>`__: Preemtively avoid warnings in NumPy 1.20 (untested).

Release `1.0.1 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.0.1>`__
==================================================================================

 (`pip <https://pypi.org/project/awkward/1.0.1/>`__)

* PR `#598 <https://github.com/scikit-hep/awkward-1.0/pull/598>`__: Fixes ak.from_buffers failure for ListArray.

Release `1.0.1rc3 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.0.1rc3>`__
========================================================================================

*(no pull requests)*

Release `1.0.1rc2 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.0.1rc2>`__
========================================================================================

* PR `#592 <https://github.com/scikit-hep/awkward-1.0/pull/592>`__: Replace to_arrayset/from_arrayset with to_buffers/from_buffers and deprecate the original.
* PR `#590 <https://github.com/scikit-hep/awkward-1.0/pull/590>`__: Change the definition of RegularArray to accept size == 0.
* PR `#583 <https://github.com/scikit-hep/awkward-1.0/pull/583>`__: Implement unflatten function.
* PR `#591 <https://github.com/scikit-hep/awkward-1.0/pull/591>`__: Now 'ak.to_numpy(ak.layout.NumpyArray(cupy.array([1, 2, 3]))' works.

Release `1.0.1rc1 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.0.1rc1>`__
========================================================================================

* PR `#587 <https://github.com/scikit-hep/awkward-1.0/pull/587>`__: Modernize ak.is_none and add an 'axis' parameter.
* PR `#586 <https://github.com/scikit-hep/awkward-1.0/pull/586>`__: Fix corner-case revealed by issue `#585 <https://github.com/scikit-hep/awkward-1.0/issues/585>`__, but distinct from that issue.
* PR `#582 <https://github.com/scikit-hep/awkward-1.0/pull/582>`__: Propagate 'posaxis' through broadcast_and_apply and recursively_apply, then implement 'axis < 0' for some functions.
* PR `#578 <https://github.com/scikit-hep/awkward-1.0/pull/578>`__: Implement ak.Array.ndim in the Numba context.
* PR `#577 <https://github.com/scikit-hep/awkward-1.0/pull/577>`__: Fix the setup.py --record argument, which is needed for bdist_rpm.
* PR `#576 <https://github.com/scikit-hep/awkward-1.0/pull/576>`__: Actually remove deprecated features.

Release `1.0.0 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.0.0>`__
==================================================================================

 (`pip <https://pypi.org/project/awkward/1.0.0/>`__)

* PR `#573 <https://github.com/scikit-hep/awkward-1.0/pull/573>`__: Fix ak.from_awkward0's Table (missing _view).
* PR `#570 <https://github.com/scikit-hep/awkward-1.0/pull/570>`__: Fix ArrayBuilder memory leak.

Release `1.0.0rc2 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.0.0rc2>`__
========================================================================================

* PR `#569 <https://github.com/scikit-hep/awkward-1.0/pull/569>`__: Rename 'master' branch to 'main'.
* PR `#568 <https://github.com/scikit-hep/awkward-1.0/pull/568>`__: Options for "NaN" strings as NaN floats.
* PR `#454 <https://github.com/scikit-hep/awkward-1.0/pull/454>`__: Add some more Loop Dependent Kernels.
* PR `#565 <https://github.com/scikit-hep/awkward-1.0/pull/565>`__: Tool to check if kernel is implemented in all places.
* PR `#566 <https://github.com/scikit-hep/awkward-1.0/pull/566>`__: Added the 'initial' argument to ak.min/ak.max.

Release `1.0.0rc1 <https://github.com/scikit-hep/awkward-1.0/releases/tag/1.0.0rc1>`__
========================================================================================

*(no pull requests)*

Release `0.4.5 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.4.5>`__
==================================================================================

 (`pip <https://pypi.org/project/awkward1/0.4.5/>`__)

* PR `#553 <https://github.com/scikit-hep/awkward-1.0/pull/553>`__: ak.with_field should not right-broadcast (by default).
* PR `#551 <https://github.com/scikit-hep/awkward-1.0/pull/551>`__: Also implemented ak.to_regular and ak.from_regular.
* PR `#548 <https://github.com/scikit-hep/awkward-1.0/pull/548>`__: concatenate bug-fix. (**also:** `#545 <https://github.com/scikit-hep/awkward-1.0/issues/545>`__)
* PR `#550 <https://github.com/scikit-hep/awkward-1.0/pull/550>`__: Implemented 'np.array(ak.Array)' in Numba.
* PR `#547 <https://github.com/scikit-hep/awkward-1.0/pull/547>`__: Implemented '__contains__' in and out of Numba.
* PR `#524 <https://github.com/scikit-hep/awkward-1.0/pull/524>`__: argsort and sort for indexed option arrays bug fix.
* PR `#544 <https://github.com/scikit-hep/awkward-1.0/pull/544>`__: Simplified and generalized ak.where using broadcast_and_apply.
* PR `#542 <https://github.com/scikit-hep/awkward-1.0/pull/542>`__: Pickle-able mixin classes from decorator.

Release `0.4.4 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.4.4>`__
==================================================================================

 (`pip <https://pypi.org/project/awkward1/0.4.4/>`__)

* PR `#539 <https://github.com/scikit-hep/awkward-1.0/pull/539>`__: concatenate for a nonzero axis operation.
* PR `#540 <https://github.com/scikit-hep/awkward-1.0/pull/540>`__: Fix issue `#538 <https://github.com/scikit-hep/awkward-1.0/issues/538>`__'s performance issues.
* PR `#537 <https://github.com/scikit-hep/awkward-1.0/pull/537>`__: Fix matrix multiplication.
* PR `#536 <https://github.com/scikit-hep/awkward-1.0/pull/536>`__: Enforce minimum Arrow version 2.0.0 and fix issues due to ARROW-9556.
* PR `#535 <https://github.com/scikit-hep/awkward-1.0/pull/535>`__: Implemented zeros_like, ones_like, and full_like, and fixed from_numpy for NumPy string arrays.
* PR `#527 <https://github.com/scikit-hep/awkward-1.0/pull/527>`__: Fix UnionArray ufuncs and parameters in merging. (**also:** `#459 <https://github.com/scikit-hep/awkward-1.0/issues/459>`__, `#522 <https://github.com/scikit-hep/awkward-1.0/issues/522>`__, `#459 <https://github.com/scikit-hep/awkward-1.0/issues/459>`__, `#522 <https://github.com/scikit-hep/awkward-1.0/issues/522>`__)
* PR `#495 <https://github.com/scikit-hep/awkward-1.0/pull/495>`__: Add a developer tool to check if kernel specification file is sorted.
* PR `#526 <https://github.com/scikit-hep/awkward-1.0/pull/526>`__: Fix (infinite) recursion bug in Arrow translation.
* PR `#525 <https://github.com/scikit-hep/awkward-1.0/pull/525>`__: Fix `#402 <https://github.com/scikit-hep/awkward-1.0/issues/402>`__: Form::getitem_field must return the Form of what Content::getitem_field would return.
* PR `#520 <https://github.com/scikit-hep/awkward-1.0/pull/520>`__: Actually remove expired deprecations (they were supposed to go in 0.4.0).
* PR `#519 <https://github.com/scikit-hep/awkward-1.0/pull/519>`__: Provide ak.local_index.
* PR `#518 <https://github.com/scikit-hep/awkward-1.0/pull/518>`__: to_pandas with IndexedArrays (and other types)
* PR `#517 <https://github.com/scikit-hep/awkward-1.0/pull/517>`__: Masked take on an empty array should behave in a way that is consistent with non-empty arrays.

Release `0.4.3 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.4.3>`__
==================================================================================

 (`pip <https://pypi.org/project/awkward1/0.4.3/>`__)

* PR `#515 <https://github.com/scikit-hep/awkward-1.0/pull/515>`__: Fix fall-through that happened with the deprecation message.
* PR `#514 <https://github.com/scikit-hep/awkward-1.0/pull/514>`__: Provides high-level access to copy and deepcopy operations.
* PR `#513 <https://github.com/scikit-hep/awkward-1.0/pull/513>`__: Blocked ufuncs on custom types, reopened them for categoricals using a new apply_ufunc interface, updated documentation.

Release `0.4.2 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.4.2>`__
==================================================================================

 (`pip <https://pypi.org/project/awkward1/0.4.2/>`__)

* PR `#512 <https://github.com/scikit-hep/awkward-1.0/pull/512>`__: Always keep references to all caches in a ak.Array.

Release `0.4.1 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.4.1>`__
==================================================================================

 (`pip <https://pypi.org/project/awkward1/0.4.1/>`__)

* PR `#510 <https://github.com/scikit-hep/awkward-1.0/pull/510>`__: Remove broadcasting over the fields of records.
* PR `#508 <https://github.com/scikit-hep/awkward-1.0/pull/508>`__: Fixes `#499 <https://github.com/scikit-hep/awkward-1.0/issues/499>`__ by removing gaps from ListArray::content.
* PR `#507 <https://github.com/scikit-hep/awkward-1.0/pull/507>`__: Fixes `#501 <https://github.com/scikit-hep/awkward-1.0/issues/501>`__ and generalizes from_numpy/to_layout to accept NumPy arrays of strings.
* PR `#505 <https://github.com/scikit-hep/awkward-1.0/pull/505>`__: Superflous line.

Release `0.4.0 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.4.0>`__
==================================================================================

 (`pip <https://pypi.org/project/awkward1/0.4.0/>`__)

* PR `#482 <https://github.com/scikit-hep/awkward-1.0/pull/482>`__: Add Python 3.9 to tests and deployment.
* PR `#488 <https://github.com/scikit-hep/awkward-1.0/pull/488>`__: Add conda install instructions.
* PR `#409 <https://github.com/scikit-hep/awkward-1.0/pull/409>`__: Fill out the 'creating arrays' section.
* PR `#485 <https://github.com/scikit-hep/awkward-1.0/pull/485>`__: Update to pybind11 2.5.0. (**also:** `#483 <https://github.com/scikit-hep/awkward-1.0/issues/483>`__)
* PR `#478 <https://github.com/scikit-hep/awkward-1.0/pull/478>`__: Fix broken CUDA tests.
* PR `#473 <https://github.com/scikit-hep/awkward-1.0/pull/473>`__: Properly broadcast over empty ListArray.
* PR `#477 <https://github.com/scikit-hep/awkward-1.0/pull/477>`__: CPU kernel source files will have 1 file per kernel.
* PR `#474 <https://github.com/scikit-hep/awkward-1.0/pull/474>`__: Fix broken links.
* PR `#471 <https://github.com/scikit-hep/awkward-1.0/pull/471>`__: We do not need blacklists for kernel/test generation anymore.
* PR `#468 <https://github.com/scikit-hep/awkward-1.0/pull/468>`__: Generate kernel header files from specification.

Release `0.3.1 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.3.1>`__
==================================================================================

 (`pip <https://pypi.org/project/awkward1/0.3.1/>`__)

* PR `#470 <https://github.com/scikit-hep/awkward-1.0/pull/470>`__: Put the 'cmake' PyPI package back into pyproject.toml.
* PR `#469 <https://github.com/scikit-hep/awkward-1.0/pull/469>`__: awkward1 must use THE SAME VERSION of awkward1-cuda-kernels when it uses any.

Release `0.3.0 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.3.0>`__
==================================================================================

 (`pip <https://pypi.org/project/awkward1/0.3.0/>`__)

* PR `#467 <https://github.com/scikit-hep/awkward-1.0/pull/467>`__: Try to solve Windows installation issue by always compiling in 'Release' mode.
* PR `#455 <https://github.com/scikit-hep/awkward-1.0/pull/455>`__: Check kernel specification into git.
* PR `#464 <https://github.com/scikit-hep/awkward-1.0/pull/464>`__: Deprecate 'keys' -> 'fields' and add properties to ak.Array and ak.Record.
* PR `#465 <https://github.com/scikit-hep/awkward-1.0/pull/465>`__: Attempt to make check for -A x64 flag more robust.
* PR `#463 <https://github.com/scikit-hep/awkward-1.0/pull/463>`__: Generalize NumPy ufunc behavioral matching.
* PR `#461 <https://github.com/scikit-hep/awkward-1.0/pull/461>`__: Apply `conda-forge/awkward1-feedstock#2 <https://github.com/conda-forge/awkward1-feedstock/issues/2>`__ here to test it in our CI.
* PR `#462 <https://github.com/scikit-hep/awkward-1.0/pull/462>`__: Remove ak.Array.tojson for 0.3.0 (use ak.to_json).
* PR `#460 <https://github.com/scikit-hep/awkward-1.0/pull/460>`__: Remove Awkward-as-a-Pandas-column feature, as discussed in `#350 <https://github.com/scikit-hep/awkward-1.0/issues/350>`__.

Release `0.2.38 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.2.38>`__
====================================================================================

 (`pip <https://pypi.org/project/awkward1/0.2.38/>`__)

* PR `#458 <https://github.com/scikit-hep/awkward-1.0/pull/458>`__: Change the Windows build following @chrisburr's suggestion.
* PR `#456 <https://github.com/scikit-hep/awkward-1.0/pull/456>`__: Add numpy boolean type to ak describe.
* PR `#453 <https://github.com/scikit-hep/awkward-1.0/pull/453>`__: Add more kernel test cases.
* PR `#452 <https://github.com/scikit-hep/awkward-1.0/pull/452>`__: Add description field to specification.
* PR `#436 <https://github.com/scikit-hep/awkward-1.0/pull/436>`__: Working on the Loop Dependent Kernels.
* PR `#451 <https://github.com/scikit-hep/awkward-1.0/pull/451>`__: Change kernel specification format.
* PR `#450 <https://github.com/scikit-hep/awkward-1.0/pull/450>`__: Explicitly cast list to CuPy array in cumsum test.

Release `0.2.37 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.2.37>`__
====================================================================================

 (`pip <https://pypi.org/project/awkward1/0.2.37/>`__)

* PR `#448 <https://github.com/scikit-hep/awkward-1.0/pull/448>`__: Set up interface between Uproot and Awkward so that Awkward can be used to optimize object-reading. (**also:** `#449 <https://github.com/scikit-hep/awkward-1.0/issues/449>`__)
* PR `#449 <https://github.com/scikit-hep/awkward-1.0/pull/449>`__: Upgrade Content::merge from a single 'other' argument to a std::vector of 'others'.
* PR `#433 <https://github.com/scikit-hep/awkward-1.0/pull/433>`__: Auto generate more CUDA kernels.
* PR `#447 <https://github.com/scikit-hep/awkward-1.0/pull/447>`__: Fix reducer dimension regularity. (**also:** `#434 <https://github.com/scikit-hep/awkward-1.0/issues/434>`__)
* PR `#446 <https://github.com/scikit-hep/awkward-1.0/pull/446>`__: Fix ak.flatten for arrays that have been sliced.
* PR `#435 <https://github.com/scikit-hep/awkward-1.0/pull/435>`__: intermittent error bugfix.
* PR `#444 <https://github.com/scikit-hep/awkward-1.0/pull/444>`__: Always assuming ListArrays/ListOffsetArrays have incompatible structure is too conservative. Check for consistency and shortcut if possible.
* PR `#441 <https://github.com/scikit-hep/awkward-1.0/pull/441>`__: Fix typo in documentation.
* PR `#440 <https://github.com/scikit-hep/awkward-1.0/pull/440>`__: Fix cuda shared object retrieval.
* PR `#438 <https://github.com/scikit-hep/awkward-1.0/pull/438>`__: Restructure test generation - store only roles in specification.

Release `0.2.36 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.2.36>`__
====================================================================================

 (`pip <https://pypi.org/project/awkward1/0.2.36/>`__)

* PR `#439 <https://github.com/scikit-hep/awkward-1.0/pull/439>`__: Libraries and includes should only go into the Python directory, not multiple places.
* PR `#429 <https://github.com/scikit-hep/awkward-1.0/pull/429>`__: Generate more CUDA kernels.
* PR `#431 <https://github.com/scikit-hep/awkward-1.0/pull/431>`__: Allow ArrayCache(None) construction.
* PR `#428 <https://github.com/scikit-hep/awkward-1.0/pull/428>`__: Fix a bug in repr due to weak cache refs.
* PR `#427 <https://github.com/scikit-hep/awkward-1.0/pull/427>`__: Implement weak reference to virtual array caches.
* PR `#426 <https://github.com/scikit-hep/awkward-1.0/pull/426>`__: Generate tests for CUDA kernels.
* PR `#425 <https://github.com/scikit-hep/awkward-1.0/pull/425>`__: Fixes `#186 <https://github.com/scikit-hep/awkward-1.0/issues/186>`__; proper string-escape sequences in util::quote.
* PR `#373 <https://github.com/scikit-hep/awkward-1.0/pull/373>`__: Generate CUDA kernels from kernel specification.
* PR `#424 <https://github.com/scikit-hep/awkward-1.0/pull/424>`__: Change const representation in kernel spec.
* PR `#422 <https://github.com/scikit-hep/awkward-1.0/pull/422>`__: Forward purelist_parameter "__doc__" in lazy slices.
* PR `#423 <https://github.com/scikit-hep/awkward-1.0/pull/423>`__: docs: add sjperkins as a contributor.
* PR `#420 <https://github.com/scikit-hep/awkward-1.0/pull/420>`__: NumpyArray::bytelength and NumpyArray::carry were wrong for non-contiguous; fixed. (**also:** `#418 <https://github.com/scikit-hep/awkward-1.0/issues/418>`__)

Release `0.2.35 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.2.35>`__
====================================================================================

 (`pip <https://pypi.org/project/awkward1/0.2.35/>`__)

*(no pull requests)*

Release `0.2.34 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.2.34>`__
====================================================================================

 (`pip <https://pypi.org/project/awkward1/0.2.34/>`__)

* PR `#417 <https://github.com/scikit-hep/awkward-1.0/pull/417>`__: Fix nesting structure of arrays passed to ArrayBuilder.append.
* PR `#416 <https://github.com/scikit-hep/awkward-1.0/pull/416>`__: Fix references to np.object (where 'np' is a NumpyMetadata singleton).
* PR `#372 <https://github.com/scikit-hep/awkward-1.0/pull/372>`__: Add to and from_cupy for Numpy Array and Identities.
* PR `#414 <https://github.com/scikit-hep/awkward-1.0/pull/414>`__: Restructure test locations.
* PR `#413 <https://github.com/scikit-hep/awkward-1.0/pull/413>`__: Better error message if unable to find cpu-kernel shared object.
* PR `#412 <https://github.com/scikit-hep/awkward-1.0/pull/412>`__: Add const info to kernel spec.

Release `0.2.33 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.2.33>`__
====================================================================================

 (`pip <https://pypi.org/project/awkward1/0.2.33/>`__)

* PR `#410 <https://github.com/scikit-hep/awkward-1.0/pull/410>`__: Fix argmin/max positions for missing values.
* PR `#407 <https://github.com/scikit-hep/awkward-1.0/pull/407>`__: Fix warnings reported by MacOS/Cling and try PIP_ONLY_BINARY instead of restricting cmake version.
* PR `#406 <https://github.com/scikit-hep/awkward-1.0/pull/406>`__: Handle cases where C for loop was translated to Python while loop.
* PR `#403 <https://github.com/scikit-hep/awkward-1.0/pull/403>`__: Introduce a 'categorical' type (behavioral, just as 'string' is) that is the only thing that passes to Arrow as DictionaryArray.
* PR `#316 <https://github.com/scikit-hep/awkward-1.0/pull/316>`__: libawkward export tuning.

Release `0.2.32 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.2.32>`__
====================================================================================

 (`pip <https://pypi.org/project/awkward1/0.2.32/>`__)

* PR `#401 <https://github.com/scikit-hep/awkward-1.0/pull/401>`__: Missed some validity error URL reporting in `#399 <https://github.com/scikit-hep/awkward-1.0/issues/399>`__, finished them here (because they were discovered as part of the Arrow bug-hunt).
* PR `#390 <https://github.com/scikit-hep/awkward-1.0/pull/390>`__: VirtualArray has correct __record__ parameter.
* PR `#399 <https://github.com/scikit-hep/awkward-1.0/pull/399>`__: All exceptions should report their version, file, and line number as GitHub links.
* PR `#397 <https://github.com/scikit-hep/awkward-1.0/pull/397>`__: Awkward data should be registered as Numba constants so that they can be passed as closures to Numba-compiled functions. (**also:** `#175 <https://github.com/scikit-hep/awkward-1.0/issues/175>`__)
* PR `#398 <https://github.com/scikit-hep/awkward-1.0/pull/398>`__: Truncate generated outparams only to required length.
* PR `#396 <https://github.com/scikit-hep/awkward-1.0/pull/396>`__: Fixes `#395 <https://github.com/scikit-hep/awkward-1.0/issues/395>`__: IndexedArray was updating both the ArrayView viewport and 'nextat'; only one is allowed.
* PR `#388 <https://github.com/scikit-hep/awkward-1.0/pull/388>`__: Abstract all uses of NumPy, so that GPU arrays will use CuPy instead.
* PR `#394 <https://github.com/scikit-hep/awkward-1.0/pull/394>`__: Fixed `#393 <https://github.com/scikit-hep/awkward-1.0/issues/393>`__ (BitMaskedArray::bytemask output should be equivalent to valid_when=False).

Release `0.2.31 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.2.31>`__
====================================================================================

 (`pip <https://pypi.org/project/awkward1/0.2.31/>`__)

* PR `#389 <https://github.com/scikit-hep/awkward-1.0/pull/389>`__: Fix that cuda-kernels build!

Release `0.2.30 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.2.30>`__
====================================================================================

 (`pip <https://pypi.org/project/awkward1/0.2.30/>`__)

* PR `#381 <https://github.com/scikit-hep/awkward-1.0/pull/381>`__: Writing more awkward-array.org documentation. (**also:** `#387 <https://github.com/scikit-hep/awkward-1.0/issues/387>`__)
* PR `#384 <https://github.com/scikit-hep/awkward-1.0/pull/384>`__: Make from_arrayset even more lazy.
* PR `#385 <https://github.com/scikit-hep/awkward-1.0/pull/385>`__: Fixed `#383 <https://github.com/scikit-hep/awkward-1.0/issues/383>`__, prevented conversion of characters in strings, and renamed ak.numbers_to_type -> ak.values_astype.
* PR `#382 <https://github.com/scikit-hep/awkward-1.0/pull/382>`__: Fixed broken kernel page in sphinx docs.

Release `0.2.29 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.2.29>`__
====================================================================================

 (`pip <https://pypi.org/project/awkward1/0.2.29/>`__)

* PR `#379 <https://github.com/scikit-hep/awkward-1.0/pull/379>`__: Fill in a lot of stubs on awkward-array.org.
* PR `#378 <https://github.com/scikit-hep/awkward-1.0/pull/378>`__: Update the what-is-awkward to align with (and include) the video.

Release `0.2.28 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.2.28>`__
====================================================================================

 (`pip <https://pypi.org/project/awkward1/0.2.28/>`__)

* PR `#377 <https://github.com/scikit-hep/awkward-1.0/pull/377>`__: Rename 'astype' to 'numbers_to_type', for use as a high-level function. Also removed 'can_cast', since it's exactly the same as the NumPy version(users should use NumPy).
* PR `#374 <https://github.com/scikit-hep/awkward-1.0/pull/374>`__: Pandas deprecation version is 0.3.0 and ak.to_pandas is documented.
* PR `#371 <https://github.com/scikit-hep/awkward-1.0/pull/371>`__: Removing the last offset parameters (missed them before because they're in an array).
* PR `#346 <https://github.com/scikit-hep/awkward-1.0/pull/346>`__: Operation to change the number type.
* PR `#366 <https://github.com/scikit-hep/awkward-1.0/pull/366>`__: Remove 'offset' arguments from all kernels, only passing in pointers that have already been offset.
* PR `#369 <https://github.com/scikit-hep/awkward-1.0/pull/369>`__: Improve C++ to C generator.
* PR `#365 <https://github.com/scikit-hep/awkward-1.0/pull/365>`__: Do not iterate over lists while comparing in pytest.
* PR `#307 <https://github.com/scikit-hep/awkward-1.0/pull/307>`__: Create specification and generate tests for kernels based on hand written labels.
* PR `#364 <https://github.com/scikit-hep/awkward-1.0/pull/364>`__: Add Pandas deprecation warnings and other clean-ups.
* PR `#363 <https://github.com/scikit-hep/awkward-1.0/pull/363>`__: jupyter-books 0.7.3 no longer supports 'headers'. (**also:** `#350 <https://github.com/scikit-hep/awkward-1.0/issues/350>`__)
* PR `#357 <https://github.com/scikit-hep/awkward-1.0/pull/357>`__: Cleanup the Docker Residues like .test files and .dockerignore.
* PR `#360 <https://github.com/scikit-hep/awkward-1.0/pull/360>`__: Ensure that Matplotlib raises ValueError on non-flat arrays and keep test_0341 from leaking Parquet files.
* PR `#358 <https://github.com/scikit-hep/awkward-1.0/pull/358>`__: Fixed typo.
* PR `#354 <https://github.com/scikit-hep/awkward-1.0/pull/354>`__: Mixin class decorators.
* PR `#356 <https://github.com/scikit-hep/awkward-1.0/pull/356>`__: Adapt to pyarrow/Arrow 1.0.
* PR `#353 <https://github.com/scikit-hep/awkward-1.0/pull/353>`__: Add docstring also to Record.

Release `0.2.27 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.2.27>`__
====================================================================================

 (`pip <https://pypi.org/project/awkward1/0.2.27/>`__)

* PR `#352 <https://github.com/scikit-hep/awkward-1.0/pull/352>`__: Replace metadata containing cache with just cache (no premature generalization).
* PR `#351 <https://github.com/scikit-hep/awkward-1.0/pull/351>`__: Put the Awkward-in-Pandas feature up for a vote, citing `#350 <https://github.com/scikit-hep/awkward-1.0/issues/350>`__.
* PR `#349 <https://github.com/scikit-hep/awkward-1.0/pull/349>`__: Fix Python SyntaxWarning.
* PR `#345 <https://github.com/scikit-hep/awkward-1.0/pull/345>`__: Prepare the Python Layer for the CUDA Kernels, and add Docker Images for CI.

Release `0.2.26 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.2.26>`__
====================================================================================

 (`pip <https://pypi.org/project/awkward1/0.2.26/>`__)

* PR `#348 <https://github.com/scikit-hep/awkward-1.0/pull/348>`__: Added form_key (optional string) to all Forms.
* PR `#347 <https://github.com/scikit-hep/awkward-1.0/pull/347>`__: Remove redundant line from reducers kernel.
* PR `#314 <https://github.com/scikit-hep/awkward-1.0/pull/314>`__: Implement axis_wrap_if_negative.
* PR `#344 <https://github.com/scikit-hep/awkward-1.0/pull/344>`__: The ak.Array/ak.Record/ak.ArrayBuilder repr quotes keys that are not good identifiers.
* PR `#343 <https://github.com/scikit-hep/awkward-1.0/pull/343>`__: Convert Arrow <--> Parquet, and hence Awkward <--> Parquet.
* PR `#340 <https://github.com/scikit-hep/awkward-1.0/pull/340>`__: Convert 64-bit unsigned 32-bit Awkward arrays into 32-bit Arrow arrays if their indexes are small enough.
* PR `#339 <https://github.com/scikit-hep/awkward-1.0/pull/339>`__: Adds a high-level interface to sorting (ak.sort) (**also:** `#304 <https://github.com/scikit-hep/awkward-1.0/issues/304>`__)
* PR `#338 <https://github.com/scikit-hep/awkward-1.0/pull/338>`__: Renamed keeplayout → keep_layout in ak.{from,to}_awkward0.
* PR `#337 <https://github.com/scikit-hep/awkward-1.0/pull/337>`__: Try to fully resolve the NumPy format string issues. (**also:** `#333 <https://github.com/scikit-hep/awkward-1.0/issues/333>`__)
* PR `#330 <https://github.com/scikit-hep/awkward-1.0/pull/330>`__: Attach docstrings to newly created highlevel arrays.

Release `0.2.25 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.2.25>`__
====================================================================================

 (`pip <https://pypi.org/project/awkward1/0.2.25/>`__)

* PR `#327 <https://github.com/scikit-hep/awkward-1.0/pull/327>`__: Lookahead assignment Python code generation.
* PR `#326 <https://github.com/scikit-hep/awkward-1.0/pull/326>`__: Fix a lot of warnings that have recently been introduced.
* PR `#322 <https://github.com/scikit-hep/awkward-1.0/pull/322>`__: Revised getitem operation for masked jagged indexers.
* PR `#325 <https://github.com/scikit-hep/awkward-1.0/pull/325>`__: These tests pass if you've pip installed awkward1, but sometimes you need to work from the localbuild directory.
* PR `#293 <https://github.com/scikit-hep/awkward-1.0/pull/293>`__: Separation of cuda-kernels and memory_trackers implementation.
* PR `#324 <https://github.com/scikit-hep/awkward-1.0/pull/324>`__: Fix an error in partitionedarray: array['field', 10].
* PR `#323 <https://github.com/scikit-hep/awkward-1.0/pull/323>`__: Fixed Python generation.
* PR `#319 <https://github.com/scikit-hep/awkward-1.0/pull/319>`__: Update doctest dummy yaml to test documentation.
* PR `#317 <https://github.com/scikit-hep/awkward-1.0/pull/317>`__: Remove redundant cpu kernels from operations.h.
* PR `#261 <https://github.com/scikit-hep/awkward-1.0/pull/261>`__: replace carry with indexedarray.
* PR `#306 <https://github.com/scikit-hep/awkward-1.0/pull/306>`__: Document interfaces of functions in sorting.cpp.
* PR `#299 <https://github.com/scikit-hep/awkward-1.0/pull/299>`__: This PR moves all the template kernels from utils to kernels.
* PR `#168 <https://github.com/scikit-hep/awkward-1.0/pull/168>`__: sort and argsort operations applied in axis.
* PR `#295 <https://github.com/scikit-hep/awkward-1.0/pull/295>`__: Update contributing.md.
* PR `#298 <https://github.com/scikit-hep/awkward-1.0/pull/298>`__: Touch up the kernels documentation.

Release `0.2.24 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.2.24>`__
====================================================================================

 (`pip <https://pypi.org/project/awkward1/0.2.24/>`__)

* PR `#270 <https://github.com/scikit-hep/awkward-1.0/pull/270>`__: Update the minimum Numba version to 0.50 when that becomes available. (**also:** `numba/numba#5717 <https://github.com/numba/numba/issues/5717>`__)
* PR `#296 <https://github.com/scikit-hep/awkward-1.0/pull/296>`__: Do not allow dynamic sized arrays in cpu-kernels.
* PR `#269 <https://github.com/scikit-hep/awkward-1.0/pull/269>`__: Generate Python code for CPU kernels.
* PR `#294 <https://github.com/scikit-hep/awkward-1.0/pull/294>`__: Fix some typos in cpu-kernels.
* PR `#292 <https://github.com/scikit-hep/awkward-1.0/pull/292>`__: Fix some typos in cpu-kernels.

Release `0.2.23 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.2.23>`__
====================================================================================

 (`pip <https://pypi.org/project/awkward1/0.2.23/>`__)

* PR `#290 <https://github.com/scikit-hep/awkward-1.0/pull/290>`__: Bug-fixes for HATS.
* PR `#288 <https://github.com/scikit-hep/awkward-1.0/pull/288>`__: fixes `#286 <https://github.com/scikit-hep/awkward-1.0/issues/286>`__ broadcast single value with field.
* PR `#287 <https://github.com/scikit-hep/awkward-1.0/pull/287>`__: docs: add nikoladze as a contributor.
* PR `#283 <https://github.com/scikit-hep/awkward-1.0/pull/283>`__: Replace std::vector with C style code.
* PR `#282 <https://github.com/scikit-hep/awkward-1.0/pull/282>`__: Remove redundant includes from cpu-kernels.

Release `0.2.22 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.2.22>`__
====================================================================================

 (`pip <https://pypi.org/project/awkward1/0.2.22/>`__)

* PR `#281 <https://github.com/scikit-hep/awkward-1.0/pull/281>`__: Revert to static linking libawkward.so because it broke the wheel-deployment.

Release `0.2.21 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.2.21>`__
====================================================================================

 (`pip <https://pypi.org/project/awkward1/0.2.21/>`__)

* PR `#280 <https://github.com/scikit-hep/awkward-1.0/pull/280>`__: Fix ak.pandas.dfs function for simple rows.
* PR `#279 <https://github.com/scikit-hep/awkward-1.0/pull/279>`__: Fix ArrayBuilder's access to type in __repr__.
* PR `#266 <https://github.com/scikit-hep/awkward-1.0/pull/266>`__: Running black and flake8 on the Python codebase.
* PR `#265 <https://github.com/scikit-hep/awkward-1.0/pull/265>`__: Fixes `#264 <https://github.com/scikit-hep/awkward-1.0/issues/264>`__, reductions at axis=N inside empty lists at axis=N-1.

Release `0.2.20 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.2.20>`__
====================================================================================

 (`pip <https://pypi.org/project/awkward1/0.2.20/>`__)

* PR `#263 <https://github.com/scikit-hep/awkward-1.0/pull/263>`__: Continue from `#224 <https://github.com/scikit-hep/awkward-1.0/issues/224>`__: fromarrow and toarrow.
* PR `#260 <https://github.com/scikit-hep/awkward-1.0/pull/260>`__: docs: add gordonwatts as a contributor.
* PR `#258 <https://github.com/scikit-hep/awkward-1.0/pull/258>`__: docs: add jpata as a contributor.
* PR `#259 <https://github.com/scikit-hep/awkward-1.0/pull/259>`__: docs: add martindurant as a contributor.
* PR `#257 <https://github.com/scikit-hep/awkward-1.0/pull/257>`__: docs: add douglasdavis as a contributor.
* PR `#256 <https://github.com/scikit-hep/awkward-1.0/pull/256>`__: docs: add bfis as a contributor.
* PR `#255 <https://github.com/scikit-hep/awkward-1.0/pull/255>`__: docs: add benkrikler as a contributor.
* PR `#254 <https://github.com/scikit-hep/awkward-1.0/pull/254>`__: docs: add Jayd-1234 as a contributor.
* PR `#253 <https://github.com/scikit-hep/awkward-1.0/pull/253>`__: docs: add guitargeek as a contributor.
* PR `#252 <https://github.com/scikit-hep/awkward-1.0/pull/252>`__: docs: add mhedges as a contributor.
* PR `#251 <https://github.com/scikit-hep/awkward-1.0/pull/251>`__: docs: add masonproffitt as a contributor.
* PR `#250 <https://github.com/scikit-hep/awkward-1.0/pull/250>`__: docs: add EscottC as a contributor.
* PR `#249 <https://github.com/scikit-hep/awkward-1.0/pull/249>`__: docs: add glass-ships as a contributor.
* PR `#248 <https://github.com/scikit-hep/awkward-1.0/pull/248>`__: docs: add veprbl as a contributor.
* PR `#247 <https://github.com/scikit-hep/awkward-1.0/pull/247>`__: docs: add Ellipse0934 as a contributor.
* PR `#246 <https://github.com/scikit-hep/awkward-1.0/pull/246>`__: docs: add trickarcher as a contributor.
* PR `#245 <https://github.com/scikit-hep/awkward-1.0/pull/245>`__: docs: add reikdas as a contributor.
* PR `#244 <https://github.com/scikit-hep/awkward-1.0/pull/244>`__: docs: add henryiii as a contributor.
* PR `#242 <https://github.com/scikit-hep/awkward-1.0/pull/242>`__: docs: add lgray as a contributor.
* PR `#241 <https://github.com/scikit-hep/awkward-1.0/pull/241>`__: docs: add ianna as a contributor.
* PR `#240 <https://github.com/scikit-hep/awkward-1.0/pull/240>`__: docs: add nsmith as a contributor.
* PR `#239 <https://github.com/scikit-hep/awkward-1.0/pull/239>`__: docs: add jpivarski as a contributor.

Release `0.2.19 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.2.19>`__
====================================================================================

 (`pip <https://pypi.org/project/awkward1/0.2.19/>`__)

* PR `#237 <https://github.com/scikit-hep/awkward-1.0/pull/237>`__: Keep writing those tutorials.
* PR `#236 <https://github.com/scikit-hep/awkward-1.0/pull/236>`__: Writing tutorials 2.
* PR `#229 <https://github.com/scikit-hep/awkward-1.0/pull/229>`__: Update to JupyterBook's new Sphinx-based build system.
* PR `#234 <https://github.com/scikit-hep/awkward-1.0/pull/234>`__: Working on `#230 <https://github.com/scikit-hep/awkward-1.0/issues/230>`__ segfault. (**also:** `#233 <https://github.com/scikit-hep/awkward-1.0/issues/233>`__)
* PR `#232 <https://github.com/scikit-hep/awkward-1.0/pull/232>`__: Fix bug in IndexForm possible types. (**also:** `#231 <https://github.com/scikit-hep/awkward-1.0/issues/231>`__)

Release `0.2.18 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.2.18>`__
====================================================================================

* PR `#228 <https://github.com/scikit-hep/awkward-1.0/pull/228>`__: Prepare the second Coffea demo.
* PR `#227 <https://github.com/scikit-hep/awkward-1.0/pull/227>`__: Write CONTRIBUTING.md. (**also:** `#219 <https://github.com/scikit-hep/awkward-1.0/issues/219>`__)

Release `0.2.17 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.2.17>`__
====================================================================================

* PR `#216 <https://github.com/scikit-hep/awkward-1.0/pull/216>`__: VirtualArray, which loads its data on demand and interacts with a cache.
* PR `#217 <https://github.com/scikit-hep/awkward-1.0/pull/217>`__: libawkward: pin vtables to the library.
* PR `#223 <https://github.com/scikit-hep/awkward-1.0/pull/223>`__: Fixed `#222 <https://github.com/scikit-hep/awkward-1.0/issues/222>`__. It failed to initialize starts and stops at the end of its output array, leaving uninitialized junk.
* PR `#220 <https://github.com/scikit-hep/awkward-1.0/pull/220>`__: Fixed localbuild.py, args now correctly parse.

Release `0.2.16 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.2.16>`__
====================================================================================

* PR `#218 <https://github.com/scikit-hep/awkward-1.0/pull/218>`__: Fix a segfault I suspected but couldn't reproduce. (**also:** `#212 <https://github.com/scikit-hep/awkward-1.0/issues/212>`__)

Release `0.2.15 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.2.15>`__
====================================================================================

* PR `#212 <https://github.com/scikit-hep/awkward-1.0/pull/212>`__: PartitionedArray, which only applies to the root of a structure.

Release `0.2.14 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.2.14>`__
====================================================================================

* PR `#189 <https://github.com/scikit-hep/awkward-1.0/pull/189>`__: Update to Numba 0.49 and make that the minimal version.

Release `0.2.13 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.2.13>`__
====================================================================================

* PR `#209 <https://github.com/scikit-hep/awkward-1.0/pull/209>`__: Try again on visibility and also ensure -frtti. (**also:** `#211 <https://github.com/scikit-hep/awkward-1.0/issues/211>`__)

Release `0.2.12 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.2.12>`__
====================================================================================

* PR `#208 <https://github.com/scikit-hep/awkward-1.0/pull/208>`__: Fixes `#207 <https://github.com/scikit-hep/awkward-1.0/issues/207>`__ (missing files in tarball)

Release `0.2.11 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.2.11>`__
====================================================================================

* PR `#206 <https://github.com/scikit-hep/awkward-1.0/pull/206>`__: This should fix `#205 <https://github.com/scikit-hep/awkward-1.0/issues/205>`__.

Release `0.2.10 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.2.10>`__
====================================================================================

 (`pip <https://pypi.org/project/awkward1/0.2.10/>`__)

* PR `#198 <https://github.com/scikit-hep/awkward-1.0/pull/198>`__: Wrote draft of tutorial for EIC.
* PR `#201 <https://github.com/scikit-hep/awkward-1.0/pull/201>`__: Rename ak.choose and ak.cross.
* PR `#197 <https://github.com/scikit-hep/awkward-1.0/pull/197>`__: All reference documentation is done.
* PR `#196 <https://github.com/scikit-hep/awkward-1.0/pull/196>`__: Keep writing those Python docs.
* PR `#195 <https://github.com/scikit-hep/awkward-1.0/pull/195>`__: Keep writing those Python docs.
* PR `#194 <https://github.com/scikit-hep/awkward-1.0/pull/194>`__: Changed some names to add underscores.
* PR `#190 <https://github.com/scikit-hep/awkward-1.0/pull/190>`__: Keep writing those Python docs.
* PR `#188 <https://github.com/scikit-hep/awkward-1.0/pull/188>`__: Really write those Python docstrings this time.

Release `0.2.9 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.2.9>`__
==================================================================================

* PR `#187 <https://github.com/scikit-hep/awkward-1.0/pull/187>`__: Set up for Python documentation (including front page)

Release `0.2.7 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.2.7>`__
==================================================================================

* PR `#185 <https://github.com/scikit-hep/awkward-1.0/pull/185>`__: Start writing doxygen comments in C++.
* PR `#183 <https://github.com/scikit-hep/awkward-1.0/pull/183>`__: Enforce a 79 character maximum on all lines, 72 on docstrings/comments.
* PR `#180 <https://github.com/scikit-hep/awkward-1.0/pull/180>`__: Configure documentation (not content, just how the workflow will work).
* PR `#171 <https://github.com/scikit-hep/awkward-1.0/pull/171>`__: Issues `#166 <https://github.com/scikit-hep/awkward-1.0/issues/166>`__, `#167 <https://github.com/scikit-hep/awkward-1.0/issues/167>`__, `#170 <https://github.com/scikit-hep/awkward-1.0/issues/170>`__ in one PR.
* PR `#155 <https://github.com/scikit-hep/awkward-1.0/pull/155>`__: fillna operation.
* PR `#164 <https://github.com/scikit-hep/awkward-1.0/pull/164>`__: Fixes `#162 <https://github.com/scikit-hep/awkward-1.0/issues/162>`__. Replaces all Raw Pointer Access with wrappers.
* PR `#165 <https://github.com/scikit-hep/awkward-1.0/pull/165>`__: Implement argmin and argmax.

Release `0.2.6 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.2.6>`__
==================================================================================

* PR `#143 <https://github.com/scikit-hep/awkward-1.0/pull/143>`__: ByteMaskedArray, BitMaskedArray, and tomask operation.
* PR `#160 <https://github.com/scikit-hep/awkward-1.0/pull/160>`__: argchoose and choose.

Release `0.2.5 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.2.5>`__
==================================================================================

* PR `#159 <https://github.com/scikit-hep/awkward-1.0/pull/159>`__: Implement 'argcross' and 'cross'.
* PR `#157 <https://github.com/scikit-hep/awkward-1.0/pull/157>`__: ak.Array and ak.Record constructors. Maybe the ``ak.zip`` function.

Release `0.2.4 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.2.4>`__
==================================================================================

* PR `#154 <https://github.com/scikit-hep/awkward-1.0/pull/154>`__: Add the ak.pandas.multiindex(array) function.

Release `0.2.3 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.2.3>`__
==================================================================================

* PR `#152 <https://github.com/scikit-hep/awkward-1.0/pull/152>`__: Finish the count/sizes/num operation and the flatten operation.

Release `0.2.2 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.2.2>`__
==================================================================================

* PR `#132 <https://github.com/scikit-hep/awkward-1.0/pull/132>`__: Merge all the rpad work (`#114 <https://github.com/scikit-hep/awkward-1.0/issues/114>`__) into new environment.
* PR `#151 <https://github.com/scikit-hep/awkward-1.0/pull/151>`__: Issues `#149 <https://github.com/scikit-hep/awkward-1.0/issues/149>`__ and `#150 <https://github.com/scikit-hep/awkward-1.0/issues/150>`__: AttributeErrors and merging ak.behavior.
* PR `#148 <https://github.com/scikit-hep/awkward-1.0/pull/148>`__: isvalid as an operation.

Release `0.2.1 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.2.1>`__
==================================================================================

* PR `#147 <https://github.com/scikit-hep/awkward-1.0/pull/147>`__: RecordArray should use its length_ parameter, regardless of contents_.size()

Release `0.2.0 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.2.0>`__
==================================================================================

 (`pip <https://pypi.org/project/awkward1/0.2.0/>`__)

* PR `#140 <https://github.com/scikit-hep/awkward-1.0/pull/140>`__: Make __typestr__ a behavior, not a data property.

Release `0.1.141 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.141>`__
======================================================================================

* PR `#142 <https://github.com/scikit-hep/awkward-1.0/pull/142>`__: Fix Windows wheel and add auditwheel.

Release `0.1.139 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.139>`__
======================================================================================

* PR `#144 <https://github.com/scikit-hep/awkward-1.0/pull/144>`__: Strings in Numba.

Release `0.1.138 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.138>`__
======================================================================================

*(no pull requests)*

Release `0.1.137 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.137>`__
======================================================================================

* PR `#137 <https://github.com/scikit-hep/awkward-1.0/pull/137>`__: Fix deployment.
* PR `#139 <https://github.com/scikit-hep/awkward-1.0/pull/139>`__: EmptyArrays have float64 type (like NumPy), but are integer arrays when used as a slice (fixing `scikit-hep/awkward-array#236 <https://github.com/scikit-hep/awkward-array/issues/236>`__).
* PR `#135 <https://github.com/scikit-hep/awkward-1.0/pull/135>`__: Convert between Awkward0 and Awkward1.

Release `0.1.133 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.133>`__
======================================================================================

* PR `#133 <https://github.com/scikit-hep/awkward-1.0/pull/133>`__: Fix the tests that are currently skipped due to setidentity segfaults.

Release `0.1.131 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.131>`__
======================================================================================

* PR `#131 <https://github.com/scikit-hep/awkward-1.0/pull/131>`__: Reintroduce Numba "cpointers" test.

Release `0.1.129 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.129>`__
======================================================================================

* PR `#129 <https://github.com/scikit-hep/awkward-1.0/pull/129>`__: Improved build procedure building from setup.py.
* PR `#130 <https://github.com/scikit-hep/awkward-1.0/pull/130>`__: Rename 'FillableArray' to 'ArrayBuilder' and all 'fillable' to 'builder'. (**also:** `#129 <https://github.com/scikit-hep/awkward-1.0/issues/129>`__)

Release `0.1.128 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.128>`__
======================================================================================

 (`pip <https://pypi.org/project/awkward1/0.1.128/>`__)

* PR `#128 <https://github.com/scikit-hep/awkward-1.0/pull/128>`__: Any tweaks that are necessary for Henry's demo.

Release `0.1.122 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.122>`__
======================================================================================

* PR `#118 <https://github.com/scikit-hep/awkward-1.0/pull/118>`__: Replace Numba StructModels with CPointers and check all reference counts.

Release `0.1.121 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.121>`__
======================================================================================

* PR `#121 <https://github.com/scikit-hep/awkward-1.0/pull/121>`__: Better distribution: drop case-sensitive name and ensure that RapidJSON is in the source distribution.

Release `0.1.120 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.120>`__
======================================================================================

* PR `#120 <https://github.com/scikit-hep/awkward-1.0/pull/120>`__: Support the Autograd library in much the same way as NumExpr.

Release `0.1.119 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.119>`__
======================================================================================

* PR `#119 <https://github.com/scikit-hep/awkward-1.0/pull/119>`__: Support NumExpr and add a 'broadcast_arrays' function.

Release `0.1.117 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.117>`__
======================================================================================

* PR `#115 <https://github.com/scikit-hep/awkward-1.0/pull/115>`__: Add reducer operations (with an 'axis' parameter).

Release `0.1.116 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.116>`__
======================================================================================

* PR `#116 <https://github.com/scikit-hep/awkward-1.0/pull/116>`__: Refactor pyawkward.cpp both for compilation speed and so that arrays can be dynamically loaded by dependent Python modules.

Release `0.1.111 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.111>`__
======================================================================================

* PR `#111 <https://github.com/scikit-hep/awkward-1.0/pull/111>`__: Allow Awkward Arrays to be used as slices, including list-type and option-type.

Release `0.1.110 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.110>`__
======================================================================================

* PR `#110 <https://github.com/scikit-hep/awkward-1.0/pull/110>`__: Fix several issues and anything that looks in need of cleanup. (**also:** `#91 <https://github.com/scikit-hep/awkward-1.0/issues/91>`__, `#109 <https://github.com/scikit-hep/awkward-1.0/issues/109>`__, `#109 <https://github.com/scikit-hep/awkward-1.0/issues/109>`__, `#108 <https://github.com/scikit-hep/awkward-1.0/issues/108>`__, `#101 <https://github.com/scikit-hep/awkward-1.0/issues/101>`__)

Release `0.1.107 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.107>`__
======================================================================================

* PR `#107 <https://github.com/scikit-hep/awkward-1.0/pull/107>`__: Assign fields to records (deeply, through all structure).
* PR `#96 <https://github.com/scikit-hep/awkward-1.0/pull/96>`__: Implemented *::count for axis != 0.

Release `0.1.106 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.106>`__
======================================================================================

* PR `#106 <https://github.com/scikit-hep/awkward-1.0/pull/106>`__: Record should hold a pointer to RecordArray, not an instance.

Release `0.1.94 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.94>`__
====================================================================================

* PR `#93 <https://github.com/scikit-hep/awkward-1.0/pull/93>`__: UnionArray::simplify and IndexedOptionArray::simplify, to be used by functions that would naively return uniontypes and optiontypes. (**also:** `#62 <https://github.com/scikit-hep/awkward-1.0/issues/62>`__)
* PR `#105 <https://github.com/scikit-hep/awkward-1.0/pull/105>`__: Turn study/flatten.py into array documentation.
* PR `#98 <https://github.com/scikit-hep/awkward-1.0/pull/98>`__: Bugfix for NumpyArray 32-bit vs 64-bit errors.
* PR `#97 <https://github.com/scikit-hep/awkward-1.0/pull/97>`__: Guard against inspecting __main__ module.
* PR `#83 <https://github.com/scikit-hep/awkward-1.0/pull/83>`__: *::flatten for axis != 0.
* PR `#94 <https://github.com/scikit-hep/awkward-1.0/pull/94>`__: ak.Array.__getattr__ for record fields. (**also:** `#62 <https://github.com/scikit-hep/awkward-1.0/issues/62>`__)

Release `0.1.92 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.92>`__
====================================================================================

 (`pip <https://pypi.org/project/awkward1/0.1.92/>`__)

* PR `#92 <https://github.com/scikit-hep/awkward-1.0/pull/92>`__: Make ak.Array a Pandas DType extension.

Release `0.1.89 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.89>`__
====================================================================================

* PR `#89 <https://github.com/scikit-hep/awkward-1.0/pull/89>`__: Address issues `#88 <https://github.com/scikit-hep/awkward-1.0/issues/88>`__ and `#61 <https://github.com/scikit-hep/awkward-1.0/issues/61>`__ with more complete tests.

Release `0.1.87 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.87>`__
====================================================================================

* PR `#87 <https://github.com/scikit-hep/awkward-1.0/pull/87>`__: Wrote a demo for Numba.

Release `0.1.86 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.86>`__
====================================================================================

* PR `#86 <https://github.com/scikit-hep/awkward-1.0/pull/86>`__: For issue `#60 <https://github.com/scikit-hep/awkward-1.0/issues/60>`__, NEP 13: allow NumPy ufuncs to be called on ak.Array. (**also:** `#66 <https://github.com/scikit-hep/awkward-1.0/issues/66>`__)
* PR `#30 <https://github.com/scikit-hep/awkward-1.0/pull/30>`__: Starting cpp version of PR026 test.

Release `0.1.84 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.84>`__
====================================================================================

* PR `#84 <https://github.com/scikit-hep/awkward-1.0/pull/84>`__: UnionArray: only the basics so that any JSON can be ingested.

Release `0.1.82 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.82>`__
====================================================================================

* PR `#82 <https://github.com/scikit-hep/awkward-1.0/pull/82>`__: Finish up IndexedArray. (**also:** `#52 <https://github.com/scikit-hep/awkward-1.0/issues/52>`__, `#53 <https://github.com/scikit-hep/awkward-1.0/issues/53>`__, `#52 <https://github.com/scikit-hep/awkward-1.0/issues/52>`__, `#53 <https://github.com/scikit-hep/awkward-1.0/issues/53>`__)

Release `0.1.81 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.81>`__
====================================================================================

* PR `#81 <https://github.com/scikit-hep/awkward-1.0/pull/81>`__: Issue `#50 <https://github.com/scikit-hep/awkward-1.0/issues/50>`__: IndexedArray::flatten for axis=0. (**also:** `#51 <https://github.com/scikit-hep/awkward-1.0/issues/51>`__)
* PR `#45 <https://github.com/scikit-hep/awkward-1.0/pull/45>`__: Start flatten implementation and add tests.

Release `0.1.49 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.49>`__
====================================================================================

* PR `#49 <https://github.com/scikit-hep/awkward-1.0/pull/49>`__: Use ak.Array vs ak.Record to distinguish RecordArray from Record.

Release `0.1.47 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.47>`__
====================================================================================

* PR `#48 <https://github.com/scikit-hep/awkward-1.0/pull/48>`__: Reproduce and fix issue `#47 <https://github.com/scikit-hep/awkward-1.0/issues/47>`__.

Release `0.1.46 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.46>`__
====================================================================================

* PR `#46 <https://github.com/scikit-hep/awkward-1.0/pull/46>`__: Start IndexedArray (with and without OptionType).

Release `0.1.43 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.43>`__
====================================================================================

* PR `#44 <https://github.com/scikit-hep/awkward-1.0/pull/44>`__: Reproduce and fix issue `#43 <https://github.com/scikit-hep/awkward-1.0/issues/43>`__.
* PR `#42 <https://github.com/scikit-hep/awkward-1.0/pull/42>`__: Create stubs for the ``flatten`` operation.

Release `0.1.40 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.40>`__
====================================================================================

* PR `#40 <https://github.com/scikit-hep/awkward-1.0/pull/40>`__: Rename Identity/id -> Identities/identities and location -> identity.
* PR `#41 <https://github.com/scikit-hep/awkward-1.0/pull/41>`__: Bring FillableArray::index to regular python interface.

Release `0.1.39 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.39>`__
====================================================================================

* PR `#39 <https://github.com/scikit-hep/awkward-1.0/pull/39>`__: Replacing hanging types with Parameters on each Content.

Release `0.1.38 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.38>`__
====================================================================================

 (`pip <https://pypi.org/project/awkward1/0.1.38/>`__)

* PR `#38 <https://github.com/scikit-hep/awkward-1.0/pull/38>`__: Static methods to make empty arrays of a given type.

Release `0.1.37 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.37>`__
====================================================================================

* PR `#37 <https://github.com/scikit-hep/awkward-1.0/pull/37>`__: Replace 'lookup' and 'reverselookup' with a single property (that acts like 'reverselookup').

Release `0.1.36 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.36>`__
====================================================================================

* PR `#36 <https://github.com/scikit-hep/awkward-1.0/pull/36>`__: Continue working on the Coffea demo.

Release `0.1.33 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.33>`__
====================================================================================

* PR `#33 <https://github.com/scikit-hep/awkward-1.0/pull/33>`__: Creating a demo for Coffea will motivate improvements.

Release `0.1.32 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.32>`__
====================================================================================

* PR `#32 <https://github.com/scikit-hep/awkward-1.0/pull/32>`__: Replace DressedType with parameters on all the Types.

Release `0.1.31 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.31>`__
====================================================================================

* PR `#31 <https://github.com/scikit-hep/awkward-1.0/pull/31>`__: Now the types need to pass through Numba.

Release `0.1.28 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.28>`__
====================================================================================

* PR `#28 <https://github.com/scikit-hep/awkward-1.0/pull/28>`__: Start the high-level layer: awkward.Array.

Release `0.1.26 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.26>`__
====================================================================================

* PR `#26 <https://github.com/scikit-hep/awkward-1.0/pull/26>`__: Add RecordArray (and Record) to Numba.

Release `0.1.25 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.25>`__
====================================================================================

 (`pip <https://pypi.org/project/awkward1/0.1.25/>`__)

* PR `#25 <https://github.com/scikit-hep/awkward-1.0/pull/25>`__: Start writing RecordArray (C++ and Fillable, but not Numba).

Release `0.1.24 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.24>`__
====================================================================================

* PR `#24 <https://github.com/scikit-hep/awkward-1.0/pull/24>`__: Start using RegularArray everywhere it needs to be used.

Release `0.1.23 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.23>`__
====================================================================================

* PR `#23 <https://github.com/scikit-hep/awkward-1.0/pull/23>`__: Introduce RegularArray for rectilinear blocks of any type of awkward array.

Release `0.1.22 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.22>`__
====================================================================================

* PR `#22 <https://github.com/scikit-hep/awkward-1.0/pull/22>`__: FillableArrays must be usable in Numba.

Release `0.1.21 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.21>`__
====================================================================================

* PR `#21 <https://github.com/scikit-hep/awkward-1.0/pull/21>`__: Create EmptyArray with unknown type.

Release `0.1.20 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.20>`__
====================================================================================

* PR `#20 <https://github.com/scikit-hep/awkward-1.0/pull/20>`__: Support unsigned index type for 32-bit.

Release `0.1.19 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.19>`__
====================================================================================

*(no pull requests)*

Release `0.1.18 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.18>`__
====================================================================================

* PR `#19 <https://github.com/scikit-hep/awkward-1.0/pull/19>`__: Use a JSON library to feed FillableArray.

Release `0.1.17 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.17>`__
====================================================================================

 (`pip <https://pypi.org/project/awkward1/0.1.17/>`__)

*(no pull requests)*

Release `0.1.16 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.16>`__
====================================================================================

*(no pull requests)*

Release `0.1.15 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.15>`__
====================================================================================

* PR `#18 <https://github.com/scikit-hep/awkward-1.0/pull/18>`__: Implement ``Fillable``, which are append-only, non-readable arrays.

Release `0.1.14 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.14>`__
====================================================================================

* PR `#17 <https://github.com/scikit-hep/awkward-1.0/pull/17>`__: Put all array classes in an 'array' directory ('include', 'src', and '_numba').

Release `0.1.13 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.13>`__
====================================================================================

* PR `#16 <https://github.com/scikit-hep/awkward-1.0/pull/16>`__: Finish getitem for RawArray.

Release `0.1.12 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.12>`__
====================================================================================

* PR `#15 <https://github.com/scikit-hep/awkward-1.0/pull/15>`__: Implement all of the getitem cases for NumpyArray/ListArray in Numba.

Release `0.1.11 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.11>`__
====================================================================================

* PR `#14 <https://github.com/scikit-hep/awkward-1.0/pull/14>`__: Finish up getitem: handle all slice types but newaxis.

Release `0.1.10 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.10>`__
====================================================================================

* PR `#13 <https://github.com/scikit-hep/awkward-1.0/pull/13>`__: Error-handling as a struct, rather than just a string.

Release `0.1.9 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.9>`__
==================================================================================

*(no pull requests)*

Release `0.1.8 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.8>`__
==================================================================================

* PR `#12 <https://github.com/scikit-hep/awkward-1.0/pull/12>`__: Access ListArray::getitem in Numba.

Release `0.1.7 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.7>`__
==================================================================================

 (`pip <https://pypi.org/project/awkward1/0.1.7/>`__)

*(no pull requests)*

Release `0.1.6 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.6>`__
==================================================================================

*(no pull requests)*

Release `0.1.5 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.5>`__
==================================================================================

*(no pull requests)*

Release `0.1.4 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.4>`__
==================================================================================

* PR `#11 <https://github.com/scikit-hep/awkward-1.0/pull/11>`__: Implemented ListArray and ListOffsetArray's __getitem__.
* PR `#9 <https://github.com/scikit-hep/awkward-1.0/pull/9>`__: Propagate identity through NumpyArray::getitem.

Release `0.1.3 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.3>`__
==================================================================================

*(no pull requests)*

Release `0.1.2 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.2>`__
==================================================================================

*(no pull requests)*

Release `0.1.1 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.1>`__
==================================================================================

* PR `#8 <https://github.com/scikit-hep/awkward-1.0/pull/8>`__: Deep __getitem__ in C++.
* PR `#7 <https://github.com/scikit-hep/awkward-1.0/pull/7>`__: 32-bit and 64-bit versions of Index, Identifer, and ListOffsetArray (and all future Arrays).
* PR `#6 <https://github.com/scikit-hep/awkward-1.0/pull/6>`__: Iterators, deep iteration, iteration in Python and Numba.
* PR `#5 <https://github.com/scikit-hep/awkward-1.0/pull/5>`__: Numba version of the Identity class.

Release `0.1.0 <https://github.com/scikit-hep/awkward-1.0/releases/tag/0.1.0>`__
==================================================================================

 (`pip <https://pypi.org/project/awkward1/0.1.0/>`__)

* PR `#4 <https://github.com/scikit-hep/awkward-1.0/pull/4>`__: Design an "identity" system, like the surrogate key in PartiQL.
* PR `#3 <https://github.com/scikit-hep/awkward-1.0/pull/3>`__: Develop Numba extensions for NumpyArray and ListOffsetArray.
