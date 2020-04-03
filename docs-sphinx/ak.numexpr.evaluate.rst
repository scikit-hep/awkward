ak.numexpr.evaluate
-------------------

.. py:function:: ak.numexpr.evaluate(expression, local_dict=None, global_dict=None, order="K", casting="safe", **kwargs)

See `numexpr.evaluate <https://numexpr.readthedocs.io/projects/NumExpr3/en/latest/api.html#numexpr.evaluate>`__
for a description of each parameter. This function reproduces NumExpr's interface, except that
the expression can contain references to :doc:`_auto/ak.Array` objects as well as NumPy arrays.

The arrays are broadcasted according to rules described in :doc:`_auto/ak.broadcast_arrays`. The
expression applies to the numeric leaves of the data structure and the output maintains that
structure, just as `ak.Array.__array_ufunc__ <_auto/ak.Array.html#ak-array-array-ufunc>`__
preserves structure through NumPy
[universal functions](https://docs.scipy.org/doc/numpy/reference/ufuncs.html).

To do: examples.

See also :doc:`ak.numexpr.re_evaluate`.
