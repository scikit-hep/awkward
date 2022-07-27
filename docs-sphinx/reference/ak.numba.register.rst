ak.numba.register
-----------------

.. py:function:: ak.numba.register()

Registers :func:`ak.Array`, :func:`ak.Record`, and :func:`ak.ArrayBuilder`
as types that can be used in a function compiled by `Numba <http://numba.pydata.org/>`__.

Users shouldn't need to call this function, as it is an
`entry point <https://numba.pydata.org/numba-doc/latest/extending/entrypoints.html>`__.

(But if Numba doesn't recognize an :func:`ak.Array`, :func:`ak.Record`, or
:func:`ak.ArrayBuilder`, try calling this function anyway!)
