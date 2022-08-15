ak.numba.register
-----------------

.. py:function:: ak.numba.register()

Registers :class:`ak.Array`, :class:`ak.Record`, and :class:`ak.ArrayBuilder`
as types that can be used in a function compiled by `Numba <http://numba.pydata.org/>`__.

Users shouldn't need to call this function, as it is an
`entry point <https://numba.pydata.org/numba-doc/latest/extending/entrypoints.html>`__.

(But if Numba doesn't recognize an :class:`ak.Array`, :class:`ak.Record`, or
:class:`ak.ArrayBuilder`, try calling this function anyway!)
