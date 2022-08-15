ak.pandas.register
------------------

.. py:function:: ak.pandas.register()

Registers :class:`ak.Array` as a column type for `Pandas <https://pandas.pydata.org/>`__
DataFrames and Series.

Users should not need to call this function because it is invoked whenever a Pandas-specific
method on :class:`ak.Array` is called.

(But if Pandas doesn't recognize an :class:`ak.Array`, try calling this function anyway!)
