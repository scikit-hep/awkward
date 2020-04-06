ak.pandas.register
------------------

.. py:function:: ak.pandas.register()

Registers :doc:`_auto/ak.Array` as a column type for `Pandas <https://pandas.pydata.org/>`__
DataFrames and Series.

Users should not need to call this function because it is invoked whenever a Pandas-specific
method on :doc:`_auto/ak.Array` is called.

(But if Pandas doesn't recognize an :doc:`_auto/ak.Array`, try calling this function anyway!)
