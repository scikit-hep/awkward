ak.pandas.df
------------

.. py:function:: ak.pandas.df(array, how="inner", levelname=lambda i: "sub"*i + "entry", anonymous="values")

    :param array: :doc:`_auto/ak.Array` to convert into a Pandas DataFrame that does not contain Awkward Arrays.
    :param how: Passed to [pd.merge](https://pandas.pydata.org/pandas-docs/version/1.0.3/reference/api/pandas.merge.html) to combine DataFrames for each multiplicity into one DataFrame.
    :param levelname: Computes a name for each level of the row index from the number of levels deep.
    :param anonymous: Column name to use if the Array does not contain records; otherwise, column names are derived from record fields.

Converts Awkward data structures into Pandas
`MultiIndex <https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html>`__
rows and columns *instead of* embedding the Awkward Array in the DataFrame as
a column.

:doc:`_auto/ak.Array` data structures cannot be losslessly converted into a single
DataFrame; different fields in a record structure might have different nested list
lengths, but a DataFrame can have only one index. This function actually calls
:doc:`ak.pandas.dfs` to make as many DataFrames as are needed and then uses
`pd.merge <https://pandas.pydata.org/pandas-docs/version/1.0.3/reference/api/pandas.merge.html>`__
to combine those DataFrames using Pandas's ``how`` parameter.

To do: examples.
