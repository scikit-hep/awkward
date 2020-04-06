ak.pandas.dfs
-------------

.. py:function:: ak.pandas.dfs(array, levelname=lambda i: "sub"*i + "entry", anonymous="values")

    :param array: :doc:`_auto/ak.Array` to convert into a Pandas DataFrame that does not contain Awkward Arrays.
    :param levelname: Computes a name for each level of the row index from the number of levels deep.
    :param anonymous: Column name to use if the Array does not contain records; otherwise, column names are derived from record fields.

Converts Awkward data structures into a list of Pandas DataFrames with
`MultiIndex <https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html>`__
rows and columns, *instead of* embedding the Awkward Array inside any DataFrames.

For convenience, you can additionally
`pd.merge <https://pandas.pydata.org/pandas-docs/version/1.0.3/reference/api/pandas.merge.html>`__
the DataFrames into a single DataFrame with the :doc:`ak.pandas.df` function.
