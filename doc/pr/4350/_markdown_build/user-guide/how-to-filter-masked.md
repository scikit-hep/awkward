# How to filter with arrays containing missing values

```ipython3
import awkward as ak
import numpy as np
```

<a id="how-to-filter-ragged-indexing-with-missing-values"></a>

## Indexing with missing values

In [Building an awkward index](sphinx-llm:56821310b1324922819fa59d7e0a1622#how-to-filter-masked-building-an-awkward-index), we looked building arrays of integers to perform awkward indexing using [`ak.argmin()`](sphinx-llm:836050fe860646e49e64fda769d3ddaa#ak.argmin) and [`ak.argmax()`](sphinx-llm:5588cbb2c2804faf860257487414bbc0#ak.argmax). In particular, the `keepdims` argument of [`ak.argmin()`](sphinx-llm:836050fe860646e49e64fda769d3ddaa#ak.argmin) and [`ak.argmax()`](sphinx-llm:5588cbb2c2804faf860257487414bbc0#ak.argmax) is very useful for creating arrays that can be used to index into the original array. However, reducers such as [`ak.argmax()`](sphinx-llm:5588cbb2c2804faf860257487414bbc0#ak.argmax) behave differently when they are asked to operate upon empty lists.

Let’s first create an array that contains empty sublists:

```ipython3
array = ak.Array(
    [
        [],
        [10, 3, 2, 9],
        [4, 5, 5, 12, 6],
        [],
        [8, 9, -1],
    ]
)
array
```

Awkward reducers accept a `mask_identity` argument, which changes the [`ak.Array.type`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array.type) and the values of the result:

```ipython3
ak.argmax(array, keepdims=True, axis=-1, mask_identity=False)
```

```ipython3
ak.argmax(array, keepdims=True, axis=-1, mask_identity=True)
```

Setting `mask_identity=True` yields the identity value for the reducer instead of `None` when reducing empty lists. From the above examples of [`ak.argmax()`](sphinx-llm:5588cbb2c2804faf860257487414bbc0#ak.argmax), we can see that the identity for the [`ak.argmax()`](sphinx-llm:5588cbb2c2804faf860257487414bbc0#ak.argmax) is `-1`: What happens if we try and use the array produced with `mask_identity=False` to index into `array`?

As discussed in [Indexing with argmin and argmax](sphinx-llm:56821310b1324922819fa59d7e0a1622#how-to-filter-ragged-indexing-with-argmin-and-argmax), we first need to convert *at least* one dimension to a ragged dimension

```ipython3
index = ak.from_regular(
    ak.argmax(array, keepdims=True, axis=-1, mask_identity=False)
)
```

Now, if we try and index into `array` with `index`, it will raise an exception

```ipython3
array[index]
```

```ipythontb
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
Cell In[6], line 1
----> 1 array[index]

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/highlevel.py:1118, in Array.__getitem__(self, where)
    689 def __getitem__(self, where):
    690     """
    691     Args:
    692         where (many types supported; see below): Index of positions to
   (...)   1116     have the same dimension as the array being indexed.
   1117     """
-> 1118     with ak._errors.SlicingErrorContext(self, where):
   1119         # Handle named axis
   1120         (_, ndim) = self._layout.minmax_depth
   1121         named_axis = _get_named_axis(self)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/_errors.py:79, in ErrorContext.__exit__(self, exception_type, exception_value, traceback)
     77     self._slate.__dict__.clear()
     78     # Handle caught exception
---> 79     raise self.decorate_exception(exception_type, exception_value)
     80 else:
     81     # Step out of the way so that another ErrorContext can become primary.
     82     if self.primary() is self:

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/highlevel.py:1126, in Array.__getitem__(self, where)
   1122 where = _normalize_named_slice(named_axis, where, ndim)
   1124 NamedAxis.mapping = named_axis
-> 1126 indexed_layout = prepare_layout(self._layout._getitem(where, NamedAxis))
   1128 if NamedAxis.mapping:
   1129     return ak.operations.ak_with_named_axis._impl(
   1130         indexed_layout,
   1131         named_axis=NamedAxis.mapping,
   (...)   1134         attrs=self._attrs,
   1135     )

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/contents/content.py:659, in Content._getitem(self, where, named_axis)
    656         return out._getitem_at(0)
    658 elif isinstance(where, ak.highlevel.Array):
--> 659     return self._getitem(where.layout, named_axis)
    661 # Convert between nplikes of different backends
    662 elif (
    663     isinstance(where, ak.contents.Content)
    664     and where.backend is not self._backend
    665 ):

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/contents/content.py:739, in Content._getitem(self, where, named_axis)
    733     return self._carry(
    734         Index64.empty(0, self._backend.nplike),
    735         allow_lazy=True,
    736     )
    738 elif isinstance(where, Content):
--> 739     return self._getitem((where,), named_axis)
    741 elif is_sized_iterable(where):
    742     # Do we have an array
    743     nplike = nplike_of_obj(where, default=None)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/contents/content.py:651, in Content._getitem(self, where, named_axis)
    642 named_axis.mapping = _named_axis
    644 next = ak.contents.RegularArray(
    645     this,
    646     this.length,
    647     1,
    648     parameters=None,
    649 )
--> 651 out = next._getitem_next(nextwhere[0], nextwhere[1:], None)
    653 if out.length is not unknown_length and out.length == 0:
    654     return out._getitem_nothing()

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/contents/regulararray.py:768, in RegularArray._getitem_next(self, head, tail, advanced)
    752     assert head.offsets.nplike is nplike
    753     self._maybe_index_error(
    754         self._backend[
    755             "awkward_RegularArray_getitem_jagged_expand",
   (...)    766         slicer=head,
    767     )
--> 768     down = self._content._getitem_next_jagged(
    769         multistarts, multistops, head._content, tail
    770     )
    772     return RegularArray(
    773         down, headlength, self.length, parameters=self._parameters
    774     )
    776 elif isinstance(head, ak.contents.IndexedOptionArray):

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/contents/listoffsetarray.py:451, in ListOffsetArray._getitem_next_jagged(self, slicestarts, slicestops, slicecontent, tail)
    445 def _getitem_next_jagged(
    446     self, slicestarts: Index, slicestops: Index, slicecontent: Content, tail
    447 ) -> Content:
    448     out = ak.contents.ListArray(
    449         self.starts, self.stops, self._content, parameters=self._parameters
    450     )
--> 451     return out._getitem_next_jagged(slicestarts, slicestops, slicecontent, tail)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/contents/listarray.py:588, in ListArray._getitem_next_jagged(self, slicestarts, slicestops, slicecontent, tail)
    577 nextcarry = ak.index.Index64.empty(carrylen, self._backend.nplike)
    579 assert (
    580     outoffsets.nplike is self._backend.nplike
    581     and nextcarry.nplike is self._backend.nplike
   (...)    586     and self._stops.nplike is self._backend.nplike
    587 )
--> 588 self._maybe_index_error(
    589     self._backend[
    590         "awkward_ListArray_getitem_jagged_apply",
    591         outoffsets.dtype.type,
    592         nextcarry.dtype.type,
    593         slicestarts.dtype.type,
    594         slicestops.dtype.type,
    595         sliceindex.dtype.type,
    596         self._starts.dtype.type,
    597         self._stops.dtype.type,
    598     ](
    599         outoffsets.data,
    600         nextcarry.data,
    601         slicestarts.data,
    602         slicestops.data,
    603         slicestarts.length,
    604         sliceindex.data,
    605         sliceindex.length,
    606         self._starts.data,
    607         self._stops.data,
    608         self._content.length,
    609     ),
    610     slicer=ak.contents.ListArray(slicestarts, slicestops, slicecontent),
    611 )
    612 nextcontent = self._content._carry(nextcarry, True)
    613 nexthead, nexttail = ak._slicing.head_tail(tail)

File ~/micromamba/envs/awkward-docs/lib/python3.11/site-packages/awkward/contents/content.py:297, in Content._maybe_index_error(self, error, slicer)
    295 else:
    296     message = self._backend.format_kernel_error(error)
--> 297     raise ak._errors.index_error(self, slicer, message)

IndexError: cannot slice ListArray (of length 5) with [[-1], [0], [3], [-1], [1]]: index out of range while attempting to get index -1 (in compiled code: https://github.com/scikit-hep/awkward/blob/awkward-cpp-56/awkward-cpp/src/cpu-kernels/awkward_ListArray_getitem_jagged_apply.cpp#L43)

This error occurred while attempting to slice

    <Array [[], [10, 3, 2, 9], ..., [], [8, 9, -1]] type='5 * var * int64'>

with

    <Array [[-1], [0], [3], [-1], [1]] type='5 * var * int64'>
```

From the error message, it is clear that for some sublist(s) the index `-1` is out of range. This makes sense; some of our sublists are empty, meaning that there is no valid integer to index into them.

Now let’s look at the result of indexing with `mask_identity=True`.

```ipython3
index = ak.argmax(array, keepdims=True, axis=-1, mask_identity=True)
```

Because it contains an option type, `index` already satisfies rule (2) in [Building an awkward index](sphinx-llm:56821310b1324922819fa59d7e0a1622#how-to-filter-masked-building-an-awkward-index), and we do not need to convert it to a ragged array. We can see that this index succeeds:

```ipython3
array[index]
```

Here, the missing values in the index array correspond to missing values *in the output array*.

## Indexing with missing sublists

Ragged indexing also supports using `None` in place of *empty sublists* within an index. For example, given the following array

```ipython3
array = ak.Array(
    [
        [10, 3, 2, 9],
        [4, 5, 5, 12, 6],
        [],
        [8, 9, -1],
    ]
)
array
```

let’s use build a ragged index to pull out some particular values. Rather than using empty lists, we can use `None` to mask out sublists that we don’t care about:

```ipython3
array[
    [
        [0, 1],
        None,
        [],
        [2],
    ],
]
```

If we compare this with simply providing an empty sublist,

```ipython3
array[
    [
        [0, 1],
        [],
        [],
        [2],
    ],
]
```

we can see that the `None` value introduces an option-type into the final result. `None` values can be used at *any* level in the index array to introduce an option-type at that depth in the result.
