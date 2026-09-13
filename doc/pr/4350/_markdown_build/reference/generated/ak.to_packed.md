# ak.to_packed

Defined in [awkward.operations.ak_to_packed](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_to_packed.py) on [line 14](https://github.com/scikit-hep/awkward/blob/eb8319fc07e72e9c77dc6c53b77980ef89c43fe4/src/awkward/operations/ak_to_packed.py#L14).

#### ak.to_packed(array, \*, highlevel=True, behavior=None, attrs=None)

Packs an array’s inner structure and materializes its virtual buffers.

- [`ak.contents.NumpyArray`](sphinx-llm:6f8cd60f60b04ce7a9499f2967197e79#ak.contents.NumpyArray) becomes C-contiguous (if it isn’t already)
- [`ak.contents.RegularArray`](sphinx-llm:c74b5b5ed0864b78bcc7b754cd89cd00#ak.contents.RegularArray) trims unreachable content
- [`ak.contents.ListArray`](sphinx-llm:e2e74f5b6c644d858b90eaa78cd4dedc#ak.contents.ListArray) becomes [`ak.contents.ListOffsetArray`](sphinx-llm:00659f65177e4a1db8f94cec2f25984d#ak.contents.ListOffsetArray), making all list data contiguous
- [`ak.contents.ListOffsetArray`](sphinx-llm:00659f65177e4a1db8f94cec2f25984d#ak.contents.ListOffsetArray) starts at `offsets[0] == 0`, trimming unreachable content
- [`ak.contents.RecordArray`](sphinx-llm:2475335b802d4bf79c938eaf015c0299#ak.contents.RecordArray) trims unreachable contents
- [`ak.contents.IndexedArray`](sphinx-llm:0ec798a3207a4746a193b3e535c37f41#ak.contents.IndexedArray) gets projected
- [`ak.contents.IndexedOptionArray`](sphinx-llm:3f1e6f3c5e154ca39fcd977247e4a221#ak.contents.IndexedOptionArray) remains an [`ak.contents.IndexedOptionArray`](sphinx-llm:3f1e6f3c5e154ca39fcd977247e4a221#ak.contents.IndexedOptionArray) (with simplified `index`)
  if it contains records, becomes [`ak.contents.ByteMaskedArray`](sphinx-llm:98670076b6264905953fbfa4fd1faf70#ak.contents.ByteMaskedArray) otherwise
- [`ak.contents.ByteMaskedArray`](sphinx-llm:98670076b6264905953fbfa4fd1faf70#ak.contents.ByteMaskedArray) becomes an [`ak.contents.IndexedOptionArray`](sphinx-llm:3f1e6f3c5e154ca39fcd977247e4a221#ak.contents.IndexedOptionArray) if it contains records,
  stays a [`ak.contents.ByteMaskedArray`](sphinx-llm:98670076b6264905953fbfa4fd1faf70#ak.contents.ByteMaskedArray) otherwise
- [`ak.contents.BitMaskedArray`](sphinx-llm:a08c35b97c5a4c40b230a9d2289feebd#ak.contents.BitMaskedArray) becomes an [`ak.contents.IndexedOptionArray`](sphinx-llm:3f1e6f3c5e154ca39fcd977247e4a221#ak.contents.IndexedOptionArray) if it contains records,
  stays a [`ak.contents.BitMaskedArray`](sphinx-llm:a08c35b97c5a4c40b230a9d2289feebd#ak.contents.BitMaskedArray) otherwise
- [`ak.contents.UnionArray`](sphinx-llm:dd41ab713e6a4626a50f392463f45ad1#ak.contents.UnionArray) gets projected contents
- [`ak.record.Record`](sphinx-llm:c769a66df3124cdc8442e55ad70f6980#ak.record.Record) becomes a record over a single-item [`ak.contents.RecordArray`](sphinx-llm:2475335b802d4bf79c938eaf015c0299#ak.contents.RecordArray)

Performing these operations will minimize the output size of data sent to
[`ak.to_buffers`](sphinx-llm:0f37278395f642a289396d6393d4d837#ak.to_buffers) (though conversions through Arrow, [`ak.to_arrow`](sphinx-llm:b7375fef0bca4ff1837b125c50ce5ad8#ak.to_arrow) and
[`ak.to_parquet`](sphinx-llm:34bdf9ca41984994beeef0eed40a786b#ak.to_parquet), do not need this because packing is part of that conversion).

See also [`ak.to_buffers`](sphinx-llm:0f37278395f642a289396d6393d4d837#ak.to_buffers).

* **Parameters:**
  * **array** – Array-like data (anything [`ak.to_layout`](sphinx-llm:403123410bfd489c81c0ce600f3e7695#ak.to_layout) recognizes).
  * **highlevel** ([*bool*](https://docs.python.org/3/library/functions.html#bool)) – If True, return an [`ak.Array`](sphinx-llm:c265a53f66b3419cb9df4fa097ec719d#ak.Array); otherwise, return
    a low-level [`ak.contents.Content`](sphinx-llm:bda3fe6771b14645a594aa3bbe64f5f8#ak.contents.Content) subclass.
  * **behavior** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom [`ak.behavior`](sphinx-llm:4c349445204c475eb8c2285d838e2099#ak.behavior) for the output array, if
    high-level.
  * **attrs** (*None* *or* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict)) – Custom attributes for the output array, if
    high-level.
* **Returns:**
  An array with the same type and values as the input,
  with all virtual buffers materialized (see [`ak.materialize`](sphinx-llm:9018c1e8640d4963a7bbd4d1f613ec4c#ak.materialize)) and inner structures packed.

### Examples

```pycon
>>> a = ak.Array([[1, 2, 3], [], [4, 5], [6], [7, 8, 9, 10]])
>>> b = a[::-1]
>>> b.layout
<ListArray len='5'>
    <starts><Index dtype='int64' len='5'>
        [6 5 3 3 0]
    </Index></starts>
    <stops><Index dtype='int64' len='5'>
        [10  6  5  3  3]
    </Index></stops>
    <content><NumpyArray dtype='int64' len='10'>
        [ 1  2  3  4  5  6  7  8  9 10]
    </NumpyArray></content>
</ListArray>
```

```pycon
>>> c = ak.to_packed(b)
>>> c.layout
<ListOffsetArray len='5'>
    <offsets><Index dtype='int64' len='6'>[ 0  4  5  7  7 10]</Index></offsets>
    <content><NumpyArray dtype='int64' len='10'>
        [ 7  8  9 10  6  4  5  1  2  3]
    </NumpyArray></content>
</ListOffsetArray>
```
