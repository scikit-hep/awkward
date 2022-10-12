from __future__ import annotations

import numbers
from collections.abc import Sized
from typing import TypeVar

import jax.numpy
import numpy as np

T = TypeVar("T")


### Fast C++ Kernels
def kernel_list_index_to_content_index(
    list_index: np.ndarray, offsets: np.ndarray
) -> np.ndarray:
    content_index = []
    for i in list_index:
        start = offsets[i]
        stop = offsets[i + 1]
        for j in range(start, stop):
            content_index.append(j)
    return np.array(content_index, dtype=np.int64)


### Layout classes
def normalise_slice(item: slice, length: int) -> slice:
    if (start := item.start) is None:
        start = 0
    elif start < 0:
        start = length + start

    if (stop := item.stop) is None:
        stop = length
    elif stop < 0:
        stop = length + stop

    return slice(start, stop, slice.step)


class Content(Sized):
    def ravel(self) -> Content:
        """Flatten this content and its children into a single array.

        Returns:
            The flattend array.
        """
        raise NotImplementedError

    def getitem_one(self, item: int):
        """Return the item at the given index.

        Args:
            item: index of item.

        Returns:
            The indexed item.
        """
        raise NotImplementedError

    def getitem_ragged(self, item: Content) -> Content:
        """Perform a ragged index into this layout.

        Args:
            item: the ragged index array.

        Returns:
            The indexed-array.
        """
        raise NotImplementedError

    def getitem_range(self, start: int, stop: int) -> Content:
        """Return a range slice over this layout.

        Args:
            start: slice start index
            stop: slice end index

        Returns:
            A copy of this layout containing only the items within the given range.
        """
        raise NotImplementedError

    def getitem_index(self, index: np.ndarray) -> Content:
        """Return an index slice over this layout.

        Args:
            index: array of index integers

        Returns:
            A copy of this layout containing only the items indexed by the index array.
        """
        raise NotImplementedError

    def __getitem__(self, item: slice | int | Content):
        if isinstance(item, slice):
            # Primitive array[x:y]
            if item.step is None:
                item = normalise_slice(item, len(self))
                return self.getitem_range(item.start, item.stop)
            # Non-primitive array[x:y:z]
            else:
                index = np.array(item.indices(len(self)), dtype=np.int64)
                return self.getitem_index(index)
        # Primitive array[0]
        elif isinstance(item, int):
            return self.getitem_one(item)
        # Indexing with an integer NumPyArray
        elif isinstance(item, NumPyArray) and item.dtype == np.dtype("int64"):
            return self.getitem_index(item.data)
        # Indexing with a non-NumpyArray content
        elif isinstance(item, Content):
            return self.getitem_ragged(item)
        else:
            raise TypeError(item)


ArrayType = TypeVar("ArrayType", np.ndarray, jax.numpy.DeviceArray)


class NumPyArray(Content):
    def __init__(self, data: ArrayType):
        self._data = data

    @property
    def data(self) -> ArrayType:
        return self._data

    @property
    def dtype(self):
        return self._data.dtype

    def __array__(self) -> ArrayType:
        return self._data

    def __len__(self) -> int:
        return len(self._data)

    def ravel(self):
        return self

    def validate(self):
        assert self._data.ndim == 1

    def getitem_one(self, item: int) -> numbers.Number:
        return self._data[item]

    def getitem_range(self, start: int, stop: int) -> NumPyArray:
        return NumPyArray(self._data[start:stop])

    def getitem_ragged(self, item: NumPyArray) -> NumPyArray:
        raise AssertionError("This should never be reached")

    def getitem_index(self, item: np.ndarray) -> NumPyArray:
        assert item.dtype == np.dtype("int64")
        return NumPyArray(self._data[item])


class ListOffsetArray(Content):
    def __init__(self, offsets: np.ndarray, content: Content):
        self._offsets = offsets
        self._content = content

    def __len__(self):
        return len(self._offsets) - 1

    @property
    def content(self) -> Content:
        return self._content

    @property
    def offsets(self) -> np.ndarray:
        return self._offsets

    @property
    def is_regular(self) -> bool:
        count = np.diff(self._offsets)
        return np.alltrue(count[1:] == count[:-1])

    def ravel(self) -> Content:
        return self._content.getitem_range(self._offsets[0], self._offsets[-1]).ravel()

    def simplify(self) -> ListOffsetArray:
        """Return a new version of this array with trivial (starting at zero) offsets"""
        return ListOffsetArray(
            self._offsets - self._offsets[0],
            self._content.getitem_range(self._offsets[0], self._offsets[-1]),
        )

    def getitem_one(self, item: int) -> Content:
        start = self._offsets[item]
        stop = self._offsets[item + 1]
        return self._content.getitem_range(start, stop)

    def getitem_range(self, start: int, stop: int) -> ListOffsetArray:
        return ListOffsetArray(self._offsets[start : stop + 1], self._content)

    def getitem_index(self, index: np.ndarray) -> ListOffsetArray:
        assert index.dtype == np.dtype("int64")

        # Compute the indices into the content items that appear in the index result
        content_index = kernel_list_index_to_content_index(index, self._offsets)
        new_content = self._content.getitem_index(content_index)

        # Determine the lengths of the indexed sublists
        inner_lengths = np.diff(self._offsets)
        new_inner_lengths = inner_lengths[index]

        # Compute new zero-based offsets from the lengths
        # of the indexed sublists
        new_offsets = np.empty(len(index) + 1, dtype=np.int64)
        new_offsets[0] = 0
        new_offsets[1:] = np.cumsum(new_inner_lengths)

        return ListOffsetArray(new_offsets, new_content)


### Tests
def test_numpy_array():
    array = NumPyArray(np.arange(12))

    assert array[0] == 0

    assert np.asarray(array[1:3]).tolist() == [1, 2]

    ix = NumPyArray(np.array([0, 3, 5], dtype=np.int64))
    assert np.asarray(array[ix]).tolist() == [0, 3, 5]


def test_list_over_numpy():
    content = NumPyArray(np.arange(13))
    array = ListOffsetArray(
        np.array([1, 4, 7, 10, 13], dtype=np.int64),
        content,
    )

    # Can we index with a scalar
    assert array.is_regular
    assert np.asarray(array[1]).tolist() == [4, 5, 6]

    # Can we index with an array
    ix = NumPyArray(np.array([0, 2], dtype=np.int64))
    indexed = array[ix]
    assert np.asarray(indexed.ravel()).tolist() == [1, 2, 3, 7, 8, 9]

    # Can we index with a simple slice
    sliced = array[1:3]
    assert np.asarray(sliced.ravel()).tolist() == [4, 5, 6, 7, 8, 9]

    # Does simplify behave properly
    simplified = array.simplify()
    assert simplified.offsets.tolist() == [0, 3, 6, 9, 12]
    assert np.asarray(simplified.ravel()).tolist() == np.asarray(array.ravel()).tolist()


def test_list_over_list_over_numpy():
    leaf = NumPyArray(np.arange(12))
    leaf_parent = ListOffsetArray(np.array([0, 3, 6, 9, 12], dtype=np.int64), leaf)
    root = ListOffsetArray(np.array([0, 3, 4], dtype=np.int64), leaf_parent)

    assert not root.is_regular
