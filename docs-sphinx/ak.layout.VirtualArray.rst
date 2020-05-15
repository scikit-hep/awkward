ak.layout.VirtualArray
----------------------

A VirtualArray is an array that may be generated later.

It requires a :doc:`ak.layout.ArrayGenerator` or :doc:`ak.layout.SliceGenerator`,
a function with arguments that generates and checks the generated array against
an expected :doc:`ak.forms.Form`.

It can optionally take a :doc:`ak.layout.ArrayCache`, which wraps a MutableMapping
with ``__getitem__`` and ``__setitem__`` methods to store arrays. Without this
cache, the VirtualArray would call its generator every time an array is needed.
With the cache, it first checks to see if the array is already in the cache
(though it is assumed that arrays might get evicted from this cache at any time).

It can optionally be given a ``cache_key`` (str), which is the string it passes
to the ``__getitem__`` of its :doc:`ak.layout.ArrayCache`. This key ought to be
unique in the cache. If a ``cache_key`` is not provided, a string will be generated
that is unique in the currently running process.

VirtualArray has no Apache Arrow equivalent.

Below is a simplified implementation of a VirtualArray class in pure Python
that.

.. code-block:: python

    class VirtualArray(Content):
        def __init__(self, generator, cache=None, cache_key=None):
            assert callable(generator)
            if cache is not None:
                assert isinstance(cache, collections.abc.MutableMapping)
            if cache_key is not None:
                assert isinstance(cache_key, str)
            self.generator = generator
            self.cache = cache
            self.cache_key = cache_key

        @staticmethod
        def random(minlen, choices):
            raise NotImplementedError("FIXME!")
            
        def __len__(self):
            raise NotImplementedError("FIXME!")

        def __getitem__(self, where):
            raise NotImplementedError("FIXME!")

        def __repr__(self):
            raise NotImplementedError("FIXME!")

        def xml(self, indent="", pre="", post=""):
            raise NotImplementedError("FIXME!")

In addition to the properties and methods described in :doc:`ak.layout.Content`,
a VirtualArray has the following.

ak.layout.VirtualArray.__init__
===============================

.. py:method:: ak.layout.VirtualArray.__init__(generator, cache=None, cache_key=None, identities=None, parameters=None)

ak.layout.VirtualArray.generator
================================

.. py:attribute:: ak.layout.VirtualArray.generator

ak.layout.VirtualArray.cache
============================

.. py:attribute:: ak.layout.VirtualArray.cache

ak.layout.VirtualArray.cache_key
================================

.. py:attribute:: ak.layout.VirtualArray.cache_key

ak.layout.VirtualArray.peek_array
=================================

.. py:attribute:: ak.layout.VirtualArray.peek_array

Does not materialize the array; might return None.

ak.layout.VirtualArray.array
============================

.. py:attribute:: ak.layout.VirtualArray.array

Materializes the array if necessary; never returns None.
