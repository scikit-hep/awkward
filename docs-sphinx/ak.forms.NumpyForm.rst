ak.forms.NumpyForm
------------------

The form of a :doc:`ak.layout.NumpyArray`, which has :doc:`ak.types.PrimitiveType`, possibly wrapped
in one or more layers of :doc:`ak.types.RegularType`.

In addition to the properties and methods described in :doc:`ak.forms.Form`,
this has the following.

ak.forms.NumpyForm.__init__
===========================

.. py:method:: ak.forms.NumpyForm.__init__(inner_shape, itemsize, format, has_identities=False, parameters=None)

The ``inner_shape`` is the ``shape[1:]`` of the corresponding :doc:`ak.layout.NumpyArray` (*default* ``[]``).

The ``format`` is the dtype string returned by pybind11, which is platform-dependent. A platform-independent
interpretation of this is ``primitive``. When reading from JSON, either the ``primitive`` or the ``format``
needs to be specified.

The ``itemsize`` is redundant, but included for safety because of the platform-dependence of ``format``.

ak.forms.NumpyForm.inner_shape
==============================

.. py:attribute:: ak.forms.NumpyForm.inner_shape

ak.forms.NumpyForm.itemsize
===========================

.. py:attribute:: ak.forms.NumpyForm.itemsize

ak.forms.NumpyForm.format
=========================

.. py:attribute:: ak.forms.NumpyForm.format

ak.forms.NumpyForm.has_identities
=================================

.. py:attribute:: ak.forms.NumpyForm.has_identities

ak.forms.NumpyForm.parameters
=============================

.. py:attribute:: ak.forms.NumpyForm.parameters
