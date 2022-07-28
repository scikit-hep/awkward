.. py:currentmodule:: ak.forms

ak.forms.NumpyForm
------------------

The form of a :class:`ak.layout.NumpyArray`, which has :class:`ak.types.PrimitiveType`, possibly wrapped
in one or more layers of :class:`ak.types.RegularType`.

In addition to the properties and methods described in :class:`ak.forms.Form`,
this has the following.

.. py:class:: NumpyForm(inner_shape, itemsize, format, has_identities=False, parameters=None)

.. _ak.forms.NumpyForm.__init__:

.. py:method:: NumpyForm.__init__(inner_shape, itemsize, format, has_identities=False, parameters=None)

The ``inner_shape`` is the ``shape[1:]`` of the corresponding :class:`ak.layout.NumpyArray` (*default* ``[]``).

The ``format`` is the dtype string returned by pybind11, which is platform-dependent. A platform-independent
interpretation of this is ``primitive``. When reading from JSON, either the ``primitive`` or the ``format``
needs to be specified.

The ``itemsize`` is redundant, but included for safety because of the platform-dependence of ``format``.

.. _ak.forms.NumpyForm.inner_shape:

.. py:attribute:: NumpyForm.inner_shape

.. _ak.forms.NumpyForm.itemsize:

.. py:attribute:: NumpyForm.itemsize

.. _ak.forms.NumpyForm.format:

.. py:attribute:: NumpyForm.format

.. _ak.forms.NumpyForm.has_identities:

.. py:attribute:: NumpyForm.has_identities

.. _ak.forms.NumpyForm.parameters:

.. py:attribute:: NumpyForm.parameters
