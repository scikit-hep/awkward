ak.forms.Form
-------------

Superclass of type nodes that describe a low-level data type or "form". Unlike
the high-level type (see :doc:`_auto/ak.type`), there is an exact one-to-one
relationship between each :doc:`ak.layout.Content` class (see
`ak.Array.layout <_auto/ak.Array.html#ak-array-layout>`_) and each Form.

Forms are rendered as JSON strings, the same JSON that can be used to construct
them.

The type subclasses are listed below.

   * :doc:`ak.forms.EmptyForm` for :doc:`ak.layout.EmptyArray`
   * :doc:`ak.forms.NumpyForm` for :doc:`ak.layout.NumpyArray`
   * :doc:`ak.forms.RegularForm` for :doc:`ak.layout.RegularArray`
   * :doc:`ak.forms.ListForm` for :doc:`ak.layout.ListArray`
   * :doc:`ak.forms.ListOffsetForm` for :doc:`ak.layout.ListOffsetArray`
   * :doc:`ak.forms.RecordForm` for :doc:`ak.layout.RecordArray`
   * :doc:`ak.forms.IndexedForm` for :doc:`ak.layout.IndexedArray`
   * :doc:`ak.forms.IndexedOptionForm` for :doc:`ak.layout.IndexedOptionArray`
   * :doc:`ak.forms.ByteMaskedForm` for :doc:`ak.layout.ByteMaskedArray`
   * :doc:`ak.forms.BitMaskedForm` for :doc:`ak.layout.BitMaskedArray`
   * :doc:`ak.forms.UnmaskedForm` for :doc:`ak.layout.UnmaskedArray`
   * :doc:`ak.forms.UnionForm` for :doc:`ak.layout.UnionArray`
   * :doc:`ak.forms.VirtualForm` for :doc:`ak.layout.VirtualArray`

All :doc:`ak.forms.Form` instances have the following properties and methods
in common.

ak.forms.Form.__eq__
====================

.. py:method:: ak.forms.Form.__eq__(other)

True if two forms are equal; False otherwise.

ak.forms.Form.__ne__
====================

.. py:method:: ak.forms.Form.__ne__()

True if two forms are not equal; False otherwise.

ak.forms.Form.__repr__
======================

.. py:method:: ak.forms.Form.__repr__()

String representation of the form, which is pretty, non-verbose #ak.forms.Form.tojson.

ak.forms.Form.__getstate__
==========================

.. py:method:: ak.forms.Form.__getstate__()

Forms can be pickled.

ak.forms.Form.__setstate__
==========================

.. py:method:: ak.forms.Form.__setstate__(arg0)

Forms can be pickled.

ak.forms.Form.tojson
====================

.. py:method:: ak.forms.Form.tojson(pretty, verbose)

Converts to a JSON string. If ``pretty`` (bool), it will be multi-line and indented;
if ``verbose``, all fields will be shown, even defaults.

ak.forms.Form.type
==================

.. py:method:: ak.forms.Form.type(typestrs)

The single high-level type associated with this low-level form. Conversion in the
other direction is not unique. ``typestrs`` is a dict of ``__record__`` to type-string
names (see `Custom type names <ak.behavior.html#custom-type-names>`_).

ak.forms.Form.parameters
========================

.. py:attribute:: ak.forms.Form.parameters

Returns the parameters associated with this form.

ak.forms.Form.parameter
=======================

.. py:method:: ak.forms.Form.parameter(key)

Returns the parameter associated with ``key``. (Always returns, possibly None.)
