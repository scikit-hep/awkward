ak.forms.Form
-------------

Superclass of type nodes that describe a low-level data type or "form". Unlike
the high-level type (see :class:`ak.contents.type`), there is an exact one-to-one
relationship between each :class:`ak.contents.Content` class (see
:attr:`ak.Array.layout`) and each Form.

Forms are rendered as JSON strings, the same JSON that can be used to construct
them.

The type subclasses are listed below.

   * :class:`ak.forms.EmptyForm` for :class:`ak.contents.EmptyArray`
   * :class:`ak.forms.NumpyForm` for :class:`ak.contents.NumpyArray`
   * :class:`ak.forms.RegularForm` for :class:`ak.contents.RegularArray`
   * :class:`ak.forms.ListForm` for :class:`ak.contents.ListArray`
   * :class:`ak.forms.ListOffsetForm` for :class:`ak.contents.ListOffsetArray`
   * :class:`ak.forms.RecordForm` for :class:`ak.contents.RecordArray`
   * :class:`ak.forms.IndexedForm` for :class:`ak.contents.IndexedArray`
   * :class:`ak.forms.IndexedOptionForm` for :class:`ak.contents.IndexedOptionArray`
   * :class:`ak.forms.ByteMaskedForm` for :class:`ak.contents.ByteMaskedArray`
   * :class:`ak.forms.BitMaskedForm` for :class:`ak.contents.BitMaskedArray`
   * :class:`ak.forms.UnmaskedForm` for :class:`ak.contents.UnmaskedArray`
   * :class:`ak.forms.UnionForm` for :class:`ak.contents.UnionArray`
   * :class:`ak.forms.VirtualForm` for :class:`ak.contents.VirtualArray`

All :class:`ak.forms.Form` instances have the following properties and methods
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
