ak.forms.Form
-------------

.. py:currentmodule:: ak.forms

Superclass of type nodes that describe a low-level data type or "form". Unlike
the high-level type (see :func:`ak.type`), there is an exact one-to-one
relationship between each :class:`ak.layout.Content` class (see
`ak.Array.layout <_auto/ak.Array.html#ak-array-layout>`_) and each Form.

Forms are rendered as JSON strings, the same JSON that can be used to construct
them.

The type subclasses are listed below.

   * :class:`ak.forms.EmptyForm` for :class:`ak.layout.EmptyArray`
   * :class:`ak.forms.NumpyForm` for :class:`ak.layout.NumpyArray`
   * :class:`ak.forms.RegularForm` for :class:`ak.layout.RegularArray`
   * :class:`ak.forms.ListForm` for :class:`ak.layout.ListArray`
   * :class:`ak.forms.ListOffsetForm` for :class:`ak.layout.ListOffsetArray`
   * :class:`ak.forms.RecordForm` for :class:`ak.layout.RecordArray`
   * :class:`ak.forms.IndexedForm` for :class:`ak.layout.IndexedArray`
   * :class:`ak.forms.IndexedOptionForm` for :class:`ak.layout.IndexedOptionArray`
   * :class:`ak.forms.ByteMaskedForm` for :class:`ak.layout.ByteMaskedArray`
   * :class:`ak.forms.BitMaskedForm` for :class:`ak.layout.BitMaskedArray`
   * :class:`ak.forms.UnmaskedForm` for :class:`ak.layout.UnmaskedArray`
   * :class:`ak.forms.UnionForm` for :class:`ak.layout.UnionArray`
   * :class:`ak.forms.VirtualForm` for :class:`ak.layout.VirtualArray`

All :class:`ak.forms.Form` instances have the following properties and methods
in common.

.. py:class:: Form

    .. py:method:: Form.__eq__(other)
        
        True if two forms are equal; False otherwise.
        
    .. py:method:: Form.__ne__()
        
        True if two forms are not equal; False otherwise.
        
    .. py:method:: Form.__repr__()
        
        String representation of the form, which is pretty, non-verbose #ak.forms.Form.tojson.
        
    .. py:method:: Form.__getstate__()
        
        Forms can be pickled.
        
    .. py:method:: Form.__setstate__(arg0)
        
        Forms can be pickled.
        
    .. py:method:: Form.tojson(pretty, verbose)
        
        Converts to a JSON string. If ``pretty`` (bool), it will be multi-line and indented;
        if ``verbose``, all fields will be shown, even defaults.
        
    .. py:method:: Form.type(typestrs)
        
        The single high-level type associated with this low-level form. Conversion in the
        other direction is not unique. ``typestrs`` is a dict of ``__record__`` to type-string
        names (see `Custom type names <ak.behavior.html#custom-type-names>`_).
        
    .. py:attribute:: Form.parameters
        
        Returns the parameters associated with this form.
        
Returns the parameter associated with ``key``. (Always returns, possibly None.)
