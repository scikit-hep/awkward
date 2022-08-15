.. py:currentmodule:: ak.forms

ak.forms.RecordForm
-------------------

The form of a :class:`ak.layout.RecordArray`, which has :class:`ak.types.RecordType`.

In addition to the properties and methods described in :class:`ak.forms.Form`,
this has the following.

.. py:class:: RecordForm(contents, has_identities=False, parameters=None)

    .. py:method:: RecordForm.__init__(contents, has_identities=False, parameters=None)
        
        The ``contents`` is a list of Forms.
        
    .. py:attribute:: RecordForm.contents
        
    .. py:attribute:: RecordForm.has_identities
        
    .. py:attribute:: RecordForm.parameters
        
    .. py:attribute:: RecordForm.istuple
        
    .. py:attribute:: RecordForm.numfields
        
    .. py:method:: RecordForm.fieldindex(key)
        
    .. py:method:: RecordForm.key(fieldindex)
        
    .. py:method:: RecordForm.haskey(key)
        
    .. py:method:: RecordForm.keys()
        
    .. py:method:: RecordForm.content(fieldindex)
        
    .. py:method:: RecordForm.content(key)
        
    .. py:method:: RecordForm.items()
        