.. py:currentmodule:: ak.forms

ak.forms.UnionForm
------------------

The form of a :class:`ak.layout.UnionArray`, which has :class:`ak.types.UnionType`.

In addition to the properties and methods described in :class:`ak.forms.Form`,
this has the following.

.. py:class:: UnionForm(tags, index, contents, has_identities=False, parameters=None)

    .. py:method:: UnionForm.__init__(tags, index, contents, has_identities=False, parameters=None)
        
        The ``contents`` is a list of Forms.
        
    .. py:attribute:: UnionForm.tags
        
    .. py:attribute:: UnionForm.index
        
    .. py:attribute:: UnionForm.contents
        
    .. py:attribute:: UnionForm.has_identities
        
    .. py:attribute:: UnionForm.parameters
        