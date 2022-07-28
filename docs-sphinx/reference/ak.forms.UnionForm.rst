.. py:currentmodule:: ak.forms

ak.forms.UnionForm
------------------

The form of a :class:`ak.layout.UnionArray`, which has :class:`ak.types.UnionType`.

In addition to the properties and methods described in :class:`ak.forms.Form`,
this has the following.

.. py:class:: UnionForm(tags, index, contents, has_identities=False, parameters=None)

.. _ak.forms.UnionForm.__init__:

.. py:method:: UnionForm.__init__(tags, index, contents, has_identities=False, parameters=None)

The ``contents`` is a list of Forms.

.. _ak.forms.UnionForm.tags:

.. py:attribute:: UnionForm.tags

.. _ak.forms.UnionForm.index:

.. py:attribute:: UnionForm.index

.. _ak.forms.UnionForm.contents:

.. py:attribute:: UnionForm.contents

.. _ak.forms.UnionForm.has_identities:

.. py:attribute:: UnionForm.has_identities

.. _ak.forms.UnionForm.parameters:

.. py:attribute:: UnionForm.parameters

.. _ak.forms.UnionForm.content:

.. py:method:: UnionForm.content(index)
