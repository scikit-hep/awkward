.. py:currentmodule:: ak.forms

ak.forms.RecordForm
-------------------

The form of a :class:`ak.layout.RecordArray`, which has :doc:`ak.types.RecordType`.

In addition to the properties and methods described in :class:`ak.forms.Form`,
this has the following.

.. py:class:: RecordForm(contents, has_identities=False, parameters=None)

.. _ak.forms.RecordForm.__init__:

.. py:method:: RecordForm.__init__(contents, has_identities=False, parameters=None)

The ``contents`` is a list of Forms.

.. _ak.forms.RecordForm.contents:

.. py:attribute:: RecordForm.contents

.. _ak.forms.RecordForm.has_identities:

.. py:attribute:: RecordForm.has_identities

.. _ak.forms.RecordForm.parameters:

.. py:attribute:: RecordForm.parameters

.. _ak.forms.RecordForm.istuple:

.. py:attribute:: RecordForm.istuple

.. _ak.forms.RecordForm.numfields:

.. py:attribute:: RecordForm.numfields

.. _ak.forms.RecordForm.fieldindex:

.. py:method:: RecordForm.fieldindex(key)

.. _ak.forms.RecordForm.key:

.. py:method:: RecordForm.key(fieldindex)

.. _ak.forms.RecordForm.haskey:

.. py:method:: RecordForm.haskey(key)

.. _ak.forms.RecordForm.keys:

.. py:method:: RecordForm.keys()

.. _ak.forms.RecordForm.content:

.. py:method:: RecordForm.content(fieldindex)

.. py:method:: RecordForm.content(key)

.. _ak.forms.RecordForm.items:

.. py:method:: RecordForm.items()

.. _ak.forms.RecordForm.values:

.. py:method:: RecordForm.values()
