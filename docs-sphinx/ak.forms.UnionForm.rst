ak.forms.UnionForm
------------------

The form of a :doc:`ak.layout.UnionArray`, which has :doc:`ak.types.UnionType`.

In addition to the properties and methods described in :doc:`ak.forms.Form`,
this has the following.

ak.forms.UnionForm.__init__
===========================

.. py:method:: ak.forms.UnionForm.__init__(tags, index, contents, has_identities=False, parameters=None)

The ``contents`` is a list of Forms.

ak.forms.UnionForm.tags
=======================

.. py:attribute:: ak.forms.UnionForm.tags

ak.forms.UnionForm.index
========================

.. py:attribute:: ak.forms.UnionForm.index

ak.forms.UnionForm.contents
===========================

.. py:attribute:: ak.forms.UnionForm.contents

ak.forms.UnionForm.has_identities
=================================

.. py:attribute:: ak.forms.UnionForm.has_identities

ak.forms.UnionForm.parameters
=============================

.. py:attribute:: ak.forms.UnionForm.parameters

ak.forms.UnionForm.content
===========================

.. py:method:: ak.forms.UnionForm.content(index)
