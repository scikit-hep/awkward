.. py:currentmodule:: ak.forms

ak.forms.UnionForm
------------------

The form of a :doc:`ak.layout.UnionArray`, which has :doc:`ak.types.UnionType`.

In addition to the properties and methods described in :doc:`ak.forms.Form`,
this has the following.

.. py:class:: UnionForm(tags, index, contents, has_identities=False, parameters=None)

ak.forms.UnionForm.__init__
===========================

.. py:method:: UnionForm.__init__(tags, index, contents, has_identities=False, parameters=None)

The ``contents`` is a list of Forms.

ak.forms.UnionForm.tags
=======================

.. py:attribute:: UnionForm.tags

ak.forms.UnionForm.index
========================

.. py:attribute:: UnionForm.index

ak.forms.UnionForm.contents
===========================

.. py:attribute:: UnionForm.contents

ak.forms.UnionForm.has_identities
=================================

.. py:attribute:: UnionForm.has_identities

ak.forms.UnionForm.parameters
=============================

.. py:attribute:: UnionForm.parameters

ak.forms.UnionForm.content
===========================

.. py:method:: UnionForm.content(index)
