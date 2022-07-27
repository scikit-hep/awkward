.. py:currentmodule:: ak.forms

ak.forms.RecordForm
-------------------

The form of a :doc:`ak.layout.RecordArray`, which has :doc:`ak.types.RecordType`.

In addition to the properties and methods described in :doc:`ak.forms.Form`,
this has the following.

.. py:class:: RecordForm(contents, has_identities=False, parameters=None)

ak.forms.RecordForm.__init__
============================

.. py:method:: RecordForm.__init__(contents, has_identities=False, parameters=None)

The ``contents`` is a list of Forms.

ak.forms.RecordForm.contents
============================

.. py:attribute:: RecordForm.contents

ak.forms.RecordForm.has_identities
===================================

.. py:attribute:: RecordForm.has_identities

ak.forms.RecordForm.parameters
===============================

.. py:attribute:: RecordForm.parameters

ak.forms.RecordForm.istuple
===========================

.. py:attribute:: RecordForm.istuple

ak.forms.RecordForm.numfields
=============================

.. py:attribute:: RecordForm.numfields

ak.forms.RecordForm.fieldindex
==============================

.. py:method:: RecordForm.fieldindex(key)

ak.forms.RecordForm.key
=======================

.. py:method:: RecordForm.key(fieldindex)

ak.forms.RecordForm.haskey
==========================

.. py:method:: RecordForm.haskey(key)

ak.forms.RecordForm.keys
========================

.. py:method:: RecordForm.keys()

ak.forms.RecordForm.content
===========================

.. py:method:: RecordForm.content(fieldindex)

.. py:method:: RecordForm.content(key)

ak.forms.RecordForm.items
=========================

.. py:method:: RecordForm.items()

ak.forms.RecordForm.values
==========================

.. py:method:: RecordForm.values()
