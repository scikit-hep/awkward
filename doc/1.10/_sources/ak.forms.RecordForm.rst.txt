ak.forms.RecordForm
-------------------

The form of a :doc:`ak.layout.RecordArray`, which has :doc:`ak.types.RecordType`.

In addition to the properties and methods described in :doc:`ak.forms.Form`,
this has the following.

ak.forms.RecordForm.__init__
============================

.. py:method:: ak.forms.RecordForm.__init__(contents, has_identities=False, parameters=None)

The ``contents`` is a list of Forms.

ak.forms.RecordForm.contents
============================

.. py:attribute:: ak.forms.RecordForm.contents

ak.forms.RecordForm.has_identities
===================================

.. py:attribute:: ak.forms.RecordForm.has_identities

ak.forms.RecordForm.parameters
===============================

.. py:attribute:: ak.forms.RecordForm.parameters

ak.forms.RecordForm.istuple
===========================

.. py:attribute:: ak.forms.RecordForm.istuple

ak.forms.RecordForm.numfields
=============================

.. py:attribute:: ak.forms.RecordForm.numfields

ak.forms.RecordForm.fieldindex
==============================

.. py:method:: ak.forms.RecordForm.fieldindex(key)

ak.forms.RecordForm.key
=======================

.. py:method:: ak.forms.RecordForm.key(fieldindex)

ak.forms.RecordForm.haskey
==========================

.. py:method:: ak.forms.RecordForm.haskey(key)

ak.forms.RecordForm.keys
========================

.. py:method:: ak.forms.RecordForm.keys()

ak.forms.RecordForm.content
===========================

.. py:method:: ak.forms.RecordForm.content(fieldindex)

.. py:method:: ak.forms.RecordForm.content(key)

ak.forms.RecordForm.items
=========================

.. py:method:: ak.forms.RecordForm.items()

ak.forms.RecordForm.values
==========================

.. py:method:: ak.forms.RecordForm.values()
