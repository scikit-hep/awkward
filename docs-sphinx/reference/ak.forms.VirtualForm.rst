.. py:currentmodule:: ak.forms

ak.forms.VirtualForm
--------------------

The form of a :doc:`ak.layout.VirtualArray`, which has the type of its ``form`` (if known).

In addition to the properties and methods described in :doc:`ak.forms.Form`,
this has the following.

.. py:class:: VirtualForm(form, has_length, has_identities=False, parameters=None)

ak.forms.VirtualForm.__init__
=============================

.. py:method:: VirtualForm.__init__(form, has_length, has_identities=False, parameters=None)

``form`` may be None or a form for the materialized array.

``has_length`` is boolean.

ak.forms.VirtualForm.form
=========================

.. py:attribute:: VirtualForm.form

ak.forms.VirtualForm.has_length
===============================

.. py:attribute:: VirtualForm.has_length

ak.forms.VirtualForm.has_identities
===================================

.. py:attribute:: VirtualForm.has_identities

ak.forms.VirtualForm.parameters
===============================

.. py:attribute:: VirtualForm.parameters
