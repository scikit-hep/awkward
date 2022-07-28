.. py:currentmodule:: ak.forms

ak.forms.VirtualForm
--------------------

The form of a :class:`ak.layout.VirtualArray`, which has the type of its ``form`` (if known).

In addition to the properties and methods described in :class:`ak.forms.Form`,
this has the following.

.. py:class:: VirtualForm(form, has_length, has_identities=False, parameters=None)

.. _ak.forms.VirtualForm.__init__:

.. py:method:: VirtualForm.__init__(form, has_length, has_identities=False, parameters=None)

``form`` may be None or a form for the materialized array.

``has_length`` is boolean.

.. _ak.forms.VirtualForm.form:

.. py:attribute:: VirtualForm.form

.. _ak.forms.VirtualForm.has_length:

.. py:attribute:: VirtualForm.has_length

.. _ak.forms.VirtualForm.has_identities:

.. py:attribute:: VirtualForm.has_identities

.. _ak.forms.VirtualForm.parameters:

.. py:attribute:: VirtualForm.parameters
