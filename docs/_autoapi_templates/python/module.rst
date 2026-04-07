{% if obj.display and not is_internal_module(obj) %}
   {% if is_own_page %}
{{ obj.id | ak_name }}
{{ "=" * (obj.id | ak_name | length) }}

.. py:module:: {{ obj.name }}

      {% if obj.docstring %}
.. autoapi-nested-parse::

   {{ obj.docstring|process_docstring(obj)|indent(3) }}

      {% endif %}

      {% block submodules %}
         {% set visible_subpackages = obj.subpackages|selectattr("display")|list %}
         {% set visible_submodules = obj.submodules|selectattr("display")|list %}
         {% set visible_submodules = (visible_subpackages + visible_submodules)|sort %}
         {% if visible_submodules %}
Submodules
----------

.. autoapisummary::

            {% for submodule in visible_submodules %}
   {{ submodule.id }}
            {% endfor %}


         {% endif %}
      {% endblock %}
      {% block content %}
         {% set visible_children = obj.children|selectattr("display")|list %}
         {% if visible_children %}
            {% set visible_attributes = visible_children|selectattr("type", "equalto", "data")|list %}
            {% if visible_attributes %}
               {% if "attribute" in own_page_types or "show-module-summary" in autoapi_options %}
Attributes
----------

.. autoapisummary::

                  {% for attribute in visible_attributes %}
   {{ attribute.id }}
                  {% endfor %}
               {% endif %}


            {% endif %}
            {% set visible_exceptions = visible_children|selectattr("type", "equalto", "exception")|list %}
            {% if visible_exceptions %}
               {% if "exception" in own_page_types or "show-module-summary" in autoapi_options %}
Exceptions
----------

.. autoapisummary::

                  {% for exception in visible_exceptions %}
   {{ exception.id }}
                  {% endfor %}
               {% endif %}


            {% endif %}
            {% set visible_classes = visible_children|selectattr("type", "equalto", "class")|list %}
            {% if visible_classes %}
               {% if "class" in own_page_types or "show-module-summary" in autoapi_options %}
Classes
-------

.. autoapisummary::

                  {% for klass in visible_classes %}
   {{ klass.id }}
                  {% endfor %}
               {% endif %}


            {% endif %}
            {% set visible_functions = visible_children|selectattr("type", "equalto", "function")|list %}
            {% if visible_functions %}
               {% if "function" in own_page_types or "show-module-summary" in autoapi_options %}
Functions
---------

.. autoapisummary::

                  {% for function in visible_functions %}
   {{ function.id }}
                  {% endfor %}
               {% endif %}


            {% endif %}
            {% set this_page_children = visible_children|rejectattr("type", "in", own_page_types)|list %}
            {% if this_page_children %}
{{ obj.type|title }} Contents
{{ "-" * obj.type|length }}---------

               {% for obj_item in this_page_children %}
{{ obj_item.render()|indent(0) }}
               {% endfor %}
            {% endif %}
         {% endif %}
      {% endblock %}
   {% else %}
.. py:module:: {{ obj.name }}

      {% if obj.docstring %}
   .. autoapi-nested-parse::

      {{ obj.docstring|process_docstring(obj)|indent(6) }}

      {% endif %}
      {% for obj_item in visible_children %}
   {{ obj_item.render()|indent(3) }}
      {% endfor %}
   {% endif %}
{% endif %}
