{% if obj.display %}
   {% if is_own_page %}
{{ obj.id }}
{{ "=" * obj.id | length }}

   {% endif %}
.. py:property:: {% if is_own_page %}{{ obj.id}}{% else %}{{ obj.short_name }}{% endif %}
   {% if obj.annotation %}

   :type: {{ obj.annotation }}
   {% endif %}
   {% for property in obj.properties %}

   :{{ property }}:
   {% endfor %}

   {% if obj.obj["is_experimental"] %}

   .. warning::

      This API is experimental and may change or be removed in any release.
   {% endif %}

   {% if obj.docstring %}

   {{ obj.docstring|process_docstring(obj)|indent(3) }}
   {% endif %}
{% endif %}
