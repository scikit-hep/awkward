{% if obj.display %}
   {% if is_own_page %}
{{ obj.id | ak_name }}
{{ "=" * (obj.id | ak_name | length) }}

{{ github_source_link(obj) }}

   {% endif %}
.. py:function:: {% if is_own_page %}{{ obj.id | ak_name }}{% else %}{{ obj.short_name }}{% endif %}{% if obj.type_params %}[{{ obj.type_params }}]{% endif %}({{ obj.args }}){% if obj.return_annotation is not none %} -> {{ obj.return_annotation }}{% endif %}
   {% for (args, return_annotation) in obj.overloads %}

                 {%+ if is_own_page %}{{ obj.id | ak_name }}{% else %}{{ obj.short_name }}{% endif %}({{ args }}){% if return_annotation is not none %} -> {{ return_annotation }}{% endif %}
   {% endfor %}
   {% for property in obj.properties %}

   :{{ property }}:
   {% endfor %}

   {% if obj.docstring %}

   {{ obj.docstring|process_docstring(obj)|indent(3) }}
   {% endif %}
{% endif %}
