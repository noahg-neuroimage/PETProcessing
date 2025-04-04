{% extends "python/module.rst" %}
{#
{% if obj.display %}
   {% if is_own_page %}
{{ obj.short_name.str.upper() }}
{{ "=" * obj.short_name | length }}
      {% block subpackages %}
         {% set visible_subpackages = obj.subpackages|selectattr("display")|list %}
         {% if visible_subpackages %}
Subpackages
-----------

.. autoapisummary::

            {% for subpackage in visible_subpackages %}
   {{ subpackage.id }}
            {% endfor %}

.. toctree::
   :numbered:
   :hidden:

            {% for subpackage in visible_subpackages %}
   {{ subpackage.include_path }}
            {% endfor %}


         {% endif %}
      {% endblock %}
   {% endif %}
{% endif %}
#}