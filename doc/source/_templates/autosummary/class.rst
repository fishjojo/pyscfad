{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:

   {% block attributes %}
   {% if attributes %}

   {% endif %}
   {% endblock %}

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :toctree: methods/

      {% for item in methods %}
         {% if item not in inherited_members and (not item.startswith('_') or item in ['__call__']) %}
           ~{{ name }}.{{ item }}
         {% endif %}
      {%- endfor %}

   {% endif %}
   {% endblock %}
