# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
import warnings
import inspect

# Workaround to avoid expanding type aliases. Copied from jax
from typing import ForwardRef
def _do_not_evaluate(
    self, globalns, *args, _evaluate=ForwardRef._evaluate, **kwargs,
):
  if globalns.get('__name__', '').startswith('pyscfad'):
    return self
  return _evaluate(self, globalns, *args, **kwargs)
ForwardRef._evaluate = _do_not_evaluate

# Workaround for class attributes
from sphinx.ext.napoleon.docstring import GoogleDocstring
def _parse_attributes_section(self, section):
    fields = []
    for _name, _type, _desc in self._consume_fields():
        if not _type:
            _type = self._lookup_annotation(_name)
        fields.append((_name, _type, _desc))
    return self._format_fields('Attributes', fields)
GoogleDocstring._parse_attributes_section = _parse_attributes_section


project = "pyscfad"
copyright = "2021-2026, Xing Zhang"
author = "Xing Zhang"

import pyscfad
version = str(pyscfad.__version__)
release = version

language = "en"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
#    "IPython.sphinxext.ipython_directive",
#    "IPython.sphinxext.ipython_console_highlighting",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
#    "numpydoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
#    "nbsphinx",
    "sphinxemoji.sphinxemoji", # emoji
    "myst_nb",
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://scipy.github.io/devdocs/', None),
    'pyscf': ('https://pyscf.org/', None),
}

templates_path = ["_templates"]
source_suffix = ['.rst', '.ipynb', '.md']
source_encoding = "utf-8"
master_doc = "index"
exclude_patterns = [
    'build/html',
    'build/jupyter_execute',
    '*/tutorial/*.md',
]

autosummary_generate = True

napolean_use_rtype = False
napoleon_use_ivar = False

autodoc_typehints_format = "short"
autodoc_typehints = "description"
autodoc_typehints_description_target = "all"
autodoc_type_aliases = {
    'ArrayLike': 'pyscfad.typing.ArrayLike',
}

#always_document_param_types = True
#set_type_checking_flag = True

#numpydoc_show_class_members = False
#numpydoc_show_inherited_class_members = False
#numpydoc_attributes_as_param_list = False

pygments_style = "sphinx"

#nb_execution_timeout = 60
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]
nb_execution_excludepatterns = [
    "*/tutorial/*.ipynb",
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_show_sourcelink = False
html_logo = "_static/pyscfad_logo.svg"
html_favicon = "_static/pyscfad_logo.svg"
html_sourcelink_suffix = ""
html_last_updated_fmt = ""

html_context = {
    "github_user": "fishjojo",
    "github_repo": "pyscfad",
    "github_version": "doc",
    "doc_path": "doc/source",
}

html_static_path = ["_static"]
#html_css_files = ["css/pyscfad.css"]
html_js_files = ["custom-icon.js"]

#if ".dev" in version:
#    switcher_version = "dev"
#else:
#    # only keep major.minor version number to match versions.json
#    switcher_version = ".".join(version.split(".")[:2])

html_theme_options = {
    "logo": {"text": "pyscfad"},
    "use_edit_page_button": True,
    "navbar_align": "left",
    "github_url": "https://github.com/fishjojo/pyscfad",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/pyscfad",
            "icon": "fa-custom fa-pypi",
        },
    ],
    #"switcher": {
    #    "json_url": "_static/version.json",
    #    "version_match": switcher_version,
    #},
    "show_version_warning_banner": True,
    "footer_start": ["copyright"],
    "footer_center": ["sphinx-version"],
}

# based on numpy doc/source/conf.py
def linkcode_resolve(domain, info) -> str | None:
    """
    Determine the URL corresponding to Python object
    """
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            with warnings.catch_warnings():
                # Accessing deprecated objects will generate noisy warnings
                warnings.simplefilter("ignore", FutureWarning)
                obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        fn = inspect.getsourcefile(inspect.unwrap(obj))
    except TypeError:
        try:  # property
            fn = inspect.getsourcefile(inspect.unwrap(obj.fget))
        except (AttributeError, TypeError):
            fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except TypeError:
        try:  # property
            source, lineno = inspect.getsourcelines(obj.fget)
        except (AttributeError, TypeError):
            lineno = None
    except OSError:
        lineno = None

    if lineno:
        linespec = f"#L{lineno}-L{lineno + len(source) - 1}"
    else:
        linespec = ""

    fn = os.path.relpath(fn, start=os.path.dirname(pyscfad.__file__))

    return (
        f"https://github.com/fishjojo/pyscfad/blob/"
        f"doc/pyscfad/{fn}{linespec}"  # direct to the doc branch for now
    )

def rstjinja(app, docname, source) -> None:
    """
    Render our pages as a jinja template for fancy templating goodness.
    """
    # https://www.ericholscher.com/blog/2016/jul/25/integrating-jinja-rst-sphinx/
    # Make sure we're outputting HTML
    if app.builder.format != "html":
        return
    src = source[0]
    rendered = app.builder.templates.render_string(src, app.config.html_context)
    source[0] = rendered


def setup(app) -> None:
    app.connect("source-read", rstjinja)
