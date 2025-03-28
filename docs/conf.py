# # Configuration file for the Sphinx documentation builder.
#
project = 'PETPAL (Positron Emission Tomography Analysis Library)'
copyright = '2025, Furqan Dar, Bradley Judge, Noah Goldman, Kenan Oestreich'
author = 'Furqan Dar, Bradley Judge, Noah Goldman, Kenan Oestreich'
release = '0.1.0'

language = 'English (US)'

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_title = 'PETPAL'
pygments_style = 'sphinx'

extensions = [
    'autoapi.extension',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
]

exclude_patterns = ['_build']
templates_path = ["_templates"]

source_suffix = '.rst'
master_doc = 'index'

# autoapi configuration
autoapi_type = 'python'
autoapi_dirs = ['../petpal']
autoapi_ignore = ['*cli*']
autoapi_own_page_level = 'function'

# Options: https://sphinx-autoapi.readthedocs.io/en/latest/reference/config.html#customisation-options
autoapi_options = [
    'members',
    'undoc-members',
    # 'inherited-members',
    'private-members',
    'special-members',
    'show-inheritance',
    'show-module-summary'
]

autoapi_python_class_content = 'both'

autoapi_keep_files = True
autoapi_generate_api_docs = True

toc_object_entries_show_parents = "hide"

napoleon_use_ivar = True
napoleon_use_rtype = False

intersphinx_mapping = {
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'numba': ('https://numba.readthedocs.io/en/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}