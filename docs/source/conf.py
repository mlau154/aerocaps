import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
import aerocaps.version

# Project Information
project = 'aerocaps'
copyright = '2024, Matthew G. Lauer'
author = 'Matthew G. Lauer'

# Release Information
release = aerocaps.version.get_major_project_version()
version = aerocaps.version.__version__

# Sphinx Extensions
extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.autosectionlabel',
    'sphinx_design',
]

# autodoc_mock_imports = ['PyQt5']

# Allow Sphinx to get links to individual modules of external Python libraries
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    "pyvista": ("https://docs.pyvista.org/", None),
    "rust_nurbs": ("https://rust-nurbs.readthedocs.io/en/latest/", None),
}

templates_path = ['_templates']

html_theme = 'pydata_sphinx_theme'

# html_context = {
#     'display_github': True,
#     'github_user': 'mlau154',
#     'github_repo': 'pymead',
#     'github_version': 'master',
# }

add_module_names = False

numfig = False

navigation_depth = 2  # For the table of contents

html_static_path = ['_static']

html_css_files = [
    'css/custom.css',
]

html_logo = "_static/aerocaps_logo.png"

autosectionlabel_prefix_document = False
