# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Pocket Laboratories'
copyright = '2024, Leonardo Bermeo'
author = 'Leonardo Bermeo'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',   # Para generar documentación automáticamente desde los docstrings
    'sphinx.ext.napoleon',  # Para soportar docstrings estilo Google o NumPy
    'sphinx.ext.mathjax',   # Para ecuaciones en LaTeX
    'sphinx.ext.autosummary',
    "sphinx.ext.githubpages",

]

templates_path = ['_templates']
exclude_patterns = []


autosummary_generate = True
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#import solar_theme
html_theme = "sphinx_rtd_theme"
# html_theme_path = [solar_theme.theme_path]

#html_theme = 'sphinx_rtd_theme'
html_static_path = ['/home/leonardo/sharefolder/ProyectoSabatico/ThermalSystem/code/python_code/docs/_static']
add_module_names = False
html_use_modindex = False
import os
import sys

sys.path.insert(0, os.path.abspath('/home/leonardo/sharefolder/ProyectoSabatico/ThermalSystem/code/python_code/'))

sys.path.insert(0, os.path.abspath('/home/leonardo/sharefolder/ProyectoSabatico/ThermalSystem/code/python_code/docs/_static'))