# docs/source/conf.py
import os
import sys
sys.path.insert(0, os.path.abspath('../..')) # <-- Змінити на це

project = 'NLP'
copyright = '2025, Matushenko Andriy'
author = 'Matushenko Andriy'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # Автоматично генерує документацію з докстрінгів
    'sphinx.ext.viewcode', # Додає посилання на вихідний код зі сторінок документації
    'sphinx.ext.napoleon', # Дозволяє розуміти докстрінги в стилі Google
]

templates_path = ['_templates']
exclude_patterns = []

language = 'uk'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
