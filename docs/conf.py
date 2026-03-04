import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

project = "slosizer"
copyright = "2025"
author = "OpenAI"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "furo"
html_static_path = ["assets"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
