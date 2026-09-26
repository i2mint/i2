"""Run the README's python examples, in order, in one shared namespace.

The README's examples build on each other (later blocks reuse names defined in
earlier ones), so they are executed cumulatively, as a reader would run them.
"""

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[2] / "README.md"


def _python_blocks():
    if not README.is_file():  # e.g. running from an installed wheel
        return []
    return re.findall(r"```python\n(.*?)```", README.read_text(), re.S)


@pytest.mark.skipif(not README.is_file(), reason="README.md not found")
def test_readme_python_examples_run():
    namespace = {}
    for i, block in enumerate(_python_blocks()):
        exec(compile(block, f"README.md python block {i}", "exec"), namespace)
