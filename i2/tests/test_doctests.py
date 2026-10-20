"""Tests for i2.doctests: the doctest tools merged onto one parser (#36)."""

import doctest
import importlib
import pkgutil
import re

import pytest

import i2
from i2 import doc_mint
from i2.doctests import (
    assert_wants,
    docstring_to_code,
    non_doctest_lines,
    output_prefix,
    split_text_and_doctests,
)

_doctest_line = re.compile(r"\s*>>>")
_empty_line = re.compile(r"\s*$")


def _non_doctest_lines_by_scan(doc):
    """The line-by-line scanner non_doctest_lines used before #36: the oracle."""
    last_line_was_a_doc_test = False
    for line in doc.splitlines():
        if not _doctest_line.match(line):
            if not last_line_was_a_doc_test:
                yield line
            elif _empty_line.match(line):
                last_line_was_a_doc_test = False
        else:
            last_line_was_a_doc_test = True


def _i2_docstrings():
    for info in pkgutil.walk_packages(i2.__path__, "i2."):
        if any(part in info.name for part in ("scrap", "tests", "examples")):
            continue
        try:
            module = importlib.import_module(info.name)
        except Exception:  # optional dependencies
            continue
        for test in doctest.DocTestFinder().find(module):
            if test.docstring:
                yield test.docstring


def test_non_doctest_lines_matches_the_old_scanner_on_every_i2_docstring():
    docs = list(_i2_docstrings())
    assert len(docs) > 100
    for doc in docs:
        assert list(non_doctest_lines(doc)) == list(_non_doctest_lines_by_scan(doc))


def test_non_doctest_lines_keeps_indentation_and_drops_comment_only_examples():
    doc = "    Text.\n    >>> # just a comment\n\n    More text.\n"
    assert list(non_doctest_lines(doc)) == ["    Text.", "    More text."]


def test_docstring_to_code_runs_as_a_test():
    def add(a, b):
        """Add.

        >>> add(2, 3)
        5
        >>> add('a', 'b')
        'ab'
        """
        return a + b

    code = docstring_to_code(add)
    assert code.startswith("# Add.")
    exec(code, {"add": add})  # the asserts hold
    with pytest.raises(AssertionError):
        exec(code, {"add": lambda a, b: None})


def test_docstring_to_code_for_markdown_and_without_text():
    doc = "Text.\n\n>>> 1 + 1\n2\n"
    assert docstring_to_code(doc, example_callback=output_prefix, text_callback=None) == (
        "1 + 1\n# OUTPUT: 2"
    )
    assert docstring_to_code(doc, example_callback=assert_wants) == (
        "# Text.\nassert (1 + 1) == 2 #"
    )


def test_split_blocks_alternate_text_and_doctests():
    blocks = list(split_text_and_doctests("a\n>>> 1\n1\n\nb\n"))
    assert [isinstance(b, str) for b in blocks] == [True, False, True]


def test_doc_mint_still_exports_the_moved_names():
    for name in ("doctest_string", "doctest_string_print", "non_doctest_lines",
                 "split_text_and_doctests", "DoctestBlock", "ExampleX", "assert_wants"):
        assert getattr(doc_mint, name) is getattr(importlib.import_module("i2.doctests"), name)


def test_non_doctest_lines_on_crlf_matches_lf():
    doc = "a\n>>> x\n1\n\nb\n"
    assert list(non_doctest_lines(doc.replace("\n", "\r\n"))) == list(non_doctest_lines(doc)) == ["a", "b"]


def test_an_example_expecting_an_exception_becomes_a_raises_check():
    code = docstring_to_code(">>> int('x')\nTraceback (most recent call last):\n...\nValueError: x\n")
    exec(code)  # int('x') raises, so the check passes
    with pytest.raises(AssertionError):
        exec(code.replace("int('x')", "int('1')"))
