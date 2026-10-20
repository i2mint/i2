"""Doctest tools: split a docstring into text and doctests, and transform doctests.

Docstrings carry two kinds of content: prose, and doctest examples. The tools here
take them apart (:func:`split_text_and_doctests`, :func:`non_doctest_lines`) and
turn the examples into other code: ``assert`` statements for ordinary tests
(:func:`assert_wants`) or source followed by its output as comments, for markdown
(:func:`output_prefix`), over a whole object with :func:`doctest_string`.

Everything that reads a docstring goes through one parser, the standard library's
:class:`doctest.DocTestParser`, by way of :func:`split_text_and_doctests`, so that
"what is a doctest" (where its expected output ends, say) is decided in one place.
:func:`docstring_to_code` uses both halves at once: the prose as comments, the
examples transformed, which is what a test module or a markdown page wants.

These used to live in :mod:`i2.doc_mint`, which still re-exports every one of them.
"""

import doctest
import re
from bisect import bisect_left, bisect_right
from collections.abc import Callable
from functools import partial
from inspect import getdoc
from itertools import accumulate, groupby
from textwrap import dedent

MAX_LINE_LENGTH = 72  # https://en.wikipedia.org/wiki/Characters_per_line


def _prefix_lines(s: str, prefix: str = "# ", even_if_empty: bool = False) -> str:
    r"""
    Prefix every line of s with given prefix.

    :param s: String whose lines you want to prefix.
    :param prefix: Desired prefix string. Default is '# ', to have the effect of "commenting out" line
    :param even_if_empty: Whether to prefix empty strings or not.
    :return: A string whose lines have been prefixed.

    >>> _prefix_lines('something to comment out')
    '# something to comment out'
    >>> _prefix_lines('A line you want to prefix', prefix='PREFIX: ')
    'PREFIX: A line you want to prefix'
    >>> print(_prefix_lines('What happens\nif the thing to comment out\nhas multiple lines?'))
    # What happens
    # if the thing to comment out
    # has multiple lines?
    >>> _prefix_lines('')  # see that an empty string is returned as is
    ''
    >>> _prefix_lines('', even_if_empty=True)  # unless you ask for it
    '# '
    """
    if not even_if_empty:
        if len(s) == 0:
            return s
    return "\n".join(map(lambda x: prefix + x, s.split("\n")))


class ExampleX(doctest.Example):
    """doctest.Example eXtended to have more convenient methods"""

    source_prefix = ">>> "
    source_continuation = "... "

    def __init__(
        self,
        source,
        want=None,
        exc_msg=None,
        lineno=0,
        indent=0,
        options=None,
    ):
        # if source is already a doctest.Example instance, use its properties as args
        if isinstance(source, doctest.Example):
            o = source
            source, want, exc_msg, lineno, indent, options = (
                o.source,
                o.want,
                o.exc_msg,
                o.lineno,
                o.indent,
                o.options,
            )
        super().__init__(source, want, exc_msg, lineno, indent, options)

    @property
    def indent_str(self):
        return self.indent * " "

    @property
    def _source_str(self):
        indented_continuation = f"{self.indent_str}{self.source_continuation}"
        t = self.source.replace("\n", f"\n{indented_continuation}")
        # remove the trailing '...' if present
        if t.endswith(indented_continuation):
            t = t[: -len(indented_continuation)]
        return t

    def __str__(self):
        want = self.want
        if self.want:
            want = self.indent_str + want
        return self.indent_str + self.source_prefix + self._source_str + want

    def __repr__(self):
        return str(self)


class DoctestBlock(list):
    """A list that (should) contain doctest Example instances"""

    def __init__(self, seq=()):
        super().__init__(map(ExampleX, seq))

    def __str__(self):
        return "".join(map(str, self))

    def __repr__(self):
        return f"<DoctestBlock with {len(self)} examples>\n" + str(self)


parse_doctest = doctest.DocTestParser().parse
#: The doctest parser's pattern for one example (source and expected output), so that
#: what counts as an example is decided in one place, without the parser's checks.
_EXAMPLE_RE = doctest.DocTestParser._EXAMPLE_RE


def split_text_and_doctests(doc_string: str):
    r"""
    Generates alternating blocks of "text" (string) and "doctest blocks"
    (``DoctestBlock`` instances, which are essentially a list of ``ExampleX`` instances).

    >>> example = '''
    ...     This is to test the doctest splitter.
    ...     Until now, we're in a text block.
    ...     The following is a doctest block:
    ...
    ...     >>> 2 + 3
    ...     5
    ...     >>> t = 5
    ...     >>> tt = 10
    ...
    ...     This is another text block, followed with another doctest block:
    ...
    ...     >>> def foo():
    ...     ...     return 42
    ...     >>> foo()
    ...     42
    ...
    ... '''
    >>>
    >>> blocks = list(split_text_and_doctests(example))

    There are 5 blocks:

    >>> len(blocks)
    5

    The first block is a string, corresponding to explanatory text of the doc string:

    >>> isinstance(blocks[0], str)
    True
    >>> print(blocks[0])
    <BLANKLINE>
    This is to test the doctest splitter.
    Until now, we're in a text block.
    The following is a doctest block:
    <BLANKLINE>
    <BLANKLINE>

    The next block is a ``DoctestBlock`` instance.

    >>> block = blocks[1]
    >>> isinstance(block, DoctestBlock)
    True

    This block has 3 elements (``ExampleX`` instances)

    >>> len(block)
    3

    If you ask for the string representation of this block, you'll get a doctest string:

    >>> str(block)
    '    >>> 2 + 3\n    5\n    >>> t = 5\n    >>> tt = 10\n'
    """

    def is_string(item):
        if isinstance(item, str):
            if item == "":
                return False
            else:
                return True
        else:
            return False

    t = parse_doctest(doc_string)
    g = groupby(t, key=is_string)
    for _is_string, group_data in g:
        if _is_string:
            yield "\n".join(list(group_data))
        else:
            yield DoctestBlock(filter(None, group_data))


comment_strip_p = re.compile(r"(?m)^ *#.*\n?")
doctest_line_p = re.compile(r"\s*>>>")
empty_line = re.compile(r"\s*$")


def non_doctest_lines(doc):
    r"""Generator of lines of the doc string that are not in a doctest scope.

    >>> def _test_func():
    ...     '''Line 1
    ...     Another
    ...     >>> doctest_1
    ...     >>> doctest_2
    ...     line_after_a_doc_test
    ...     another_line_that_is_in_the_doc_test scope
    ...
    ...     But now we're out of a doctest's scope
    ...
    ...     >>> Oh no, another doctest!
    ...     '''
    >>> from inspect import getdoc
    >>>
    >>> list(non_doctest_lines(getdoc(_test_func)))
    ['Line 1', 'Another', "But now we're out of a doctest's scope", '']

    A doctest's scope is the example and its expected output, which ends at a blank
    line; that blank line is part of the scope too. Examples are found by the doctest
    parser's own pattern, the one :func:`split_text_and_doctests` uses, and the text
    lines are returned as they are, indentation included.
    """
    doc = doc.replace("\r\n", "\n").replace("\r", "\n")  # the pattern's newline
    lines = doc.splitlines(keepends=True)
    starts = list(accumulate((len(line) for line in lines), initial=0))
    in_scope = set()
    for m in _EXAMPLE_RE.finditer(doc):  # every example, comment-only ones included
        first, end = bisect_right(starts, m.start()) - 1, bisect_left(starts, m.end())
        in_scope.update(range(first, end))
        if end < len(lines) and empty_line.match(lines[end]):
            in_scope.add(end)  # the blank line that closed the doctest's output
    yield from (
        line.rstrip("\r\n") for i, line in enumerate(lines) if i not in in_scope
    )


def strip_comments(code):
    r"""Remove whole-line ``#`` comments from ``code`` (inline comments are kept).

    >>> strip_comments("# header\nx = 1  # set x\n")
    'x = 1  # set x\n'
    """
    code = str(code)
    return comment_strip_p.sub("", code)


def mk_example_wants_callback(source_want_func: Callable[[str, str], Callable]):
    """Make a ``doctest.Example`` callback from a ``(source, want) -> str`` function.

    The callback returns the example's source untouched when the example expects no
    output.
    """

    def example_wants_callback(example, *args, **kwargs):
        want = example.want.strip()
        if want:
            source = example.source.strip()
            return source_want_func(source, want, *args, **kwargs)
        else:
            return example.source

    return example_wants_callback


def split_line_comments(s):
    """Split a single line into its code and its ``#`` comment (empty if none).

    >>> split_line_comments("f(1)  # a comment")
    ('f(1)  ', ' a comment')
    >>> split_line_comments("f(1)  # a # comment")
    ('f(1)  ', ' a # comment')
    """
    s, _, comment = s.partition("#")
    return s, comment


def _assert_wants(source, want, wrap_func_name=None):
    is_a_multiline = len(source.split("\n")) > 1

    if not is_a_multiline:
        source, comment = split_line_comments(source)
        if wrap_func_name is None:
            t = f"({source}) == {want} #{comment}"
        else:
            t = f"{wrap_func_name}({source}) == {wrap_func_name}({want}) #{comment}"
        if "'" in t and not '"' in t:
            strchr = '"'
            return "assert {t}, {strchr}{t}{strchr}".format(t=t, strchr=strchr)
        elif '"' in t and not "'" in t:
            strchr = "'"
            return "assert {t}, {strchr}{t}{strchr}".format(t=t, strchr=strchr)
        else:
            return f"assert {t}"
    else:  # if you didn't return before
        if wrap_func_name is None:
            return f"actual = {source}\nexpected = {want}\nassert actual == expected"
        else:
            return (
                f"actual = {wrap_func_name}({source})\nexpected = {wrap_func_name}({want})\n"
                "assert actual == expected"
            )


def _output_prefix(source, want, prefix="# OUTPUT: "):
    return source + "\n" + prefix + want + "\n"


output_prefix = mk_example_wants_callback(_output_prefix)
output_prefix.__doc__ = (
    "Render a ``doctest.Example`` as its source followed by a ``# OUTPUT:`` line.\n\n"
    "    >>> import doctest\n"
    "    >>> output_prefix(doctest.Example(source='1 + 1\\n', want='2\\n'))\n"
    "    '1 + 1\\n# OUTPUT: 2\\n'\n"
)
assert_wants = mk_example_wants_callback(_assert_wants)
assert_wants.__doc__ = (
    "Render a ``doctest.Example`` as an ``assert`` comparing its source to its want.\n\n"
    "    >>> import doctest\n"
    "    >>> assert_wants(doctest.Example(source='1 + 1\\n', want='2\\n'))\n"
    "    'assert (1 + 1) == 2 #'\n"
)

# def example_to_doctest_string(source, want):
#     want.replace()
#     return source +


def doctest_string_trans_lines(
    doctest_obj: doctest.DocTest, example_callback=assert_wants
):
    """Yield ``example_callback(example)`` for each example of a ``doctest.DocTest``."""
    for example in doctest_obj.examples:
        yield example_callback(example)


def find_doctests(obj, *, recurse=True) -> list[doctest.DocTest]:
    """The doctests of ``obj`` (a module, class, function...), and, with ``recurse``,
    of its attributes, as :class:`doctest.DocTestFinder` finds them.

    >>> [t.name for t in find_doctests(split_line_comments)]
    ['split_line_comments']
    """
    return doctest.DocTestFinder(verbose=False, recurse=recurse).find(obj)


def _doctest_string_gen(obj, example_callback, recurse=True):
    for doctest_obj in find_doctests(obj, recurse=recurse):
        yield from doctest_string_trans_lines(doctest_obj, example_callback)


def doctest_string(obj, example_callback=assert_wants, recurse=True):
    """
    Extract the doctests found in given object.

    :param obj: Object (module, class, function, etc.) you want to extract doctests from.
    :params output_prefix:
    :param recurse: Whether the process should find doctests in the attributes of the object, recursively.
    :return: A string containing the doctests, with output lines prefixed by '# Output:'
    """
    return "\n".join(_doctest_string_gen(obj, example_callback, recurse=recurse))


doctest_string.for_output_prefix = partial(
    doctest_string, example_callback=output_prefix
)
doctest_string.for_assert_wants = partial(doctest_string, example_callback=assert_wants)


def doctest_string_print(obj, example_callback=assert_wants, recurse=True):
    """
    Extract the doctests found in given object.

    :param obj: Object (module, class, function, etc.) you want to extract doctests from.
    :param recurse: Whether the process should find doctests in the attributes of the object, recursively.
    :return: A string containing the doctests, with output lines prefixed by '# Output:'
    """
    return print(doctest_string(obj, example_callback, recurse=recurse))


def old_doctest_string(
    obj,
    output_prefix="# OUTPUT: ",
    include_attr_without_doctests=False,
    recurse=True,
):
    """
    Extract the doctests found in given object.

    :param obj: Object (module, class, function, etc.) you want to extract doctests from.
    :param output_prefix:
    :param recurse: Whether the process should find doctests in the attributes of the object, recursively.
    :return: A string containing the doctests, with output lines prefixed by '# Output:'
    """
    s = ""
    for rr in find_doctests(obj, recurse=recurse):
        header = f"# {rr.name} "
        header += "#" * max(0, MAX_LINE_LENGTH - len(header)) + "\n"
        ss = ""
        for example in rr.examples:
            want = example.want
            want = want.strip()
            ss += "\n" + example.source + _prefix_lines(want, prefix=output_prefix)
        if include_attr_without_doctests:
            s += header + ss
        elif len(ss) > 0:  # only append this attr if ss is non-empty
            s += header + ss
    return s


def comment_text(text: str, prefix: str = "# ") -> str:
    r"""A text block as comment lines: dedented, blank lines kept blank.

    >>> print(comment_text('\n    Some prose.\n\n    More prose.\n'))
    # Some prose.
    <BLANKLINE>
    # More prose.
    """
    lines = dedent(text).strip("\n").splitlines()
    return "\n".join(prefix + line if line.strip() else "" for line in lines)


def docstring_to_code(
    doc,
    *,
    example_callback: Callable[[doctest.Example], str] = assert_wants,
    text_callback: Callable[[str], str] | None = comment_text,
) -> str:
    """A docstring as code: its prose through ``text_callback``, its doctest examples
    through ``example_callback``, in their order.

    ``doc`` is a docstring or an object with one. With the defaults the prose becomes
    comments and each example an ``assert`` (:func:`assert_wants`), which is a test;
    ``example_callback=output_prefix`` writes each output as a comment under its
    source instead, for a markdown page. ``text_callback=None`` drops the prose.
    With :func:`assert_wants`, an example expecting an exception becomes a check that
    it raises one; an example whose output is printed, not returned, gives an assert
    that fails, as a doctest's want cannot tell the two apart.

    >>> doc = '''
    ...     Add two numbers.
    ...
    ...     >>> 2 + 3
    ...     5
    ...     >>> x = 1
    ... '''
    >>> print(docstring_to_code(doc))
    # Add two numbers.
    assert (2 + 3) == 5 #
    x = 1
    >>> print(docstring_to_code(doc, example_callback=output_prefix, text_callback=None))
    2 + 3
    # OUTPUT: 5
    x = 1
    """
    if not isinstance(doc, str):
        doc = getdoc(doc) or ""

    def parts():
        for block in split_text_and_doctests(doc):
            if isinstance(block, str):
                if text_callback is not None and block.strip():
                    yield text_callback(block)
            else:
                for example in block:
                    if example.exc_msg is not None and example_callback is assert_wants:
                        yield _assert_raises(example)
                    else:
                        yield example_callback(example).rstrip("\n")

    return "\n".join(parts())


def _assert_raises(example: doctest.Example) -> str:
    r"""Code checking that ``example`` raises, for an example expecting an exception.

    >>> print(_assert_raises(doctest.Example('int("x")\n', '', exc_msg='ValueError: x\n')))
    try:
        int("x")
    except Exception:
        pass
    else:
        raise AssertionError('expected an exception from: int("x")')
    """
    source = example.source.strip()
    body = "\n".join("    " + line for line in source.splitlines())
    return (
        f"try:\n{body}\nexcept Exception:\n    pass\n"
        f"else:\n    raise AssertionError({'expected an exception from: ' + source!r})"
    )
